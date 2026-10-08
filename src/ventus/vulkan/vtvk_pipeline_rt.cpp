#include "vtvk_pipeline.h"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "compiler/spirv/nir_spirv.h"
#include "compiler/glsl_types.h"
#include "nir/nir.h"
#include "util/hash_table.h"
#include "util/log.h"
#include "util/ralloc.h"
#include "vk_pipeline.h"
#include "vk_util.h"
#include "vtvk_entrypoints.h"
#include "ventus/compiler/vt_nir.h"
#include "ventus/llvm/ventus_nir_to_llvm.h"

/* X11 platform headers define common tokens such as None/Always as macros.
 * LLVM uses the same names in scoped enums, so clear them before LLVM headers.
 */
#ifdef None
#undef None
#endif
#ifdef Always
#undef Always
#endif
#ifdef Status
#undef Status
#endif
#ifdef True
#undef True
#endif
#ifdef False
#undef False
#endif

#include "llvm/IR/Module.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <unistd.h>

static const struct nir_shader_compiler_options vtvk_nir_options = {};

static const struct spirv_to_nir_options vtvk_spirv_options = {
   .environment = NIR_SPIRV_VULKAN,
   .ubo_addr_format = nir_address_format_vec2_index_32bit_offset,
   .ssbo_addr_format = nir_address_format_vec2_index_32bit_offset,
   .phys_ssbo_addr_format = nir_address_format_64bit_global,
   .push_const_addr_format = nir_address_format_logical,
   .shared_addr_format = nir_address_format_32bit_offset,
   .constant_addr_format = nir_address_format_64bit_global,
};
static bool
vtvk_env_enabled(const char *name)
{
   const char *value = getenv(name);
   if (!value || !value[0])
      return false;
   return strcmp(value, "0") != 0 &&
          strcmp(value, "false") != 0 &&
          strcmp(value, "FALSE") != 0 &&
          strcmp(value, "off") != 0 &&
          strcmp(value, "OFF") != 0;
}

/* This is a pipeline specialization, not a dynamic shader knob: generated
 * code, launch geometry and RTL must agree.  Invalid values intentionally
 * retain the compatibility configuration. */
static uint8_t
vtvk_rt_group_warps(void)
{
   const char *value = getenv("VENTUS_VK_RT_GROUP_WARPS");
   if (!value || !value[0])
      return 1;
   unsigned long n = strtoul(value, NULL, 10);
   return n == 1 || n == 2 || n == 4 || n == 8 ? n : 1;
}

static void
vtvk_configure_rt_group(struct vt_nir_rt_config *config)
{
   config->rt_group_warps = vtvk_rt_group_warps();
   /* Enable the secondary collective entry. Lowering keeps primary rays
    * and candidate resumes on ordinary traversal; software must arrange
    * the secondary cohort before setting the per-trace collective marker. */
   config->enable_group_reorder = true;
}

struct vtvk_rt_execution_profile_config {
   VkVentusRtExecutionProfile profile;
   VkVentusRtStageRole stage_role;
   uint32_t state_descriptor_set;
   uint32_t state_descriptor_binding;
};

static bool
vtvk_rt_stage_role_is_staged_dispatch(VkVentusRtStageRole role)
{
   switch (role) {
   case VK_VENTUS_RT_STAGE_ROLE_PRIMARY:
   case VK_VENTUS_RT_STAGE_ROLE_DIRECT_SHADOW:
   case VK_VENTUS_RT_STAGE_ROLE_REFLECTION_0:
   case VK_VENTUS_RT_STAGE_ROLE_REFLECTION_1:
      return true;
   default:
      return false;
   }
}

/* Execution profiles are deliberately a private pipeline-create pNext.  A
 * shader-side branch cannot establish this contract: the selected lowering
 * changes ABI and dispatch ownership before shader compilation starts. */
static VkResult
vtvk_parse_rt_execution_profile(
   const VkRayTracingPipelineCreateInfoKHR *info,
   struct vtvk_rt_execution_profile_config *out)
{
   *out = {
      .profile = VK_VENTUS_RT_EXECUTION_PROFILE_MEGAKERNEL,
      .stage_role = VK_VENTUS_RT_STAGE_ROLE_NONE,
      .state_descriptor_set = 0,
      .state_descriptor_binding = 0,
   };

   const VkVentusRtExecutionProfileInfo *profile_info = NULL;
   for (const VkBaseInStructure *node =
           (const VkBaseInStructure *)info->pNext;
        node; node = node->pNext) {
      if (node->sType != VK_STRUCTURE_TYPE_VENTUS_RT_EXECUTION_PROFILE_INFO)
         continue;

      if (profile_info) {
         mesa_loge("ventus-vk: duplicate RT execution-profile pNext");
         return VK_ERROR_FEATURE_NOT_PRESENT;
      }
      profile_info = (const VkVentusRtExecutionProfileInfo *)node;
   }

   if (!profile_info)
      return VK_SUCCESS;

   if (profile_info->version != VK_VENTUS_RT_EXECUTION_PROFILE_VERSION_1) {
      mesa_loge("ventus-vk: unsupported RT execution-profile version %u",
                profile_info->version);
      return VK_ERROR_FEATURE_NOT_PRESENT;
   }

   switch (profile_info->profile) {
   case VK_VENTUS_RT_EXECUTION_PROFILE_MEGAKERNEL:
      if (profile_info->stageRole != VK_VENTUS_RT_STAGE_ROLE_NONE ||
          profile_info->stateDescriptorSet != 0 ||
          profile_info->stateDescriptorBinding != 0) {
         mesa_loge("ventus-vk: megakernel profile must not provide staged "
                   "role or path-state descriptor");
         return VK_ERROR_FEATURE_NOT_PRESENT;
      }
      break;
   case VK_VENTUS_RT_EXECUTION_PROFILE_STAGED_DISPATCH:
      if (!vtvk_rt_stage_role_is_staged_dispatch(profile_info->stageRole)) {
         mesa_loge("ventus-vk: staged-dispatch profile requires a fixed stage role");
         return VK_ERROR_FEATURE_NOT_PRESENT;
      }
      if (profile_info->stateDescriptorSet >= VTVK_MAX_BOUND_DESCRIPTOR_SETS ||
          profile_info->stateDescriptorBinding >= VTVK_MAX_DESCRIPTOR_BINDINGS) {
         mesa_loge("ventus-vk: staged-dispatch path-state descriptor set=%u binding=%u "
                   "exceeds %u sets x %u bindings",
                   profile_info->stateDescriptorSet,
                   profile_info->stateDescriptorBinding,
                   VTVK_MAX_BOUND_DESCRIPTOR_SETS,
                   VTVK_MAX_DESCRIPTOR_BINDINGS);
         return VK_ERROR_FEATURE_NOT_PRESENT;
      }
      break;
   default:
      mesa_loge("ventus-vk: unsupported RT execution profile %u",
                (unsigned)profile_info->profile);
      return VK_ERROR_FEATURE_NOT_PRESENT;
   }

   out->profile = profile_info->profile;
   out->stage_role = profile_info->stageRole;
   out->state_descriptor_set = profile_info->stateDescriptorSet;
   out->state_descriptor_binding = profile_info->stateDescriptorBinding;
   return VK_SUCCESS;
}

struct vtvk_rt_nir_stats {
   unsigned trace_ray;
   unsigned load_vulkan_descriptor;
   unsigned vulkan_resource_index;
   unsigned vulkan_resource_reindex;
   unsigned load_deref;
   unsigned store_deref;
   unsigned load_ubo;
   unsigned load_ssbo;
   unsigned image_store;
   unsigned load_ray_world_origin;
   unsigned load_ray_world_direction;
   unsigned load_ray_object_origin;
   unsigned load_ray_object_direction;
   unsigned load_ray_t_min;
   unsigned load_ray_t_max;
   unsigned load_primitive_id;
   unsigned function_temp_vars;
   unsigned shader_call_data_vars;
   unsigned ray_hit_attrib_vars;
};

static void
vtvk_rt_count_variable(nir_variable *var, vtvk_rt_nir_stats *stats)
{
   if (var->data.mode == nir_var_function_temp)
      stats->function_temp_vars++;
   if (var->data.mode == nir_var_shader_call_data)
      stats->shader_call_data_vars++;
   if (var->data.mode == nir_var_ray_hit_attrib)
      stats->ray_hit_attrib_vars++;
}

static void
vtvk_rt_count_intrinsic(nir_intrinsic_instr *intrin, vtvk_rt_nir_stats *stats)
{
   switch (intrin->intrinsic) {
   case nir_intrinsic_trace_ray:
      stats->trace_ray++;
      break;
   case nir_intrinsic_load_vulkan_descriptor:
      stats->load_vulkan_descriptor++;
      break;
   case nir_intrinsic_vulkan_resource_index:
      stats->vulkan_resource_index++;
      break;
   case nir_intrinsic_vulkan_resource_reindex:
      stats->vulkan_resource_reindex++;
      break;
   case nir_intrinsic_load_deref:
      stats->load_deref++;
      break;
   case nir_intrinsic_store_deref:
      stats->store_deref++;
      break;
   case nir_intrinsic_load_ubo:
      stats->load_ubo++;
      break;
   case nir_intrinsic_load_ssbo:
      stats->load_ssbo++;
      break;
   case nir_intrinsic_image_store:
      stats->image_store++;
      break;
   case nir_intrinsic_load_ray_world_origin:
      stats->load_ray_world_origin++;
      break;
   case nir_intrinsic_load_ray_world_direction:
      stats->load_ray_world_direction++;
      break;
   case nir_intrinsic_load_ray_object_origin:
      stats->load_ray_object_origin++;
      break;
   case nir_intrinsic_load_ray_object_direction:
      stats->load_ray_object_direction++;
      break;
   case nir_intrinsic_load_ray_t_min:
      stats->load_ray_t_min++;
      break;
   case nir_intrinsic_load_ray_t_max:
      stats->load_ray_t_max++;
      break;
   case nir_intrinsic_load_primitive_id:
      stats->load_primitive_id++;
      break;
   default:
      break;
   }
}

static void
vtvk_rt_log_nir_stats(nir_shader *shader, const char *checkpoint)
{
   if (!vtvk_env_enabled("VENTUS_VK_NIR_PROBE") &&
       !vtvk_env_enabled("VENTUS_VK_PROBE_LOG"))
      return;

   vtvk_rt_nir_stats stats = {};
   nir_foreach_variable_in_shader(var, shader)
      vtvk_rt_count_variable(var, &stats);

   nir_foreach_function_impl(impl, shader) {
      nir_foreach_variable_in_list(var, &impl->locals)
         vtvk_rt_count_variable(var, &stats);

      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic)
               vtvk_rt_count_intrinsic(nir_instr_as_intrinsic(instr), &stats);
         }
      }
   }

   mesa_logi("ventus-vk: NIR %s stage=%s trace_ray=%u desc_load=%u "
             "res_index=%u res_reindex=%u load_deref=%u store_deref=%u "
             "ubo=%u ssbo=%u image_store=%u ray_world=(%u,%u) "
             "ray_object=(%u,%u) ray_t=(%u,%u) primitive_id=%u "
             "vars={func:%u call:%u hitattr:%u} scratch=%u",
             checkpoint, _mesa_shader_stage_to_string(shader->info.stage),
             stats.trace_ray, stats.load_vulkan_descriptor,
             stats.vulkan_resource_index, stats.vulkan_resource_reindex,
             stats.load_deref, stats.store_deref, stats.load_ubo,
             stats.load_ssbo, stats.image_store,
             stats.load_ray_world_origin, stats.load_ray_world_direction,
             stats.load_ray_object_origin, stats.load_ray_object_direction,
             stats.load_ray_t_min, stats.load_ray_t_max,
             stats.load_primitive_id, stats.function_temp_vars,
             stats.shader_call_data_vars, stats.ray_hit_attrib_vars,
             shader->scratch_size);
}

static void
vtvk_rt_reserve_global_slot_region(nir_shader *shader)
{
   uint32_t slot_end = 0;

   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            const nir_src *slot = NULL;
            if (intrin->intrinsic == nir_intrinsic_ventus_rt_load_slot)
               slot = &intrin->src[0];
            else if (intrin->intrinsic == nir_intrinsic_ventus_rt_store_slot)
               slot = &intrin->src[1];
            else
               continue;

            /* RT ABI accesses are word-granular.  These abstract slots will
             * later be bound to one global CPS frame, so shader-call spills
             * must begin strictly after the highest such word. */
            if (nir_src_is_const(*slot))
               slot_end = MAX2(slot_end, nir_src_as_uint(*slot) + 4);
         }
      }
   }

   shader->scratch_size = MAX2(shader->scratch_size,
                               ALIGN_POT(slot_end, 16));
}

static void
vtvk_rt_finalize_abi(nir_shader *shader,
                     const struct vt_nir_rt_config *config)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   if (!impl)
      return;

   enum vt_rt_shader_layout layout = vt_nir_layout_for_stage(shader->info.stage);
   shader->scratch_size =
      MAX2(shader->scratch_size, config->scratchpad[layout].private_base_bytes);
   if (config->execution_model == VT_RT_EXECUTION_WAVEFRONT_GLOBAL_WORKER)
      vtvk_rt_reserve_global_slot_region(shader);

   nir_shader_gather_info(shader, impl);
}

static void
vtvk_rt_lower_abi_with_probe(nir_shader *shader,
                             const struct vt_nir_rt_config *config)
{
   if (config->execution_model == VT_RT_EXECUTION_WAVEFRONT_GLOBAL_WORKER) {
      /* Global workers have no PDS ABI, but every RT interface must still be
       * lowered to abstract slots before a hit/miss shader is inlined into a
       * raygen root continuation.  The later global pass binds those slots to
       * completion/CPS storage rather than a private launch allocation. */
      vtvk_rt_log_nir_stats(shader, "before global CPS ABI lowering");
      NIR_PASS(_, shader, vt_nir_lower_rt_payload, config);
      NIR_PASS(_, shader, vt_nir_lower_rt_hit_record, config);
      NIR_PASS(_, shader, vt_nir_lower_rt_control, config);
      NIR_PASS(_, shader, vt_nir_lower_shader_record_ptr, config);
      NIR_PASS(_, shader, vt_nir_lower_shader_scratch, config);
      vtvk_rt_finalize_abi(shader, config);
      vtvk_rt_log_nir_stats(shader, "after global CPS ABI lowering");
      return;
   }

   vtvk_rt_log_nir_stats(shader, "before vt_nir_lower_rt_payload");
   NIR_PASS(_, shader, vt_nir_lower_rt_payload, config);
   vtvk_rt_log_nir_stats(shader, "after vt_nir_lower_rt_payload");

   NIR_PASS(_, shader, vt_nir_lower_rt_hit_record, config);
   vtvk_rt_log_nir_stats(shader, "after vt_nir_lower_rt_hit_record");

   NIR_PASS(_, shader, vt_nir_lower_rt_control, config);
   vtvk_rt_log_nir_stats(shader, "after vt_nir_lower_rt_control");

   NIR_PASS(_, shader, vt_nir_lower_shader_record_ptr, config);
   vtvk_rt_log_nir_stats(shader, "after vt_nir_lower_shader_record_ptr");

   NIR_PASS(_, shader, vt_nir_lower_shader_scratch, config);
   vtvk_rt_log_nir_stats(shader, "after vt_nir_lower_shader_scratch");

   vtvk_rt_finalize_abi(shader, config);
   vtvk_rt_log_nir_stats(shader, "after vt_nir_rt_abi_finalize");
}

static std::string
vtvk_shell_quote(const char *text)
{
   std::string out = "'";
   for (const char *p = text; p && *p; p++) {
      if (*p == '\'')
         out += "'\\''";
      else
         out += *p;
   }
   out += "'";
   return out;
}

static std::string
vtvk_rt_artifact_base(void)
{
   const char *dir = getenv("VENTUS_VK_SHADER_ARTIFACT_DIR");
   if (!dir || !dir[0])
      dir = "/tmp";

   std::string base = dir;
   while (base.size() > 1 && base.back() == '/')
      base.pop_back();

   std::string tmpl = base + "/ventus_vk_rt_XXXXXX.ll";
   std::vector<char> writable(tmpl.begin(), tmpl.end());
   writable.push_back('\0');
   int fd = mkstemps(writable.data(), 3);
   if (fd >= 0) {
      close(fd);
      unlink(writable.data());
      std::string ll_path = writable.data();
      return ll_path.substr(0, ll_path.size() - 3);
   }

   static unsigned counter;
   char name[128];
   snprintf(name, sizeof(name), "/ventus_vk_rt_%ld_%u", (long)getpid(),
            counter++);
   return base + name;
}

static bool
vtvk_write_rt_linker_script(const std::string &path)
{
   FILE *file = fopen(path.c_str(), "w");
   if (!file) {
      mesa_loge("ventus-vk: failed to open RT linker script %s: %s",
                path.c_str(), strerror(errno));
      return false;
   }

   static const char script[] =
      "OUTPUT_ARCH(riscv)\n"
      "ENTRY(_start)\n"
      "PHDRS { text PT_LOAD FLAGS(5); data PT_LOAD FLAGS(6); }\n"
      "SECTIONS {\n"
      "  . = 0x80000000;\n"
      "  .text : { *(.text .text.*) } :text\n"
      "  . = ALIGN(16);\n"
      "  .eh_frame : { *(.eh_frame .eh_frame.*) } :text\n"
      "  . = ALIGN(16);\n"
      "  .rodata : { *(.rodata .rodata.*) } :text\n"
      "  . = ALIGN(16);\n"
      "  .data : {\n"
      "    *(.data .data.*) *(.sdata .sdata.*)\n"
      "  } :data\n"
      "  . = ALIGN(64);\n"
      "  .tohost : { *(.tohost .tohost.*) } :data\n"
      "  . = ALIGN(16);\n"
      "  .bss : { *(.bss .bss.*) *(COMMON) } :data\n"
      "}\n";
   if (fwrite(script, 1, sizeof(script) - 1, file) != sizeof(script) - 1) {
      mesa_loge("ventus-vk: failed to write RT linker script %s", path.c_str());
      fclose(file);
      return false;
   }

   fclose(file);
   return true;
}

static bool
vtvk_run_tool(const std::string &command, const char *stage)
{
   int ret = system(command.c_str());
   if (ret != 0) {
      mesa_loge("ventus-vk: RT %s failed with status %d: %s", stage, ret,
                command.c_str());
      return false;
   }
   return true;
}

static std::string
vtvk_llvm_tool_env_prefix(void)
{
   const char *lib_dir = getenv("VENTUS_VK_LLVM_TOOL_LIB_DIR");
   std::string install_lib_dir;
   if (!lib_dir || !lib_dir[0]) {
      const char *install_prefix = getenv("VENTUS_INSTALL_PREFIX");
      if (install_prefix && install_prefix[0]) {
         install_lib_dir = std::string(install_prefix) + "/lib";
         if (llvm::sys::fs::exists(install_lib_dir))
            lib_dir = install_lib_dir.c_str();
      }
   }
   if (!lib_dir || !lib_dir[0])
      lib_dir = "/home/guanys/ventus-env/llvm/build/lib";
   return "LD_LIBRARY_PATH=" + vtvk_shell_quote(lib_dir) +
          ":${LD_LIBRARY_PATH:-} ";
}

static bool
vtvk_dump_rt_llvm_module_to(llvm::Module *module,
                            const std::string &ll_path,
                            const char *label)
{
   std::string text;
   llvm::raw_string_ostream text_os(text);
   module->print(text_os, nullptr);
   text_os.flush();

   const std::string system_name = "define amdgpu_cs_chain void @main(";
   size_t main_def = text.find(system_name);
   if (main_def != std::string::npos)
      text.replace(main_def, system_name.size(),
                   "define ventus_kernel void @main(");

   std::error_code ec;
   llvm::raw_fd_ostream os(ll_path, ec, llvm::sys::fs::OF_Text);
   if (ec) {
      mesa_loge("ventus-vk: failed to dump RT LLVM module %s: %s",
                ll_path.c_str(), ec.message().c_str());
      return false;
   }

   os << text;
   os.flush();
   mesa_logi("ventus-vk: dumped %s RT LLVM module to %s", label,
             ll_path.c_str());
   return true;
}

static bool
vtvk_dump_rt_llvm_module(llvm::Module *module, std::string *ll_path)
{
   std::string base = vtvk_rt_artifact_base();
   *ll_path = base + ".ll";
   return vtvk_dump_rt_llvm_module_to(module, *ll_path, "pre-lowering");
}

static bool
vtvk_find_crt0(bool rv64, std::string *path)
{
   const char *env_path = getenv("VENTUS_VK_CRT0");
   if (env_path && env_path[0]) {
      if (llvm::sys::fs::exists(env_path)) {
         *path = env_path;
         return true;
      }
      mesa_loge("ventus-vk: VENTUS_VK_CRT0 does not exist: %s", env_path);
      return false;
   }

   const char *install_prefix = getenv("VENTUS_INSTALL_PREFIX");
   if (install_prefix && install_prefix[0]) {
      std::string candidate = std::string(install_prefix) + "/lib/" +
         (rv64 ? "riscv64/crt0.o" : "crt0.o");
      if (llvm::sys::fs::exists(candidate)) {
         *path = candidate;
         return true;
      }
   }

   static const char *rv32_candidates[] = {
      "/home/guanys/ventus-env/llvm/install/lib/crt0.o",
      "/home/guanys/ventus-env/llvm/build-libclc/lib/crt0.o",
      "/home/guanys/ventus-env/llvm/build-libclc/riscv32/lib/CMakeFiles/ctr0_obj.dir/crt0.S.o",
   };
   static const char *rv64_candidates[] = {
      "/home/guanys/ventus-env/llvm/build-libclc-rv64/lib/riscv64/crt0.o",
      "/home/guanys/ventus-env/llvm/build-libclc/riscv64/lib/crt0.o",
   };
   const char *const *candidates = rv64 ? rv64_candidates : rv32_candidates;
   const size_t candidate_count = rv64 ? ARRAY_SIZE(rv64_candidates) :
                                        ARRAY_SIZE(rv32_candidates);

   for (size_t i = 0; i < candidate_count; i++) {
      const char *candidate = candidates[i];
      if (llvm::sys::fs::exists(candidate)) {
         *path = candidate;
         return true;
      }
   }

   mesa_loge("ventus-vk: RT ELF needs Ventus crt0.o startup; set "
             "VENTUS_VK_CRT0 or build LLVM libclc");
   return false;
}

static bool
vtvk_dump_rt_emit_llvm_module(llvm::Module *module,
                              const std::string &ll_path,
                              std::string *emit_ll_path)
{
   *emit_ll_path = ll_path.substr(0, ll_path.size() - 3) + ".emit.ll";
   return vtvk_dump_rt_llvm_module_to(module, *emit_ll_path, "ELF-input");
}

static bool
vtvk_rt_elf_link_command(const std::string &tool_env, const char *ld_lld,
                         const std::string &ld_path,
                         const std::string &elf_path,
                         const std::string &crt0_path,
                         const std::string &obj_path,
                         std::string *link_cmd)
{
   *link_cmd = tool_env + vtvk_shell_quote(ld_lld) + " -T " +
               vtvk_shell_quote(ld_path.c_str()) + " -o " +
               vtvk_shell_quote(elf_path.c_str()) + " " +
               vtvk_shell_quote(crt0_path.c_str()) + " " +
               vtvk_shell_quote(obj_path.c_str());
   return true;
}

static const char *
vtvk_llvm_nm_path(void)
{
   const char *llvm_nm = getenv("VENTUS_VK_LLVM_NM");
   if (!llvm_nm || !llvm_nm[0])
      llvm_nm = "/home/guanys/ventus-env/llvm/build/bin/llvm-nm";
   return llvm_nm;
}

static bool
vtvk_read_elf_symbol(const std::string &elf_path, const char *symbol,
                     uint64_t *addr)
{
   std::string cmd = vtvk_llvm_tool_env_prefix() +
                     vtvk_shell_quote(vtvk_llvm_nm_path()) +
                     " --defined-only " + vtvk_shell_quote(elf_path.c_str());
   FILE *pipe = popen(cmd.c_str(), "r");
   if (!pipe) {
      mesa_loge("ventus-vk: failed to run llvm-nm for %s", elf_path.c_str());
      return false;
   }

   char line[512];
   bool found = false;
   while (fgets(line, sizeof(line), pipe)) {
      unsigned long long value = 0;
      char type = 0;
      char name[256] = {};
      if (sscanf(line, "%llx %c %255s", &value, &type, name) == 3 &&
          strcmp(name, symbol) == 0) {
         *addr = value;
         found = true;
         break;
      }
   }

   int ret = pclose(pipe);
   if (ret != 0) {
      mesa_loge("ventus-vk: llvm-nm failed with status %d: %s", ret,
                cmd.c_str());
      return false;
   }
   if (!found) {
      mesa_loge("ventus-vk: symbol %s not found in %s", symbol,
                elf_path.c_str());
      return false;
   }

   return true;
}

static bool
vtvk_maybe_emit_rt_kernel_module(llvm::Module *module, bool rv64,
                                 char **out_path, uint64_t *out_entry)
{
   *out_path = NULL;
   *out_entry = 0;
   if (!vtvk_env_enabled("VENTUS_VK_DUMP_LLVM") &&
       !vtvk_env_enabled("VENTUS_VK_EMIT_ELF"))
      return true;

   std::string ll_path;
   if (!vtvk_dump_rt_llvm_module(module, &ll_path) ||
       !vtvk_env_enabled("VENTUS_VK_EMIT_ELF"))
      return true;

   std::string obj_path = ll_path.substr(0, ll_path.size() - 3) + ".o";
   std::string elf_path = ll_path.substr(0, ll_path.size() - 3) + ".riscv";
   std::string ld_path = ll_path.substr(0, ll_path.size() - 3) + ".ld";
   std::string emit_ll_path;
   std::string crt0_path;
   const char *llc = getenv("VENTUS_VK_LLC");
   const char *ld_lld = getenv("VENTUS_VK_LD_LLD");
   if (!llc || !llc[0])
      llc = "/home/guanys/ventus-env/llvm/build/bin/llc";
   if (!ld_lld || !ld_lld[0])
      ld_lld = "/home/guanys/ventus-env/llvm/build/bin/ld.lld";

   if (!vtvk_write_rt_linker_script(ld_path))
      return false;
   if (!vtvk_find_crt0(rv64, &crt0_path))
      return false;

   if (!vtvk_dump_rt_emit_llvm_module(module, ll_path, &emit_ll_path))
      return false;

   std::string tool_env = vtvk_llvm_tool_env_prefix();
   std::string llc_cmd = tool_env + vtvk_shell_quote(llc) +
                         (rv64 ? " -mtriple=riscv64-unknown-elf"
                               : " -mtriple=riscv32-unknown-elf") +
                         (rv64 ? " -mcpu=ventus-gpgpu-rv64"
                               : " -mcpu=ventus-gpgpu") +
                         " -filetype=obj " +
                         vtvk_shell_quote(emit_ll_path.c_str()) + " -o " +
                         vtvk_shell_quote(obj_path.c_str());
   if (!vtvk_run_tool(llc_cmd, "llc"))
      return false;

   std::string link_cmd;
   if (!vtvk_rt_elf_link_command(tool_env, ld_lld, ld_path, elf_path, crt0_path,
                                 obj_path, &link_cmd))
      return false;
   if (!vtvk_run_tool(link_cmd, "link"))
      return false;

   uint64_t kernel_entry = 0;
   if (!vtvk_read_elf_symbol(elf_path, "main", &kernel_entry))
      return false;

   *out_path = strdup(elf_path.c_str());
   if (!*out_path)
      return false;
   *out_entry = kernel_entry;
   mesa_logi("ventus-vk: emitted RT RISC-V kernel to %s entry=0x%" PRIx64,
             elf_path.c_str(), kernel_entry);
   return true;
}

static void
vtvk_maybe_emit_rt_kernel(struct vtvk_pipeline *pipeline)
{
   char *path = NULL;
   uint64_t entry = 0;
   if (!vtvk_maybe_emit_rt_kernel_module(pipeline->llvm_module,
                                         pipeline->rt_uses_rv64,
                                         &path, &entry))
      return;
   free(pipeline->generated_kernel_path);
   pipeline->generated_kernel_path = path;
   pipeline->generated_kernel_entry = entry;
}

/* CPS lowerings produce raygen-stage resume NIR.  Keep every resume in its
 * own module/ELF: the current runtime has a single `main` kernel entry and a
 * later completion dispatcher chooses the ELF from its continuation ID. */
static bool
vtvk_emit_rt_resume_kernels(struct vtvk_pipeline *pipeline,
                            const struct vt_nir_to_llvm_options *options)
{
   if (!pipeline->resume_shader_count)
      return true;

   pipeline->resume_kernel_paths = (char **)calloc(
      pipeline->resume_shader_count, sizeof(*pipeline->resume_kernel_paths));
   pipeline->resume_kernel_entries = (uint64_t *)calloc(
      pipeline->resume_shader_count, sizeof(*pipeline->resume_kernel_entries));
   if (!pipeline->resume_kernel_paths || !pipeline->resume_kernel_entries)
      return false;
   pipeline->resume_kernel_count = pipeline->resume_shader_count;
   struct vt_nir_to_llvm_options resume_options = *options;
   /* A compact dispatch indexes ray refs, not the original raygen extent.
    * Logical launch IDs are restored from the global completion plane. */
   resume_options.global_rt_compact_resume = true;

   for (uint32_t i = 0; i < pipeline->resume_shader_count; i++) {
      nir_shader *resume = pipeline->resume_shaders[i];
      llvm::Module *module = vt_nir_to_llvm_translate(&resume, 1,
                                                       &resume_options);
      if (!module) {
         mesa_loge("ventus-vk: failed to translate global RT resume %u", i);
         return false;
      }

      char *path = NULL;
      uint64_t entry = 0;
      const bool emitted = vtvk_maybe_emit_rt_kernel_module(
         module, pipeline->rt_uses_rv64, &path, &entry);
      delete module;
      if (!emitted)
         return false;

      pipeline->resume_kernel_paths[i] = path;
      pipeline->resume_kernel_entries[i] = entry;
   }

   mesa_logi("ventus-vk: prepared %u global RT CPS resume kernel%s",
             pipeline->resume_kernel_count,
             pipeline->resume_kernel_count == 1 ? "" : "s");
   return true;
}

struct vtvk_rt_callback_shader_entry {
   uint32_t group_index;
   mesa_shader_stage stage;
   nir_shader *shader;
};

struct vtvk_rt_callback_shaders {
   nir_shader *raygen;
   std::vector<vtvk_rt_callback_shader_entry> miss;
   std::vector<vtvk_rt_callback_shader_entry> closest_hit;
   std::vector<vtvk_rt_callback_shader_entry> any_hit;
   std::vector<vtvk_rt_callback_shader_entry> intersection;
};

struct vtvk_rt_pipeline_callback_data {
   const vtvk_rt_callback_shaders *shaders;
};

struct vtvk_rt_payload_location_candidate {
   int location;
   unsigned size;
   unsigned align;
   const char *name;
};

static bool
vtvk_rt_is_payload_variable(const nir_variable *var)
{
   return var && (var->data.mode == nir_var_shader_temp ||
           var->data.mode == nir_var_shader_call_data);
}

static void
vtvk_rt_collect_payload_location_candidates(
   nir_shader *shader, std::vector<vtvk_rt_payload_location_candidate> *out)
{
   if (!shader || !out)
      return;

   nir_foreach_variable_in_shader(var, shader) {
      if (!vtvk_rt_is_payload_variable(var) || var->data.location < 0)
         continue;

      unsigned size = 0;
      unsigned align = 0;
      glsl_get_natural_size_align_bytes(var->type, &size, &align);
      out->push_back({var->data.location, size, align, var->name});
   }
}

static int
vtvk_rt_unique_payload_location_for_var(
   nir_variable *var,
   const std::vector<vtvk_rt_payload_location_candidate> &candidates)
{
   unsigned size = 0;
   unsigned align = 0;
   glsl_get_natural_size_align_bytes(var->type, &size, &align);

   int shape_location = -1;
   int name_location = -1;
   bool shape_ambiguous = false;
   bool name_ambiguous = false;

   for (const auto &candidate : candidates) {
      if (candidate.size != size || candidate.align != align)
         continue;

      if (shape_location < 0)
         shape_location = candidate.location;
      else if (shape_location != candidate.location)
         shape_ambiguous = true;

      if (var->name && candidate.name && strcmp(var->name, candidate.name) == 0) {
         if (name_location < 0)
            name_location = candidate.location;
         else if (name_location != candidate.location)
            name_ambiguous = true;
      }
   }

   if (name_location >= 0 && !name_ambiguous)
      return name_location;
   if (shape_location >= 0 && !shape_ambiguous)
      return shape_location;

   return -1;
}

static int
vtvk_rt_allocate_payload_location(
   const std::vector<vtvk_rt_payload_location_candidate> &candidates)
{
   for (int location = 0; location < INT_MAX; location++) {
      bool used = false;
      for (const auto &candidate : candidates) {
         if (candidate.location == location) {
            used = true;
            break;
         }
      }
      if (!used)
         return location;
   }
   return -1;
}

static void
vtvk_rt_infer_missing_payload_locations(nir_shader **shaders,
                                        uint32_t shader_count)
{
   /* SPIR-V payload decorations arrive with data.location populated, but not
    * always with NIR's explicit_location bit.  The ABI lowering needs the
    * latter to preserve the Vulkan location across raygen and callbacks. */
   for (uint32_t i = 0; i < shader_count; i++) {
      nir_shader *shader = shaders[i];
      if (!shader)
         continue;
      nir_foreach_variable_in_shader(var, shader) {
         if (vtvk_rt_is_payload_variable(var) && var->data.location >= 0)
            var->data.explicit_location = true;
      }
   }

   /* A raygen shader commonly has exactly one shader_temp payload object
    * (for example `hitValue`) with no SPIR-V Location decoration.  It is the
    * payload passed to TraceRayEXT, not an ordinary private temporary.  Make
    * the canonical single-payload location explicit before computing P; if
    * this is left unresolved, callbacks still write H..H+P while the linked
    * ABI incorrectly reserves P=0. */
   for (uint32_t i = 0; i < shader_count; i++) {
      nir_shader *shader = shaders[i];
      if (!shader || shader->info.stage != MESA_SHADER_RAYGEN)
         continue;

      nir_variable *payload = NULL;
      bool ambiguous = false;
      nir_foreach_variable_in_shader(var, shader) {
         if (var->data.mode != nir_var_shader_temp ||
             var->data.location >= 0)
            continue;
         if (payload)
            ambiguous = true;
         else
            payload = var;
      }
      if (payload && !ambiguous) {
         payload->data.location = 0;
         payload->data.explicit_location = true;
         if (vtvk_env_enabled("VENTUS_VK_NIR_PROBE"))
            mesa_logi("ventus-vk: inferred RT raygen payload location 0 for %s",
                      payload->name ? payload->name : "<unnamed>");
      }
   }

   std::vector<vtvk_rt_payload_location_candidate> candidates;
   for (uint32_t i = 0; i < shader_count; i++)
      vtvk_rt_collect_payload_location_candidates(shaders[i], &candidates);

   if (candidates.empty()) {
      const char *name = NULL;
      unsigned size = 0;
      unsigned align = 0;
      bool have_payload = false;

      for (uint32_t i = 0; i < shader_count; i++) {
         nir_shader *shader = shaders[i];
         if (!shader)
            continue;

         nir_foreach_variable_in_shader(var, shader) {
            if (var->data.mode != nir_var_shader_temp &&
                var->data.mode != nir_var_shader_call_data)
               continue;

            unsigned var_size = 0;
            unsigned var_align = 0;
            glsl_get_natural_size_align_bytes(var->type, &var_size, &var_align);
            if (have_payload &&
                (size != var_size || align != var_align ||
                 ((name || var->name) &&
                  (!name || !var->name || strcmp(name, var->name) != 0))))
               return;

            name = var->name;
            size = var_size;
            align = var_align;
            have_payload = true;
         }
      }

      if (!have_payload)
         return;

      for (uint32_t i = 0; i < shader_count; i++) {
         nir_shader *shader = shaders[i];
         if (!shader)
            continue;

         nir_foreach_variable_in_shader(var, shader) {
            if (var->data.mode != nir_var_shader_temp &&
                var->data.mode != nir_var_shader_call_data)
               continue;

            var->data.location = 0;
            var->data.explicit_location = true;
         }
      }

      if (vtvk_env_enabled("VENTUS_VK_NIR_PROBE"))
         mesa_logi("ventus-vk: inferred RT payload location 0 for a "
                   "single unlocated payload interface");
      return;
   }

   for (uint32_t i = 0; i < shader_count; i++) {
      nir_shader *shader = shaders[i];
      if (!shader)
         continue;

      nir_foreach_variable_in_shader(var, shader) {
         if (!vtvk_rt_is_payload_variable(var) || var->data.location >= 0)
            continue;

         int location =
            vtvk_rt_unique_payload_location_for_var(var, candidates);
         if (location < 0) {
            /* Some legacy shader generators omitted Location decorations on
             * RayPayloadKHR variables.  Once a pipeline contains an
             * explicitly located payload, the old single-interface fallback
             * cannot cover these remaining globals.  Give every unmatched
             * interface a stable private location; later variables with the
             * same unique name/shape resolve to this candidate, so a
             * closest-hit TraceRay and its miss shader share one slot.
             *
             * This is a linked megakernel implementation detail: the Vulkan
             * payload operand itself remains the caller PDS address written
             * to the RT header. */
            location = vtvk_rt_allocate_payload_location(candidates);
            if (location < 0)
               continue;

            unsigned size = 0;
            unsigned align = 0;
            glsl_get_natural_size_align_bytes(var->type, &size, &align);
            candidates.push_back({location, size, align, var->name});
            if (vtvk_env_enabled("VENTUS_VK_NIR_PROBE")) {
               mesa_logi("ventus-vk: assigned missing RT payload location %d for %s",
                         location, var->name ? var->name : "<unnamed>");
            }
         }

         var->data.location = location;
         var->data.explicit_location = true;
         if (vtvk_env_enabled("VENTUS_VK_NIR_PROBE")) {
            mesa_logi("ventus-vk: inferred RT payload location %d for %s",
                      location, var->name ? var->name : "<unnamed>");
         }
      }
   }
}

static bool
vtvk_rt_is_triangle_barycentrics(nir_shader *shader, nir_variable *var,
                                  bool has_intersection)
{
   if (!shader || !var || has_intersection ||
       var->data.mode != nir_var_ray_hit_attrib)
      return false;

   unsigned size = 0, align = 0;
   glsl_get_natural_size_align_bytes(var->type, &size, &align);
   const int location = var->data.driver_location;
   return location == 0 && size == 8 && align >= 4;
}

static bool
vtvk_rt_compute_hit_attribute_rest(nir_shader **shaders, uint32_t shader_count,
                                   bool *triangle_barycentrics,
                                   uint32_t *rest_bytes)
{
   if (!triangle_barycentrics || !rest_bytes)
      return false;

   bool has_intersection = false;
   for (uint32_t i = 0; i < shader_count; i++)
      has_intersection |= shaders[i] &&
                          shaders[i]->info.stage == MESA_SHADER_INTERSECTION;

   uint32_t end = 0;
   for (uint32_t i = 0; i < shader_count; i++) {
      nir_shader *shader = shaders[i];
      if (!shader)
         continue;
      nir_foreach_variable_in_shader(var, shader) {
         if (var->data.mode != nir_var_ray_hit_attrib)
            continue;
         if (vtvk_rt_is_triangle_barycentrics(shader, var, has_intersection))
            continue;
         unsigned size = 0, align = 0;
         glsl_get_natural_size_align_bytes(var->type, &size, &align);
         const int location = var->data.driver_location;
         if (location < 0)
            return false;
         const uint64_t begin = (uint64_t)(uint32_t)location;
         const uint64_t finish = begin + size;
         if (finish > UINT32_MAX)
            return false;
         end = MAX2(end, (uint32_t)finish);
      }
   }

   *triangle_barycentrics = !has_intersection;
   *rest_bytes = ALIGN_POT(end, VT_RT_ABI_WORD_BYTES);
   return true;
}

static void
vtvk_inline_rt_callback_shader(nir_builder *b, nir_shader *shader)
{
   if (!b || !shader)
      return;

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   if (!impl)
      return;

   nir_opt_dead_cf(shader);
   vt_nir_lower_rt_callback_returns(shader);
   nir_opt_dce(shader);

   struct hash_table *var_remap = _mesa_pointer_hash_table_create(b->shader);
   if (!var_remap)
      return;

   nir_foreach_variable_in_shader(var, shader) {
      if (var->data.mode == nir_var_function_temp)
         continue;

      nir_variable *clone = nir_variable_clone(var, b->shader);
      if (!clone) {
         _mesa_hash_table_destroy(var_remap, NULL);
         return;
      }
      nir_shader_add_variable(b->shader, clone);
      _mesa_hash_table_insert(var_remap, var, clone);
   }

   b->shader->scratch_size =
      MAX2(b->shader->scratch_size, shader->scratch_size);
   nir_inline_function_impl(b, impl, NULL, var_remap);
   _mesa_hash_table_destroy(var_remap, NULL);
}

static void
vtvk_inline_rt_callback_shader_with_index(nir_builder *b,
                                          const std::vector<vtvk_rt_callback_shader_entry>
                                            &entries,
                                         nir_def *index)
{
   if (!b || !index || entries.empty())
      return;

   /* Diagnostic only: textures has exactly one closest-hit group.  Bypass
    * the dynamic SBT predicate to distinguish a callback-body/CPS-address
    * bug from a SIMT lowering bug in that predicate.  This must never become
    * a normal fallback: a pipeline may legally have hit groups with no
    * closest-hit shader. */
   if (entries.size() == 1 &&
       vtvk_env_enabled("VENTUS_VK_RT_DEBUG_DIRECT_SINGLE_CALLBACK")) {
      vtvk_inline_rt_callback_shader(b, entries[0].shader);
      return;
   }

   /* Diagnostic for multi-miss pipelines such as raytracingshadows.  Keep
    * traversal's hit/miss decision intact, but bypass only the dynamic SBT
    * record group-index load for a named group. */
   if (vtvk_env_enabled("VENTUS_VK_RT_DEBUG_DIRECT_CALLBACK_GROUP_2")) {
      for (const auto &entry : entries) {
         if (entry.group_index == 2) {
            vtvk_inline_rt_callback_shader(b, entry.shader);
            return;
         }
      }
   }

   for (size_t i = 0; i < entries.size(); i++) {
      nir_push_if(b, nir_ieq_imm(b, index, entries[i].group_index));
      vtvk_inline_rt_callback_shader(b, entries[i].shader);
      if (i + 1 < entries.size())
         nir_push_else(b, NULL);
   }
   for (size_t i = 0; i < entries.size(); i++)
      nir_pop_if(b, NULL);
}

/* A procedural report stays inside one compact intersection callback.  The
 * matching any-hit code is inlined at the report site, so locals and control
 * flow naturally continue to a later reportIntersectionEXT in the same
 * shader.  The callback decision is overwritten by each report; the
 * committed record retains the latest accepted report. */
struct vtvk_rt_intersection_report_lower_data {
   const struct vt_nir_rt_config *config;
   nir_shader *any_hit;
};

static bool
vtvk_lower_rt_intersection_report(nir_builder *b, nir_intrinsic_instr *intrin,
                                  void *data)
{
   if (intrin->intrinsic != nir_intrinsic_report_ray_intersection)
      return false;

   const vtvk_rt_intersection_report_lower_data *state =
      static_cast<const vtvk_rt_intersection_report_lower_data *>(data);
   const struct vt_nir_rt_config *config = state->config;
   const enum vt_rt_shader_layout layout =
      vt_nir_layout_for_stage(b->shader->info.stage);
   b->cursor = nir_before_instr(&intrin->instr);

   nir_def *hit_t = intrin->src[0].ssa;
   nir_def *hit_kind = intrin->src[1].ssa;
   nir_def *committed_valid = nir_ieq_imm(
      b, vt_nir_load_hit_record_meta_field(
            b, config, layout, VT_RT_HIT_RECORD_META0,
            VT_RT_HIT_RECORD_META0_VALID_SHIFT, 1),
      VT_RT_HIT_RECORD_STATUS_VALID);
   nir_def *committed_t = vt_nir_load_hit_record_word(
      b, config, layout, VT_RT_HIT_RECORD_HIT_T, 32);
   nir_def *ray_tmax = vt_nir_load_ray_state_word(
      b, config, layout, VT_RT_SLOT_TMAX);
   nir_def *current_tmax = nir_bcsel(b, committed_valid, committed_t, ray_tmax);
   nir_def *tmin = vt_nir_load_ray_state_word(
      b, config, layout, VT_RT_SLOT_TMIN);
   nir_def *terminated = nir_ieq_imm(
      b, vt_nir_load_control_word(
            b, config, layout, VT_RT_CONTROL_CALLBACK_DECISION, 32),
      VT_RT_CALLBACK_TERMINATE);
   const bool global_callback =
      config->execution_model == VT_RT_EXECUTION_WAVEFRONT_GLOBAL_CALLBACK;
   nir_def *candidate_attributes = NULL;
   nir_def *committed_attributes = NULL;
   if (global_callback) {
      nir_def *ray_ref = vt_nir_rt_global_callback_ray_ref(b);
      candidate_attributes = vt_nir_rt_global_load_completion_addr(
         b, ray_ref, VT_RT_GLOBAL_COMPLETION_CANDIDATE_HIT_ATTRIBUTE_ADDR_LO);
      committed_attributes = vt_nir_rt_global_load_completion_addr(
         b, ray_ref, VT_RT_GLOBAL_COMPLETION_HIT_ATTRIBUTE_ADDR_LO);
   }
   nir_def *in_range = nir_iand(
      b, nir_flt(b, hit_t, current_tmax), nir_fge(b, hit_t, tmin));
   nir_def *eligible = nir_iand(b, in_range, nir_inot(b, terminated));

   /* Preserve raw AS metadata populated by RTcore, changing only the
    * report-specific fields. */
   vt_nir_store_candidate_hit_record_meta_field(
      b, config, layout, VT_RT_HIT_RECORD_META0,
      VT_RT_HIT_RECORD_META0_VALID_SHIFT, 1, eligible);
   vt_nir_store_candidate_hit_record_meta_field(
      b, config, layout, VT_RT_HIT_RECORD_META0,
      VT_RT_HIT_RECORD_META0_STATUS_SHIFT,
      VT_RT_HIT_RECORD_META0_STATUS_BITS,
      nir_imm_int(b, VT_RT_TRAVERSAL_CANDIDATE_PROCEDURAL_AABB));
   vt_nir_store_candidate_hit_record_word(
      b, config, layout, VT_RT_HIT_RECORD_HIT_T, hit_t);
   vt_nir_store_candidate_hit_kind(b, config, layout, hit_kind);
   vt_nir_store_candidate_hit_record_meta_field(
      b, config, layout, VT_RT_HIT_RECORD_META0,
      VT_RT_HIT_RECORD_META0_FRONT_FACE_SHIFT, 1,
      nir_ieq_imm(b, hit_kind, 0xfe));
   vt_nir_store_control_word(
      b, config, layout, VT_RT_CONTROL_CALLBACK_DECISION,
      nir_bcsel(b, terminated, nir_imm_int(b, VT_RT_CALLBACK_TERMINATE),
                nir_bcsel(b, eligible, nir_imm_int(b, VT_RT_CALLBACK_ACCEPT),
                          nir_imm_int(b, VT_RT_CALLBACK_IGNORE))));

   /* RTcore derived this from the raw AS opacity, ray flags, and instance
    * flags when it emitted the AABB candidate.  Do not recompute it here:
    * intersection shaders cannot otherwise observe instance ForceOpaque /
    * ForceNoOpaque consistently. */
   nir_def *need_software_opacity_test = nir_ine_imm(
      b, vt_nir_load_candidate_hit_record_meta_field(
            b, config, layout, VT_RT_HIT_RECORD_META0,
            VT_RT_HIT_RECORD_META0_NEED_SOFTWARE_OPACITY_TEST_SHIFT, 1), 0);

   if (state->any_hit) {
      nir_push_if(b, nir_iand(b, eligible, need_software_opacity_test));
      vtvk_inline_rt_callback_shader(b, state->any_hit);
      nir_pop_if(b, NULL);
   }

   nir_def *decision = vt_nir_load_control_word(
      b, config, layout, VT_RT_CONTROL_CALLBACK_DECISION, 32);
   nir_def *accepted = nir_iand(b, eligible, nir_ior(
      b, nir_ieq_imm(b, decision, VT_RT_CALLBACK_ACCEPT),
      nir_ieq_imm(b, decision, VT_RT_CALLBACK_TERMINATE)));

   nir_push_if(b, accepted);
   for (uint32_t word = 0;
        word < VT_RT_HIT_RECORD_WORD_COUNT;
        word++) {
      const enum vt_rt_hit_record_word hit_word =
         static_cast<enum vt_rt_hit_record_word>(word);
      vt_nir_store_hit_record_word(
         b, config, layout, hit_word,
         vt_nir_load_candidate_hit_record_word(b, config, layout, hit_word, 32));
   }
   if (global_callback) {
      const uint32_t hit_attrib_words =
         config->scratchpad[layout].hit_attrib_size_bytes / sizeof(uint32_t);
      for (uint32_t word = 0; word < hit_attrib_words; word++) {
         nir_def *offset = nir_imm_intN_t(b, word * sizeof(uint32_t), 64);
         nir_store_global(b, nir_load_global(
                                 b, 1, 32, nir_iadd(b, candidate_attributes, offset)),
                          nir_iadd(b, committed_attributes, offset), 0x1);
      }
   } else {
      const struct vt_nir_scratchpad_layout *sp = &config->scratchpad[layout];
      for (uint32_t word = 0; word < sp->hit_attrib_size_bytes / sizeof(uint32_t);
           word++) {
         nir_def *candidate_addr = vt_nir_scratchpad_addr(
            b, config, layout, sp->hit_attrib_base_bytes + word * sizeof(uint32_t));
         nir_def *committed_addr = vt_nir_scratchpad_addr(
            b, config, layout,
            sp->committed_hit_attrib_base_bytes + word * sizeof(uint32_t));
         nir_ventus_rt_store_slot(
            b, nir_ventus_rt_load_slot(b, 1, 32, candidate_addr), committed_addr);
      }
   }
   nir_pop_if(b, NULL);

   nir_def_rewrite_uses(&intrin->def, accepted);
   nir_instr_remove(&intrin->instr);
   return true;
}

static bool
vtvk_lower_rt_intersection_reports(
   nir_shader *shader, const struct vt_nir_rt_config *config,
   nir_shader *any_hit)
{
   if (!shader || shader->info.stage != MESA_SHADER_INTERSECTION)
      return false;
   const vtvk_rt_intersection_report_lower_data data = {
      .config = config,
      .any_hit = any_hit,
   };
   return nir_shader_intrinsics_pass(shader, vtvk_lower_rt_intersection_report,
                                     nir_metadata_control_flow, (void *)&data);
}

static nir_def *
vtvk_load_rt_shader_record_group_index(nir_builder *b, nir_def *shader_record_ptr)
{
   if (!b || !shader_record_ptr)
      return NULL;

   const int32_t group_index_offset = 4 - 32;
   return nir_load_global(b, 1, 32,
                          nir_iadd_imm(b, shader_record_ptr,
                                       group_index_offset));
}

/* This is compiler-transient state derived from the trace SBT inputs and the
 * packed hit record.  It is never a field in the RTcore-visible PDS ABI. */
static nir_def *
vtvk_load_rt_hit_shader_record_ptr(
   nir_builder *b, const struct vt_nir_rt_trace_params *params,
   const struct vt_nir_rt_result *result)
{
   (void)b;
   (void)params;
   if (!result)
      return NULL;
   return result->hit_shader_record_ptr;
}

static void
vtvk_emit_rt_miss(nir_builder *b, const struct vt_nir_rt_trace_params *params,
                  const struct vt_nir_rt_result *result, void *data)
{
   (void)params;
   const struct vtvk_rt_pipeline_callback_data *cb =
      static_cast<const struct vtvk_rt_pipeline_callback_data *>(data);

   nir_def *record_ptr = result ? result->miss_shader_record_ptr : NULL;
   nir_def *index = vtvk_load_rt_shader_record_group_index(b, record_ptr);
   if (!cb || !cb->shaders || cb->shaders->miss.empty() || !index)
      return;

   index = nir_channel(b, index, 0);
   vtvk_inline_rt_callback_shader_with_index(b, cb->shaders->miss, index);
}

static void
vtvk_emit_rt_closest_hit(nir_builder *b,
                         const struct vt_nir_rt_trace_params *params,
                         const struct vt_nir_rt_result *result, void *data)
{
   (void)params;
   const struct vtvk_rt_pipeline_callback_data *cb =
      static_cast<const struct vtvk_rt_pipeline_callback_data *>(data);

   if (!cb || !cb->shaders || cb->shaders->closest_hit.empty() ||
       !params || !result)
      return;

   nir_def *record_ptr =
      vtvk_load_rt_hit_shader_record_ptr(b, params, result);
   if (!record_ptr)
      return;
   nir_def *index =
      vtvk_load_rt_shader_record_group_index(b, record_ptr);
   vtvk_inline_rt_callback_shader_with_index(b, cb->shaders->closest_hit,
                                             index);
}

static void
vtvk_emit_rt_any_hit(nir_builder *b, const struct vt_nir_rt_trace_params *params,
                     const struct vt_nir_rt_result *result, void *data)
{
   (void)params;
   const struct vtvk_rt_pipeline_callback_data *cb =
      static_cast<const struct vtvk_rt_pipeline_callback_data *>(data);

   if (!cb || !cb->shaders || cb->shaders->any_hit.empty() ||
       !params || !result)
      return;

   nir_def *record_ptr =
      vtvk_load_rt_hit_shader_record_ptr(b, params, result);
   if (!record_ptr)
      return;
   nir_def *index =
      vtvk_load_rt_shader_record_group_index(b, record_ptr);
   vtvk_inline_rt_callback_shader_with_index(b, cb->shaders->any_hit, index);
}

static void
vtvk_emit_rt_intersection(nir_builder *b, const struct vt_nir_rt_trace_params *params,
                          const struct vt_nir_rt_result *result, void *data)
{
   (void)params;
   const struct vtvk_rt_pipeline_callback_data *cb =
      static_cast<const struct vtvk_rt_pipeline_callback_data *>(data);

   if (!cb || !cb->shaders || cb->shaders->intersection.empty() ||
       !params || !result)
      return;

   nir_def *record_ptr =
      vtvk_load_rt_hit_shader_record_ptr(b, params, result);
   if (!record_ptr)
      return;
   nir_def *index =
      vtvk_load_rt_shader_record_group_index(b, record_ptr);
   vtvk_inline_rt_callback_shader_with_index(b, cb->shaders->intersection,
                                             index);
}
static bool
vtvk_rt_stage_supported(mesa_shader_stage stage)
{
   return stage == MESA_SHADER_RAYGEN ||
          stage == MESA_SHADER_MISS ||
          stage == MESA_SHADER_CLOSEST_HIT ||
          stage == MESA_SHADER_ANY_HIT ||
          stage == MESA_SHADER_INTERSECTION;
}

static bool
vtvk_map_shader_resources(nir_shader *shader,
                           const struct vtvk_pipeline_layout_data *layout)
{
   nir_foreach_variable_in_shader(var, shader) {
      /* SPIR-V carries AS bindings on vulkan_resource_index, not on its
       * bookkeeping variable (whose set/binding remain zero). */
      if (var->data.resource_type == nir_resource_type_acceleration_structure)
         continue;
      if (!(var->data.mode & (nir_var_uniform | nir_var_image | nir_var_mem_ubo | nir_var_mem_ssbo)))
         continue;
      if (var->data.descriptor_set >= layout->count)
         return false;
      int slot = vtvk_descriptor_slot(&layout->sets[var->data.descriptor_set],
                                      var->data.binding);
      if (slot < 0) {
         mesa_loge("ventus-vk: resource %s mode=0x%x set=%u binding=%u missing from layout",
                   var->name ? var->name : "<unnamed>", (unsigned)var->data.mode,
                   var->data.descriptor_set, var->data.binding);
         return false;
      }
      var->data.binding = slot;
   }
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_vulkan_resource_index)
               continue;
            uint32_t set = nir_intrinsic_desc_set(intr);
            if (set >= layout->count)
               return false;
            int slot = vtvk_descriptor_slot(&layout->sets[set], nir_intrinsic_binding(intr));
            if (slot < 0) {
               mesa_loge("ventus-vk: resource index set=%u binding=%u missing from layout",
                         set, nir_intrinsic_binding(intr));
               return false;
            }
            nir_intrinsic_set_binding(intr, slot);
         }
      }
   }
   return true;
}

static VkResult
vtvk_compile_rt_stage(struct vtvk_device *device,
                      struct vtvk_pipeline *pipeline,
                      VkPipelineCreateFlags2KHR pipeline_flags,
                      const VkPipelineShaderStageCreateInfo *stage_info,
                      nir_shader **nir_out)
{
   mesa_shader_stage stage = vk_to_mesa_shader_stage(stage_info->stage);

   if (!vtvk_rt_stage_supported(stage))
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);

   VkResult result =
      vk_pipeline_shader_stage_to_nir(&device->vk, pipeline_flags, stage_info,
                                      &vtvk_spirv_options, &vtvk_nir_options,
                                      pipeline->nir_mem_ctx, nir_out);
   if (result != VK_SUCCESS)
      return result;

   if (!vtvk_map_shader_resources(*nir_out, &pipeline->resource_layout)) {
      mesa_loge("ventus-vk: shader resource is absent from the pipeline layout");
      return vk_error(device, VK_ERROR_INITIALIZATION_FAILED);
   }
   vtvk_rt_log_nir_stats(*nir_out, "after spirv_to_nir");
   NIR_PASS(_, *nir_out, nir_lower_system_values);
   vtvk_rt_log_nir_stats(*nir_out, "after nir_lower_system_values");
   NIR_PASS(_, *nir_out, vt_nir_lower_descriptors);
   vtvk_rt_log_nir_stats(*nir_out, "after vt_nir_lower_descriptors");
   /* Convert the API push-constant variable while it is still common to the
    * root shader and every shader-call clone.  The global CPS lowering below
    * then replaces the resulting load_push_constant intrinsics with loads
    * from the immutable dispatch snapshot.  Leaving this logical variable in
    * place reaches the LLVM canonicalizer as an uninitialized function
    * temporary in a root-resume shader. */
   NIR_PASS(_, *nir_out, nir_lower_explicit_io, nir_var_mem_push_const,
            nir_address_format_32bit_offset);
   vtvk_rt_log_nir_stats(*nir_out, "after Ventus RT push-constant IO lowering");
   nir_validate_shader(*nir_out, "after Ventus RT pre-lowering");

   return VK_SUCCESS;
}

static void
vtvk_fill_group_handles(struct vtvk_pipeline *pipeline,
                        const VkRayTracingPipelineCreateInfoKHR *info)
{
   const uint32_t handle_size = 32;
   pipeline->group_count = info->groupCount;

   for (uint32_t g = 0; g < info->groupCount; g++) {
      uint8_t *handle = pipeline->group_handles + g * handle_size;
      const VkRayTracingShaderGroupCreateInfoKHR *group = &info->pGroups[g];

      memcpy(handle, "VTRT", 4);
      memcpy(handle + 4, &g, sizeof(g));
      memcpy(handle + 8, &group->type, sizeof(group->type));
      memcpy(handle + 12, &group->generalShader, sizeof(group->generalShader));
      memcpy(handle + 16, &group->closestHitShader,
             sizeof(group->closestHitShader));
      memcpy(handle + 20, &group->anyHitShader, sizeof(group->anyHitShader));
      memcpy(handle + 24, &group->intersectionShader,
             sizeof(group->intersectionShader));
   }
}

static void
vtvk_collect_rt_callback_shaders(const VkRayTracingPipelineCreateInfoKHR *info,
                                nir_shader **shaders, uint32_t stage_count,
                                vtvk_rt_callback_shaders *out)
{
   out->raygen = nullptr;
   out->miss.clear();
   out->closest_hit.clear();
   out->any_hit.clear();
   out->intersection.clear();

   auto add_shader = [&](uint32_t shader_index, uint32_t group_index,
                         mesa_shader_stage required_stage,
                         std::vector<vtvk_rt_callback_shader_entry> *dst) {
      if (shader_index == VK_SHADER_UNUSED_KHR || shader_index >= stage_count ||
          shader_index >= info->stageCount)
         return;

      mesa_shader_stage shader_stage =
         vk_to_mesa_shader_stage(info->pStages[shader_index].stage);
      if (shader_stage == required_stage) {
         dst->push_back({group_index, required_stage, shaders[shader_index]});
      }
   };

   for (uint32_t group_index = 0; group_index < info->groupCount; group_index++) {
      const VkRayTracingShaderGroupCreateInfoKHR *group = &info->pGroups[group_index];
      if (group->type == VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR) {
         if (group->generalShader != VK_SHADER_UNUSED_KHR &&
             group->generalShader < stage_count &&
             group->generalShader < info->stageCount) {
            mesa_shader_stage stage =
               vk_to_mesa_shader_stage(info->pStages[group->generalShader].stage);
            if (stage == MESA_SHADER_RAYGEN && !out->raygen)
               out->raygen = shaders[group->generalShader];
            else if (stage == MESA_SHADER_MISS)
               add_shader(group->generalShader, group_index, MESA_SHADER_MISS,
                          &out->miss);
         }
      } else if (group->type == VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR ||
                 group->type == VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR) {
         add_shader(group->closestHitShader, group_index, MESA_SHADER_CLOSEST_HIT,
                    &out->closest_hit);
         add_shader(group->anyHitShader, group_index, MESA_SHADER_ANY_HIT,
                    &out->any_hit);
         if (group->type == VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR)
            add_shader(group->intersectionShader, group_index,
                       MESA_SHADER_INTERSECTION, &out->intersection);
      }
   }

   for (uint32_t i = 0; i < stage_count && i < info->stageCount; i++) {
      mesa_shader_stage stage = vk_to_mesa_shader_stage(info->pStages[i].stage);
      if (stage == MESA_SHADER_RAYGEN && !out->raygen)
         out->raygen = shaders[i];
   }
}

/* Candidate callbacks are separate compact kernels: their control stores must
 * become visible to RTcore before the same ray_ref resumes its private
 * traversal state.  Terminal miss/closest-hit remain in the root CPS resume
 * ELFs, where post-trace raygen code can continue immediately. */
static bool
vtvk_emit_rt_global_candidate_kernels(
   struct vtvk_pipeline *pipeline,
   const std::vector<vtvk_rt_callback_shader_entry> &any_hit,
   const std::vector<vtvk_rt_callback_shader_entry> &intersection,
   const struct vt_nir_to_llvm_options *options)
{
   std::vector<vtvk_rt_callback_shader_entry> callbacks = any_hit;
   callbacks.insert(callbacks.end(), intersection.begin(), intersection.end());
   if (callbacks.empty())
      return true;

   pipeline->callback_kernel_paths = (char **)calloc(
      callbacks.size(), sizeof(*pipeline->callback_kernel_paths));
   pipeline->callback_kernel_entries = (uint64_t *)calloc(
      callbacks.size(), sizeof(*pipeline->callback_kernel_entries));
   pipeline->callback_kernel_groups = (uint32_t *)calloc(
      callbacks.size(), sizeof(*pipeline->callback_kernel_groups));
   pipeline->callback_kernel_stages = (uint32_t *)calloc(
      callbacks.size(), sizeof(*pipeline->callback_kernel_stages));
   pipeline->callback_kernel_lds_sizes = (uint32_t *)calloc(
      callbacks.size(), sizeof(*pipeline->callback_kernel_lds_sizes));
   if (!pipeline->callback_kernel_paths || !pipeline->callback_kernel_entries ||
       !pipeline->callback_kernel_groups || !pipeline->callback_kernel_stages ||
       !pipeline->callback_kernel_lds_sizes)
      return false;
   pipeline->callback_kernel_count = callbacks.size();

   struct vt_nir_rt_config config = vt_nir_rt_default_config();
   struct vt_nir_rt_abi_layout abi = vt_nir_rt_default_abi_layout();
   vt_nir_rt_config_apply_abi_layout(&config, &abi);
   vt_nir_rt_config_apply_global_cps_layout(&config, &abi);
   config.execution_model = VT_RT_EXECUTION_WAVEFRONT_GLOBAL_CALLBACK;

   struct vt_nir_to_llvm_options callback_options = *options;
   callback_options.global_rt_compact_resume = true;
   callback_options.entry_point_name = "main";
   callback_options.entry_point_is_kernel = true;
   for (uint32_t i = 0; i < callbacks.size(); i++) {
      nir_shader *shader = callbacks[i].shader;
      if (callbacks[i].stage == MESA_SHADER_INTERSECTION) {
         nir_shader *matching_any_hit = NULL;
         for (const vtvk_rt_callback_shader_entry &entry : any_hit) {
            if (entry.group_index == callbacks[i].group_index) {
               matching_any_hit = entry.shader;
               break;
            }
         }
         vtvk_lower_rt_intersection_reports(shader, &config, matching_any_hit);
      }
      uint32_t callback_lds_size = 0;
      if (!vt_nir_lower_rt_global_callback(shader, &config,
                                           &callback_lds_size)) {
         mesa_loge("ventus-vk: global candidate callback lowering failed stage=%u group=%u",
                   callbacks[i].stage, callbacks[i].group_index);
         return false;
      }
      NIR_PASS(_, shader, vt_nir_lower_descriptors);
      struct vt_nir_rt_canonicalize_error error = {};
      if (vt_nir_canonicalize_rt_for_llvm(shader, &error) !=
          VT_NIR_RT_CANONICALIZE_SUCCESS) {
         mesa_loge("ventus-vk: global candidate callback has non-SSA temporary %s",
                   error.variable_name ? error.variable_name : "<unnamed>");
         return false;
      }
      nir_validate_shader(shader, "after Ventus global candidate callback lowering");

      llvm::Module *module = vt_nir_to_llvm_translate(&shader, 1,
                                                       &callback_options);
      if (!module)
         return false;
      char *path = NULL;
      uint64_t entry = 0;
      const bool emitted = vtvk_maybe_emit_rt_kernel_module(
         module, pipeline->rt_uses_rv64, &path, &entry);
      delete module;
      if (!emitted)
         return false;
      pipeline->callback_kernel_paths[i] = path;
      pipeline->callback_kernel_entries[i] = entry;
      pipeline->callback_kernel_groups[i] = callbacks[i].group_index;
      pipeline->callback_kernel_stages[i] = callbacks[i].stage;
      pipeline->callback_kernel_lds_sizes[i] = callback_lds_size;
   }
   mesa_logi("ventus-vk: prepared %u global RT candidate callback kernel%s",
             pipeline->callback_kernel_count,
             pipeline->callback_kernel_count == 1 ? "" : "s");
   return true;
}

static VkResult
vtvk_create_one_rt_pipeline(VkDevice _device,
                            VkPipelineCreateFlags2KHR pipeline_flags,
                            const VkRayTracingPipelineCreateInfoKHR *info,
                            const VkAllocationCallbacks *pAllocator,
                            VkPipeline *out)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);

   struct vtvk_rt_execution_profile_config profile_config;
   VkResult profile_result = vtvk_parse_rt_execution_profile(info, &profile_config);
   if (profile_result != VK_SUCCESS)
      return vk_error(device, profile_result);

   struct vtvk_pipeline *pipeline =
      (struct vtvk_pipeline *)vk_pipeline_zalloc(
         &device->vk, &vtvk_pipeline_ops, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
         pipeline_flags, pAllocator, sizeof(*pipeline));
   if (!pipeline)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   if (!vtvk_get_pipeline_layout(info->layout, &pipeline->resource_layout)) {
      vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
      return vk_error(device, VK_ERROR_INITIALIZATION_FAILED);
   }
   pipeline->nir_mem_ctx = ralloc_context(nullptr);
   pipeline->shader_count = info->stageCount;
   pipeline->max_ray_recursion_depth = info->maxPipelineRayRecursionDepth;
   pipeline->rt_execution_profile = profile_config.profile;
   pipeline->rt_stage_role = profile_config.stage_role;
   pipeline->rt_state_descriptor_set = profile_config.state_descriptor_set;
   pipeline->rt_state_descriptor_binding = profile_config.state_descriptor_binding;
   pipeline->rt_uses_rv64 = vtvk_env_enabled("VENTUS_VK_RT_RV64");

   pipeline->shaders =
      (nir_shader **)vk_zalloc2(&device->vk.alloc, pAllocator,
                                sizeof(nir_shader *) * info->stageCount, 8,
                                VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   pipeline->group_handles =
      (uint8_t *)vk_zalloc2(&device->vk.alloc, pAllocator,
                            32 * info->groupCount, 8,
                            VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);

   if (!pipeline->nir_mem_ctx || !pipeline->shaders || !pipeline->group_handles) {
      vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   for (uint32_t s = 0; s < info->stageCount; s++) {
      VkResult result =
         vtvk_compile_rt_stage(device, pipeline, pipeline_flags,
                               &info->pStages[s], &pipeline->shaders[s]);
      if (result != VK_SUCCESS) {
         vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
         return result;
      }

      pipeline->vk.stages |= info->pStages[s].stage;
   }

   vtvk_rt_callback_shaders cb_shaders = {};
   vtvk_collect_rt_callback_shaders(info, pipeline->shaders, pipeline->shader_count,
                                   &cb_shaders);

   vtvk_rt_infer_missing_payload_locations(pipeline->shaders,
                                           pipeline->shader_count);

   struct vt_nir_rt_abi_layout rt_abi = vt_nir_rt_default_abi_layout();
   bool triangle_barycentrics = false;
   uint32_t hit_attribute_rest = 0;
   if (!vtvk_rt_compute_hit_attribute_rest(
          pipeline->shaders, pipeline->shader_count, &triangle_barycentrics,
          &hit_attribute_rest) ||
       !vt_nir_rt_abi_layout_set_hit_attrib_size(
          &rt_abi, hit_attribute_rest, triangle_barycentrics)) {
      mesa_loge("ventus-vk: RT hit-attribute layout exceeds ABI limits");
      vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
   }
   uint32_t rt_payload_size = 0;
   if (!vt_nir_rt_compute_pipeline_payload_size(pipeline->shaders,
                                                pipeline->shader_count,
                                                &rt_payload_size) ||
       !vt_nir_rt_abi_layout_set_payload_size(&rt_abi, rt_payload_size) ||
       !vt_nir_rt_abi_layout_set_recursion_depth(
          &rt_abi, pipeline->max_ray_recursion_depth)) {
      mesa_loge("ventus-vk: RT payload layout exceeds the v3 private ABI");
      vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
   }
   if (vtvk_env_enabled("VENTUS_VK_NIR_PROBE"))
      mesa_logi("ventus-vk: RT ABI hitattr=%u bary=%u payload=%u depth=%u nested_frames=%u total=%u",
                rt_abi.hit_attrib_size_bytes,
                rt_abi.hit_attrib_barycentrics ? 1 : 0,
                rt_abi.payload_size_bytes, rt_abi.max_ray_recursion_depth,
                rt_abi.payload_spill_size_bytes, rt_abi.total_size_bytes);

   const bool staged_dispatch =
      pipeline->rt_execution_profile ==
      VK_VENTUS_RT_EXECUTION_PROFILE_STAGED_DISPATCH;
   /* The profile, rather than an ambient environment variable, selects the
    * ABI-changing lower path.  Staged dispatch is the isolated legacy ray-ref queue
    * implementation; a normal pipeline is always the synchronous
    * megakernel path. */
   const bool global_wavefront = staged_dispatch;
   if (vtvk_env_enabled("VENTUS_VK_RT_WAVEFRONT_GLOBAL") ||
       vtvk_env_enabled("VENTUS_VK_RT_WAVEFRONT_MIRROR")) {
      mesa_logw("ventus-vk: legacy wavefront environment selection is ignored; "
                "use the staged-dispatch execution-profile pNext to select it");
   }

   {
      struct vt_nir_rt_config config = vt_nir_rt_default_config();
      vt_nir_rt_config_apply_abi_layout(&config, &rt_abi);
      vtvk_configure_rt_group(&config);

      config.defer_intersection_reports = true;
      if (global_wavefront) {
         vt_nir_rt_config_apply_global_cps_layout(&config, &rt_abi);
         config.execution_model = VT_RT_EXECUTION_WAVEFRONT_GLOBAL_WORKER;
      }

      for (uint32_t s = 0; s < pipeline->shader_count; s++) {
         vtvk_rt_lower_abi_with_probe(pipeline->shaders[s], &config);
         nir_validate_shader(pipeline->shaders[s], "after Ventus RT lowering");
         if (vtvk_env_enabled("VENTUS_VK_DUMP_NIR"))
            nir_print_shader(pipeline->shaders[s], stderr);
      }
   }

   {
      struct vt_nir_rt_config rt_io_config = vt_nir_rt_default_config();
      vt_nir_rt_config_apply_abi_layout(&rt_io_config, &rt_abi);
      vtvk_configure_rt_group(&rt_io_config);
      rt_io_config.defer_intersection_reports = true;
      if (global_wavefront) {
         vt_nir_rt_config_apply_global_cps_layout(&rt_io_config, &rt_abi);
         rt_io_config.execution_model = VT_RT_EXECUTION_WAVEFRONT_GLOBAL_WORKER;
      }
      vtvk_rt_pipeline_callback_data cb_data = {
         .shaders = &cb_shaders,
      };
      if (global_wavefront) {
         /* Keep the terminal SBT dispatch in the root continuation.  CPS
          * sees the original TraceRayEXT, then resumes into the dynamically
          * selected miss/closest-hit code; a TraceRayEXT in closest-hit is
          * therefore another root CPS call site, not a child callback path. */
         rt_io_config.enable_monolithic_shader_callbacks = true;
         rt_io_config.emit_miss = cb_shaders.miss.empty() ? NULL : vtvk_emit_rt_miss;
         rt_io_config.emit_closest_hit =
            cb_shaders.closest_hit.empty() ? NULL : vtvk_emit_rt_closest_hit;
         rt_io_config.callback_data = &cb_data;
         if (!vt_nir_lower_rt_global_root_continuations(
                cb_shaders.raygen, &rt_io_config)) {
            mesa_loge("ventus-vk: global root continuation callback lowering failed");
            vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
            return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
         }
         struct vt_nir_rt_config global_worker_config = rt_io_config;
         global_worker_config.enable_monolithic_shader_callbacks = false;
         global_worker_config.emit_miss = NULL;
         global_worker_config.emit_closest_hit = NULL;
         global_worker_config.emit_any_hit = NULL;
         global_worker_config.emit_intersection = NULL;
         global_worker_config.callback_data = NULL;
         vtvk_rt_log_nir_stats(cb_shaders.raygen,
                               "raygen before global wavefront producer");
         struct vt_nir_rt_cps_info cps = {};
         if (!vt_nir_lower_rt_cps(cb_shaders.raygen, &cps,
                                  pipeline->nir_mem_ctx)) {
            mesa_loge("ventus-vk: global wavefront requires traceRayEXT CPS lowering");
            vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
            return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
         }
         pipeline->resume_shaders = cps.resume_shaders;
         pipeline->resume_shader_count = cps.num_resume_shaders;
         pipeline->continuation_frame_size = cps.frame_size_bytes;
         if (!vt_nir_lower_rt_global_raygen(cb_shaders.raygen,
                                             &global_worker_config)) {
            mesa_loge("ventus-vk: global wavefront CPS producer lowering failed");
            vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
            return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
         }
         for (uint32_t i = 0; i < pipeline->resume_shader_count; i++) {
            if (!vt_nir_lower_rt_global_resume(pipeline->resume_shaders[i],
                                               &global_worker_config)) {
               mesa_loge("ventus-vk: global wavefront resume %u lowering failed", i);
               vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
               return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
            }
            NIR_PASS(_, pipeline->resume_shaders[i], vt_nir_lower_descriptors);
            struct vt_nir_rt_canonicalize_error resume_canonicalize_error = {};
            if (vt_nir_canonicalize_rt_for_llvm(
                   pipeline->resume_shaders[i], &resume_canonicalize_error) !=
                VT_NIR_RT_CANONICALIZE_SUCCESS) {
               mesa_loge("ventus-vk: global RT resume %u has non-SSA function "
                         "temporary %s", i,
                         resume_canonicalize_error.variable_name ?
                            resume_canonicalize_error.variable_name : "<unnamed>");
               vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
               return vk_error(device, VK_ERROR_INITIALIZATION_FAILED);
            }
            nir_validate_shader(pipeline->resume_shaders[i],
                                "after Ventus global RT resume lowering");
         }
         vtvk_rt_log_nir_stats(cb_shaders.raygen,
                               "raygen after global wavefront producer");
      } else {
         rt_io_config.enable_monolithic_shader_callbacks = true;
         rt_io_config.emit_miss = cb_shaders.miss.empty() ? NULL : vtvk_emit_rt_miss;
         rt_io_config.emit_closest_hit =
            cb_shaders.closest_hit.empty() ? NULL : vtvk_emit_rt_closest_hit;
         rt_io_config.emit_any_hit =
            cb_shaders.any_hit.empty() ? NULL : vtvk_emit_rt_any_hit;
         rt_io_config.emit_intersection =
            cb_shaders.intersection.empty() ? NULL : vtvk_emit_rt_intersection;
         rt_io_config.callback_data = &cb_data;

         for (const vtvk_rt_callback_shader_entry &intersection :
              cb_shaders.intersection) {
            nir_shader *matching_any_hit = NULL;
            for (const vtvk_rt_callback_shader_entry &any_hit : cb_shaders.any_hit) {
               if (any_hit.group_index == intersection.group_index) {
                  matching_any_hit = any_hit.shader;
                  break;
               }
            }
            vtvk_lower_rt_intersection_reports(intersection.shader, &rt_io_config,
                                               matching_any_hit);
         }

         vtvk_rt_log_nir_stats(cb_shaders.raygen,
                               "raygen before vt_nir_lower_rt_io_monolithic");
         NIR_PASS(_, cb_shaders.raygen, vt_nir_lower_rt_io_monolithic,
                  &rt_io_config);
         // The monolithic trace lowering materializes the RT-local fields,
         // but does not establish their raygen lifecycle state.  In
         // particular trace_meta1 is a packed word: its first miss-index
         // update must preserve a defined active-level (zero), rather than
         // read-modify-write an uninitialized RT-local SRAM word.
         NIR_PASS(_, cb_shaders.raygen, vt_nir_lower_rt_abi, &rt_io_config);
         vtvk_rt_log_nir_stats(cb_shaders.raygen,
                               "raygen after vt_nir_lower_rt_io_monolithic");
      }
      NIR_PASS(_, cb_shaders.raygen, vt_nir_lower_descriptors);
      vtvk_rt_log_nir_stats(cb_shaders.raygen,
                            "raygen after Ventus RT descriptor lowering");

      struct vt_nir_rt_canonicalize_error canonicalize_error = {};
      if (vt_nir_canonicalize_rt_for_llvm(cb_shaders.raygen,
                                          &canonicalize_error) !=
          VT_NIR_RT_CANONICALIZE_SUCCESS) {
         mesa_loge("ventus-vk: RT raygen has non-SSA function temporary %s; "
                   "dynamic or aliased locals are not supported yet",
                   canonicalize_error.variable_name ?
                      canonicalize_error.variable_name : "<unnamed>");
         vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
         return vk_error(device, VK_ERROR_INITIALIZATION_FAILED);
      }
      nir_validate_shader(cb_shaders.raygen,
                          "after Ventus RT raygen canonicalization");
   }

   /* Callback stages have been inlined into raygen.  Translating their
    * original NIR as well would compile dead entry points and apply the
    * per-lane-local legality check to code that is never executed. */
   nir_shader *monolithic_shaders[] = {cb_shaders.raygen};
   pipeline->per_lane_scratch_size =
      vt_nir_pipeline_compute_scratch(monolithic_shaders, 1,
                                      &pipeline->scratch_info);
   /* Keep PDS large enough for RT attributes, linked payload frames and their
    * fixed-depth spills.  The fixed 160-byte header is RT Local SRAM. */
   pipeline->per_lane_scratch_size =
      MAX2(pipeline->per_lane_scratch_size, rt_abi.private_base_bytes);
   pipeline->hit_attribute_size_bytes = rt_abi.hit_attrib_size_bytes;

   struct vt_nir_to_llvm_options llvm_options = {
      .module_name = "ventus_rt_pipeline",
      .target_triple = pipeline->rt_uses_rv64 ? "riscv64-unknown-elf" :
                                                "riscv32-unknown-elf",
      .data_layout = pipeline->rt_uses_rv64 ?
         "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128-A5-G1" :
         "e-m:e-p:32:32-i64:64-n32-S128",
      .require_ssa_function_temps = true,
      .global_rt_wavefront = global_wavefront,
      /* Megakernel private state follows PDS-backed payload frames.  The
       * staged/global legacy path keeps its own CPS storage. */
      .private_base_bytes = global_wavefront ? 0 : rt_abi.private_base_bytes,
   };

   pipeline->llvm_module =
      vt_nir_to_llvm_translate(monolithic_shaders, 1,
                               &llvm_options);
   if (!pipeline->llvm_module) {
      vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
      return vk_error(device, VK_ERROR_INITIALIZATION_FAILED);
   }
   vtvk_maybe_emit_rt_kernel(pipeline);
   if (!vtvk_emit_rt_resume_kernels(pipeline, &llvm_options)) {
      vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
      return vk_error(device, VK_ERROR_INITIALIZATION_FAILED);
   }
   if (global_wavefront &&
       !vtvk_emit_rt_global_candidate_kernels(pipeline, cb_shaders.any_hit,
                                               cb_shaders.intersection,
                                               &llvm_options)) {
      vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
      return vk_error(device, VK_ERROR_INITIALIZATION_FAILED);
   }

   vtvk_fill_group_handles(pipeline, info);

   mesa_logi("ventus-vk: RT pipeline compiled via SPIR-V -> NIR -> "
             "Ventus lowering -> LLVM; stages=%u groups=%u scratch=%u max_depth=%u llvm_functions=%zu",
             info->stageCount, info->groupCount,
             pipeline->per_lane_scratch_size,
             pipeline->max_ray_recursion_depth,
             pipeline->llvm_module->size());

   *out = vk_pipeline_to_handle(&pipeline->vk);
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_CreateRayTracingPipelinesKHR(
   VkDevice device,
   VkDeferredOperationKHR deferredOperation,
   VkPipelineCache pipelineCache,
   uint32_t createInfoCount,
   const VkRayTracingPipelineCreateInfoKHR *pCreateInfos,
   const VkAllocationCallbacks *pAllocator,
   VkPipeline *pPipelines)
{
   (void)deferredOperation;
   (void)pipelineCache;

   VkResult final_result = VK_SUCCESS;

   for (uint32_t i = 0; i < createInfoCount; i++) {
      VkPipelineCreateFlags2KHR flags = vk_rt_pipeline_create_flags(&pCreateInfos[i]);
      VkResult result =
         vtvk_create_one_rt_pipeline(device, flags, &pCreateInfos[i],
                                     pAllocator, &pPipelines[i]);
      if (result != VK_SUCCESS) {
         pPipelines[i] = VK_NULL_HANDLE;
         final_result = result;

         if (!(pCreateInfos[i].flags &
               VK_PIPELINE_CREATE_EARLY_RETURN_ON_FAILURE_BIT))
            continue;

         for (uint32_t j = i + 1; j < createInfoCount; j++)
            pPipelines[j] = VK_NULL_HANDLE;
         break;
      }
   }

   return final_result;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_GetRayTracingShaderGroupHandlesKHR(VkDevice device,
                                        VkPipeline pipeline,
                                        uint32_t firstGroup,
                                        uint32_t groupCount,
                                        size_t dataSize,
                                        void *pData)
{
   (void)device;
   struct vtvk_pipeline *vt_pipeline = vtvk_pipeline_from_handle(pipeline);
   const uint32_t handle_size = 32;

   if (!vt_pipeline || firstGroup > vt_pipeline->group_count ||
       groupCount > vt_pipeline->group_count - firstGroup)
      return VK_ERROR_UNKNOWN;

   size_t copy_size = (size_t)groupCount * handle_size;
   if (dataSize < copy_size)
      return VK_ERROR_UNKNOWN;

   memcpy(pData, vt_pipeline->group_handles + firstGroup * handle_size,
          copy_size);
   return VK_SUCCESS;
}
