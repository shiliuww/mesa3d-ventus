#include "vtvk_pipeline.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "compiler/spirv/nir_spirv.h"
#include "compiler/glsl_types.h"
#include "nir/nir.h"
#include "nir/nir_deref.h"
#include "util/hash_table.h"
#include "util/log.h"
#include "util/ralloc.h"
#include "vk_pipeline.h"
#include "vk_util.h"
#include "vtvk_entrypoints.h"
#include "ventus/common/ventus_kernel_resource.h"
#include "ventus/common/ventus_rt_v5.h"
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

struct vtvk_rt_nir_stats {
   unsigned trace_ray;
   unsigned execute_callable;
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
   case nir_intrinsic_execute_callable:
      stats->execute_callable++;
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

   mesa_logi("ventus-vk: NIR %s stage=%s trace_ray=%u execute_callable=%u "
             "desc_load=%u "
             "res_index=%u res_reindex=%u load_deref=%u store_deref=%u "
             "ubo=%u ssbo=%u image_store=%u ray_world=(%u,%u) "
             "ray_object=(%u,%u) ray_t=(%u,%u) primitive_id=%u "
             "vars={func:%u call:%u hitattr:%u} scratch=%u",
             checkpoint, _mesa_shader_stage_to_string(shader->info.stage),
             stats.trace_ray, stats.execute_callable,
             stats.load_vulkan_descriptor,
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
      MAX2(shader->scratch_size, config->scratchpad[layout].total_size_bytes);
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

   if (config->use_v5_pds &&
       shader->info.stage == MESA_SHADER_RAYGEN) {
      /* Resident terminal callbacks are already inlined here.  Their
       * stage-specific call-data and hit-attribute variables are temporarily
       * invalid in a raygen shader until the complete ABI batch consumes
       * both classes, so do not validate between the individual subpasses. */
      vt_nir_lower_rt_payload(shader, config);
      vt_nir_lower_rt_hit_record(shader, config);
      vt_nir_lower_rt_control(shader, config);
      vt_nir_lower_shader_record_ptr(shader, config);
      vt_nir_lower_shader_scratch(shader, config);
      nir_remove_dead_variables(shader,
                                nir_var_shader_temp |
                                   nir_var_shader_call_data |
                                   nir_var_ray_hit_attrib,
                                NULL);
      vtvk_rt_finalize_abi(shader, config);
      vtvk_rt_log_nir_stats(shader, "after resident V5 RT ABI lowering");
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

   NIR_PASS(_, shader, nir_remove_dead_variables,
            nir_var_shader_temp | nir_var_shader_call_data |
               nir_var_ray_hit_attrib,
            NULL);

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

static void
vtvk_remove_rt_artifact(const std::string &path)
{
   if (path.empty() || unlink(path.c_str()) == 0 || errno == ENOENT)
      return;

   mesa_logw("ventus-vk: failed to remove RT compiler artifact %s: %s",
             path.c_str(), strerror(errno));
}

struct vtvk_rt_emit_artifacts {
   explicit vtvk_rt_emit_artifacts(const std::string &base,
                                   bool retain_all,
                                   bool retain_llvm_dump)
      : ll_path(base + ".ll"),
        emit_ll_path(base + ".emit.ll"),
        obj_path(base + ".o"),
        elf_path(base + ".riscv"),
        ld_path(base + ".ld"),
        retain_all(retain_all),
        retain_ll(retain_all || retain_llvm_dump)
   {}

   ~vtvk_rt_emit_artifacts()
   {
      if (retain_all)
         return;

      if (!retain_ll)
         vtvk_remove_rt_artifact(ll_path);
      vtvk_remove_rt_artifact(emit_ll_path);
      vtvk_remove_rt_artifact(obj_path);
      vtvk_remove_rt_artifact(ld_path);
      if (!retain_elf)
         vtvk_remove_rt_artifact(elf_path);
   }

   std::string ll_path;
   std::string emit_ll_path;
   std::string obj_path;
   std::string elf_path;
   std::string ld_path;
   bool retain_all;
   bool retain_ll;
   bool retain_elf = false;
};

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
      return "";
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

   mesa_loge("ventus-vk: RT ELF needs Ventus crt0.o startup; set "
             "VENTUS_VK_CRT0 or build LLVM libclc");
   return false;
}

static bool
vtvk_dump_rt_emit_llvm_module(llvm::Module *module,
                              const std::string &emit_ll_path)
{
   return vtvk_dump_rt_llvm_module_to(module, emit_ll_path, "ELF-input");
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
      llvm_nm = "llvm-nm";
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
                                 bool retain_artifacts, bool force_elf,
                                 bool check_v5_frontier,
                                 char **out_path, uint64_t *out_entry)
{
   *out_path = NULL;
   *out_entry = 0;
   const bool dump_llvm = vtvk_env_enabled("VENTUS_VK_DUMP_LLVM");
   const bool emit_elf = force_elf ||
                         vtvk_env_enabled("VENTUS_VK_EMIT_ELF");
   if (!dump_llvm && !emit_elf)
      return true;

   vtvk_rt_emit_artifacts artifacts(vtvk_rt_artifact_base(),
                                    retain_artifacts, dump_llvm);
   if (!vtvk_dump_rt_llvm_module_to(module, artifacts.ll_path,
                                    "pre-lowering"))
      return false;
   if (!emit_elf)
      return true;

   std::string crt0_path;
   const char *llc = getenv("VENTUS_VK_LLC");
   const char *ld_lld = getenv("VENTUS_VK_LD_LLD");
   if (!llc || !llc[0])
      llc = "llc";
   if (!ld_lld || !ld_lld[0])
      ld_lld = "ld.lld";

   if (!vtvk_write_rt_linker_script(artifacts.ld_path))
      return false;
   if (!vtvk_find_crt0(rv64, &crt0_path))
      return false;

   if (!vtvk_dump_rt_emit_llvm_module(module, artifacts.emit_ll_path))
      return false;

   std::string tool_env = vtvk_llvm_tool_env_prefix();
   std::string llc_cmd = tool_env + vtvk_shell_quote(llc) +
                         (rv64 ? " -mtriple=riscv64-unknown-elf"
                               : " -mtriple=riscv32-unknown-elf") +
                         (rv64 ? " -mcpu=ventus-gpgpu-rv64"
                               : " -mcpu=ventus-gpgpu")
                         + (rv64 ? " -target-abi=lp64" : " -target-abi=ilp32") +
                         " -mattr=+m,+a,+f,+d,+v,+zfh"
                         " -filetype=obj " +
                         vtvk_shell_quote(artifacts.emit_ll_path.c_str()) +
                         " -o " +
                         vtvk_shell_quote(artifacts.obj_path.c_str());
   if (!vtvk_run_tool(llc_cmd, "llc"))
      return false;

   std::string link_cmd;
   if (!vtvk_rt_elf_link_command(tool_env, ld_lld, artifacts.ld_path,
                                 artifacts.elf_path, crt0_path,
                                 artifacts.obj_path, &link_cmd))
      return false;
   if (!vtvk_run_tool(link_cmd, "link"))
      return false;

   /* Check the final linked image before handing its path to the runtime.
    * V5 uses ordinary VGPR scoreboarding: an unchecked ELF cannot silently
    * fall back to a whole-warp RT busy mechanism. The deployment environment
    * supplies the checker from the matching Ventus toolchain. */
   if (check_v5_frontier) {
      const char *checker = getenv("VENTUS_VK_RT_COMPLETION_CHECKER");
      const char *objdump = getenv("VENTUS_VK_LLVM_OBJDUMP");
      if (!checker || !checker[0]) {
         mesa_loge("ventus-vk: V5 final ELF requires "
                   "VENTUS_VK_RT_COMPLETION_CHECKER");
         return false;
      }
      if (!objdump || !objdump[0])
         objdump = "llvm-objdump";
      const std::string check_cmd = tool_env + "python3 " +
         vtvk_shell_quote(checker) + " --mode strict --objdump " +
         vtvk_shell_quote(objdump) + " " +
         vtvk_shell_quote(artifacts.elf_path.c_str());
      if (!vtvk_run_tool(check_cmd, "V5 final ELF completion frontier"))
         return false;
      mesa_logi("ventus-vk: V5 final ELF completion frontier passed: %s",
                artifacts.elf_path.c_str());
   }

   uint64_t kernel_entry = 0;
   if (!vtvk_read_elf_symbol(artifacts.elf_path, "main", &kernel_entry))
      return false;

   *out_path = strdup(artifacts.elf_path.c_str());
   if (!*out_path)
      return false;
   artifacts.retain_elf = true;
   *out_entry = kernel_entry;
   mesa_logi("ventus-vk: emitted RT RISC-V kernel to %s entry=0x%" PRIx64,
             artifacts.elf_path.c_str(), kernel_entry);
   return true;
}

static bool
vtvk_maybe_emit_rt_kernel(struct vtvk_pipeline *pipeline)
{
   char *path = NULL;
   uint64_t entry = 0;
   if (!vtvk_maybe_emit_rt_kernel_module(pipeline->llvm_module,
                                         pipeline->rt_uses_rv64,
                                         pipeline->retain_generated_artifacts,
                                         pipeline->rt_single_elf_multi_entry,
                                         pipeline->rt_v5_enabled,
                                         &path, &entry))
      return false;
   if (pipeline->generated_kernel_path &&
       !pipeline->retain_generated_artifacts)
      vtvk_remove_rt_artifact(pipeline->generated_kernel_path);
   free(pipeline->generated_kernel_path);
   pipeline->generated_kernel_path = path;
   pipeline->generated_kernel_entry = entry;
   return true;
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
         module, pipeline->rt_uses_rv64,
         pipeline->retain_generated_artifacts, false, false, &path, &entry);
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
   nir_variable *callable_shader_record_ptr;
   uint32_t executable_index;
   const char *executable_symbol;
   uint32_t call_data_read_word_mask = 0;
   uint32_t call_data_write_word_mask = 0;
   bool call_data_summary_complete = false;
};

static bool
vtvk_rt_mark_call_data_range(uint32_t start, uint32_t bytes,
                             uint32_t *word_mask)
{
   if (!word_mask || bytes == 0 ||
       start >= VENTUS_RT_V5_CALL_DATA_CAPACITY_BYTES ||
       bytes > VENTUS_RT_V5_CALL_DATA_CAPACITY_BYTES - start)
      return false;
   const uint32_t first = start / sizeof(uint32_t);
   const uint32_t last = (start + bytes - 1) / sizeof(uint32_t);
   for (uint32_t word = first; word <= last; ++word)
      *word_mask |= 1u << word;
   return true;
}

static bool
vtvk_rt_mark_call_data_deref(nir_deref_instr *deref,
                             uint32_t components, uint32_t bit_size,
                             uint32_t component_mask, uint32_t *word_mask)
{
   nir_variable *var = deref ? nir_deref_instr_get_variable(deref) : nullptr;
   if (!var || var->data.mode != nir_var_shader_call_data ||
       !var->data.explicit_location || var->data.location < 0)
      return false;

   unsigned var_size = 0;
   unsigned var_align = 0;
   glsl_get_natural_size_align_bytes(var->type, &var_size, &var_align);
   const uint32_t var_base =
      (uint32_t)var->data.location * VT_RT_ABI_ALIGNMENT_BYTES;
   if (nir_deref_instr_has_indirect(deref))
      return vtvk_rt_mark_call_data_range(var_base, var_size, word_mask);

   const uint32_t deref_offset = nir_deref_instr_get_const_offset(
      deref, glsl_get_natural_size_align_bytes);
   const uint32_t component_bytes = DIV_ROUND_UP(bit_size, 8);
   for (uint32_t component = 0; component < components; ++component) {
      if ((component_mask & BITFIELD_BIT(component)) == 0)
         continue;
      if (!vtvk_rt_mark_call_data_range(
             var_base + deref_offset + component * component_bytes,
             component_bytes, word_mask))
         return false;
   }
   return true;
}

static bool
vtvk_rt_mark_full_call_data_var(nir_deref_instr *deref, uint32_t *word_mask)
{
   nir_variable *var = deref ? nir_deref_instr_get_variable(deref) : nullptr;
   if (!var || var->data.mode != nir_var_shader_call_data ||
       !var->data.explicit_location || var->data.location < 0)
      return false;
   unsigned size = 0;
   unsigned align = 0;
   glsl_get_natural_size_align_bytes(var->type, &size, &align);
   return vtvk_rt_mark_call_data_range(
      (uint32_t)var->data.location * VT_RT_ABI_ALIGNMENT_BYTES,
      size, word_mask);
}

static bool
vtvk_rt_compute_call_data_summary(nir_shader *shader, uint32_t *read_mask,
                                  uint32_t *write_mask)
{
   if (!shader || !read_mask || !write_mask)
      return false;
   *read_mask = 0;
   *write_mask = 0;

   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            if (intrin->intrinsic == nir_intrinsic_load_deref &&
                nir_src_is_deref(intrin->src[0])) {
               nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
               nir_variable *var = nir_deref_instr_get_variable(deref);
               if (var && var->data.mode == nir_var_shader_call_data &&
                   !vtvk_rt_mark_call_data_deref(
                      deref, intrin->def.num_components, intrin->def.bit_size,
                      BITFIELD_MASK(intrin->def.num_components), read_mask))
                  return false;
               continue;
            }
            if (intrin->intrinsic == nir_intrinsic_store_deref &&
                nir_src_is_deref(intrin->src[0])) {
               nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
               nir_variable *var = nir_deref_instr_get_variable(deref);
               if (var && var->data.mode == nir_var_shader_call_data &&
                   !vtvk_rt_mark_call_data_deref(
                      deref, nir_src_num_components(intrin->src[1]),
                      nir_src_bit_size(intrin->src[1]),
                      nir_intrinsic_write_mask(intrin), write_mask))
                  return false;
               continue;
            }

            const nir_intrinsic_info *info =
               &nir_intrinsic_infos[intrin->intrinsic];
            for (unsigned src = 0; src < info->num_srcs; ++src) {
               if (!nir_src_is_deref(intrin->src[src]))
                  continue;
               nir_deref_instr *deref = nir_src_as_deref(intrin->src[src]);
               nir_variable *var = nir_deref_instr_get_variable(deref);
               if (!var || var->data.mode != nir_var_shader_call_data)
                  continue;
               /* Calls and uncommon deref operations are conservatively
                * inout until interprocedural summaries are available. */
               if (!vtvk_rt_mark_full_call_data_var(deref, read_mask) ||
                   !vtvk_rt_mark_full_call_data_var(deref, write_mask))
                  return false;
            }
         }
      }
   }
   return true;
}

enum {
   VTVK_RT_CALLABLE_PARENT_DATA_OFFSET = VT_NIR_RT_FRAME_BODY_OFFSET,
   VTVK_RT_ENTRY_RECORD_PTR_OFFSET = VT_NIR_RT_FRAME_RECORD_PTR_OFFSET,
};

static bool
vtvk_resolve_rt_shader_entries(
   struct vtvk_pipeline *pipeline,
   const std::vector<vtvk_rt_callback_shader_entry> &entries)
{
   if (!pipeline->rt_single_elf_multi_entry)
      return true;
   if (!pipeline->generated_kernel_path) {
      mesa_loge("ventus-vk: single-ELF RT entries require an emitted kernel");
      return false;
   }

   const size_t count = entries.size();
   pipeline->rt_shader_entry_refs =
      (uint32_t *)calloc(count, sizeof(*pipeline->rt_shader_entry_refs));
   pipeline->rt_shader_entry_targets =
      (uint64_t *)calloc(count, sizeof(*pipeline->rt_shader_entry_targets));
   pipeline->rt_shader_entry_stages =
      (uint32_t *)calloc(count, sizeof(*pipeline->rt_shader_entry_stages));
   if (count && (!pipeline->rt_shader_entry_refs ||
                 !pipeline->rt_shader_entry_targets ||
                 !pipeline->rt_shader_entry_stages))
      return false;

   pipeline->rt_shader_entry_count = count;
   for (size_t i = 0; i < count; i++) {
      const vtvk_rt_callback_shader_entry &entry = entries[i];
      uint64_t target = 0;
      if (!entry.executable_symbol ||
          !vtvk_read_elf_symbol(pipeline->generated_kernel_path,
                                entry.executable_symbol, &target))
         return false;

      /* The current private SBT header stores the Vulkan shader-group index at
       * byte 4.  Treat it as an opaque entry_ref; only this table attaches a
       * code address to it. */
      pipeline->rt_shader_entry_refs[i] = entry.group_index;
      pipeline->rt_shader_entry_stages[i] = entry.stage;
      pipeline->rt_shader_entry_targets[i] = target;
      mesa_logi("ventus-vk: RT entry_ref=%u stage=%u group=%u symbol=%s "
                "target=0x%" PRIx64,
                entry.group_index, entry.stage, entry.group_index,
                entry.executable_symbol, target);
   }
   return true;
}

struct vtvk_rt_callback_shaders {
   nir_shader *raygen;
   std::vector<vtvk_rt_callback_shader_entry> miss;
   std::vector<vtvk_rt_callback_shader_entry> closest_hit;
   std::vector<vtvk_rt_callback_shader_entry> any_hit;
   std::vector<vtvk_rt_callback_shader_entry> intersection;
   std::vector<vtvk_rt_callback_shader_entry> callable;
};

static bool
vtvk_rt_compute_entry_call_data_summaries(vtvk_rt_callback_shaders *entries)
{
   if (!entries)
      return false;
   std::vector<vtvk_rt_callback_shader_entry> *groups[] = {
      &entries->miss, &entries->closest_hit, &entries->any_hit,
      &entries->intersection, &entries->callable,
   };
   for (auto *group : groups) {
      for (auto &entry : *group) {
         entry.call_data_summary_complete = vtvk_rt_compute_call_data_summary(
            entry.shader, &entry.call_data_read_word_mask,
            &entry.call_data_write_word_mask);
         if (!entry.call_data_summary_complete)
            return false;
         if (vtvk_env_enabled("VENTUS_VK_NIR_PROBE")) {
            mesa_logi("ventus-vk: RT entry group=%u stage=%s call-data "
                      "read=0x%06x write=0x%06x",
                      entry.group_index,
                      _mesa_shader_stage_to_string(entry.stage),
                      entry.call_data_read_word_mask,
                      entry.call_data_write_word_mask);
         }
      }
   }
   return true;
}

struct vtvk_rt_pipeline_callback_data {
   const vtvk_rt_callback_shaders *shaders;
   bool independent_entries;
};

struct vtvk_rt_resident_trace_shader {
   nir_shader *shader;
   vtvk_rt_pipeline_callback_data *callbacks;
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

static void
vtvk_rt_infer_missing_payload_locations(nir_shader **shaders,
                                         uint32_t shader_count)
{
   for (uint32_t i = 0; i < shader_count; i++) {
      nir_shader *shader = shaders[i];
      if (!shader)
         continue;

      nir_remove_dead_derefs(shader);
      nir_remove_dead_variables(shader,
                                nir_var_shader_temp |
                                   nir_var_shader_call_data,
                                NULL);
      nir_foreach_variable_in_shader(var, shader) {
         if (vtvk_rt_is_payload_variable(var) && var->data.location >= 0)
            var->data.explicit_location = true;
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
         if (location < 0)
            continue;

         var->data.location = location;
         var->data.explicit_location = true;
         if (vtvk_env_enabled("VENTUS_VK_NIR_PROBE")) {
            mesa_logi("ventus-vk: inferred RT payload location %d for %s",
                      location, var->name ? var->name : "<unnamed>");
         }
      }
   }

   /* glslang may omit Location on both sides of an interface when one stage
    * declares incoming and outgoing ray payloads together.  Resolve the
    * remaining named interfaces as pipeline-wide groups and place each group
    * in the first unoccupied 16-byte payload location. */
   struct unresolved_payload_group {
      const char *name;
      unsigned size;
      unsigned align;
      std::vector<nir_variable *> variables;
   };
   std::vector<unresolved_payload_group> unresolved;
   for (uint32_t i = 0; i < shader_count; ++i) {
      nir_shader *shader = shaders[i];
      if (!shader)
         continue;
      nir_foreach_variable_in_shader(var, shader) {
         if (!vtvk_rt_is_payload_variable(var) || var->data.location >= 0 ||
             !var->name)
            continue;
         unsigned size = 0;
         unsigned align = 0;
         glsl_get_natural_size_align_bytes(var->type, &size, &align);
         auto group = std::find_if(
            unresolved.begin(), unresolved.end(), [&](const auto &candidate) {
               return candidate.size == size && candidate.align == align &&
                      strcmp(candidate.name, var->name) == 0;
            });
         if (group == unresolved.end()) {
            unresolved.push_back({var->name, size, align, {var}});
         } else {
            group->variables.push_back(var);
         }
      }
   }

   bool occupied[32] = {};
   for (const auto &candidate : candidates) {
      const unsigned slots = DIV_ROUND_UP(
         candidate.size, VT_RT_ABI_ALIGNMENT_BYTES);
      for (unsigned slot = 0;
           slot < slots && candidate.location + (int)slot < 32; ++slot)
         occupied[candidate.location + slot] = true;
   }
   for (auto &group : unresolved) {
      if (group.variables.size() < 2)
         continue;
      const unsigned slots = DIV_ROUND_UP(
         group.size, VT_RT_ABI_ALIGNMENT_BYTES);
      int location = -1;
      for (unsigned candidate = 0; candidate + slots <= 32; ++candidate) {
         bool available = true;
         for (unsigned slot = 0; slot < slots; ++slot)
            available &= !occupied[candidate + slot];
         if (available) {
            location = candidate;
            break;
         }
      }
      if (location < 0)
         continue;
      for (unsigned slot = 0; slot < slots; ++slot)
         occupied[location + slot] = true;
      for (nir_variable *var : group.variables) {
         var->data.location = location;
         var->data.explicit_location = true;
      }
      if (vtvk_env_enabled("VENTUS_VK_NIR_PROBE")) {
         mesa_logi("ventus-vk: inferred RT payload location %d for "
                   "unlocated interface %s", location, group.name);
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
   /* The local SPIR-V/NIR path may represent hitAttributeEXT vec2 as a
    * padded vec3.  Only its first two components belong to the V4 record. */
   return var->data.driver_location == 0 && size >= 8 && align >= 4;
}

static bool
vtvk_rt_compute_hit_attribute_rest(nir_shader **shaders,
                                   uint32_t shader_count,
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
         if (var->data.mode != nir_var_ray_hit_attrib ||
             vtvk_rt_is_triangle_barycentrics(shader, var, has_intersection))
            continue;

         unsigned size = 0, align = 0;
         glsl_get_natural_size_align_bytes(var->type, &size, &align);
         if (var->data.driver_location < 0)
            return false;
         const uint64_t finish =
            (uint64_t)(uint32_t)var->data.driver_location + size;
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
vtvk_localize_rt_callback_halts(nir_shader *shader)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   if (!impl)
      return;

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_jump)
            continue;

         nir_jump_instr *jump = nir_instr_as_jump(instr);
         if (jump->type == nir_jump_halt)
            jump->type = nir_jump_return;
      }
   }
}

static void
vtvk_inline_rt_callback_shader(nir_builder *b, nir_shader *shader)
{
   if (!b || !shader)
      return;

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   if (!impl)
      return;

   /* OpIgnoreIntersectionKHR and OpTerminateRayKHR end the current any-hit
    * invocation.  Their NIR halt must become a callback-local return before
    * this shader is inlined, otherwise it terminates the enclosing raygen. */
   vtvk_localize_rt_callback_halts(shader);
   nir_opt_dead_cf(shader);
   nir_lower_returns(shader);
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

   for (size_t i = 0; i < entries.size(); i++) {
      nir_push_if(b, nir_ieq_imm(b, index, entries[i].group_index));
      vtvk_inline_rt_callback_shader(b, entries[i].shader);
      if (i + 1 < entries.size())
         nir_push_else(b, NULL);
   }
   for (size_t i = 0; i < entries.size(); i++)
      nir_pop_if(b, NULL);
}

static bool
vtvk_rt_shader_has_intrinsic(nir_shader *shader, nir_intrinsic_op op)
{
   if (!shader)
      return false;

   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == op)
               return true;
         }
      }
   }

   return false;
}

static nir_def *
vtvk_load_rt_dispatch_arg(nir_builder *b, uint32_t offset, unsigned bit_size)
{
   nir_intrinsic_instr *intrin =
      nir_intrinsic_instr_create(b->shader,
                                 nir_intrinsic_ventus_rt_dispatch_arg);
   intrin->src[0] = nir_src_for_ssa(nir_imm_int(b, offset));
   nir_def_init(&intrin->instr, &intrin->def, 1, bit_size);
   nir_builder_instr_insert(b, &intrin->instr);
   return &intrin->def;
}

struct vtvk_rt_callable_payload_binding {
   nir_variable *variable;
   int location;
};

struct vtvk_rt_execute_callable_lower_data {
   const std::vector<vtvk_rt_callback_shader_entry> *callables;
   std::vector<vtvk_rt_callable_payload_binding> payload_bindings;
   bool failed;
};

static bool
vtvk_rt_variable_is_entrypoint_local(nir_shader *shader, nir_variable *var,
                                     nir_function_impl *entrypoint)
{
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_deref)
               continue;

            nir_deref_instr *deref = nir_instr_as_deref(instr);
            if (deref->deref_type == nir_deref_type_var &&
                deref->var == var && impl != entrypoint)
               return false;
         }
      }
   }

   return true;
}

static bool
vtvk_rt_prepare_callable_payload(
   nir_shader *shader, nir_variable *var,
   std::vector<vtvk_rt_callable_payload_binding> *bindings,
   int *location_out)
{
   for (const vtvk_rt_callable_payload_binding &binding : *bindings) {
      if (binding.variable == var) {
         *location_out = binding.location;
         return true;
      }
   }

   if (!shader || !var ||
       (var->data.mode != nir_var_shader_temp &&
        var->data.mode != nir_var_shader_call_data) ||
       !var->data.explicit_location || var->data.location < 0 ||
       var->data.location >= 32)
      return false;

   if (var->data.mode == nir_var_shader_call_data) {
      bindings->push_back({var, var->data.location});
      *location_out = var->data.location;
      return true;
   }

   nir_function_impl *entrypoint = nir_shader_get_entrypoint(shader);
   if (!entrypoint ||
       !vtvk_rt_variable_is_entrypoint_local(shader, var, entrypoint))
      return false;

   const int location = var->data.location;
   exec_node_remove(&var->node);
   var->data.mode = nir_var_function_temp;
   var->data.explicit_location = false;
   var->data.location = -1;
   var->data.driver_location = 0;
   exec_list_push_tail(&entrypoint->locals, &var->node);
   nir_fixup_deref_modes(shader);

   bindings->push_back({var, location});
   *location_out = location;
   return true;
}

static bool
vtvk_rt_callable_types_match(const nir_variable *caller,
                             const nir_variable *callee)
{
   unsigned caller_size = 0;
   unsigned caller_align = 0;
   unsigned callee_size = 0;
   unsigned callee_align = 0;
   glsl_get_natural_size_align_bytes(caller->type, &caller_size, &caller_align);
   glsl_get_natural_size_align_bytes(callee->type, &callee_size, &callee_align);
   return caller_size == callee_size && caller_align == callee_align;
}

static bool
vtvk_lower_rt_callable_shader_record_ptr(nir_builder *b,
                                         nir_intrinsic_instr *intrin,
                                         void *data)
{
   if (intrin->intrinsic != nir_intrinsic_load_shader_record_ptr)
      return false;

   nir_variable *shader_record_ptr = static_cast<nir_variable *>(data);
   if (!shader_record_ptr)
      return false;

   b->cursor = nir_before_instr(&intrin->instr);
   nir_def *selected_record_ptr = nir_load_var(b, shader_record_ptr);
   nir_def_rewrite_uses(&intrin->def, selected_record_ptr);
   nir_instr_remove(&intrin->instr);
   return true;
}

static bool
vtvk_prepare_rt_callable_shader_record_ptr(
   nir_shader *shader, nir_variable **shader_record_ptr_out)
{
   *shader_record_ptr_out = nullptr;
   if (!vtvk_rt_shader_has_intrinsic(
          shader, nir_intrinsic_load_shader_record_ptr))
      return true;

   nir_variable *shader_record_ptr =
      nir_variable_create(shader, nir_var_shader_temp, glsl_uint64_t_type(),
                          "ventus_callable_shader_record_ptr");
   if (!shader_record_ptr)
      return false;

   const bool progress = nir_shader_intrinsics_pass(
      shader, vtvk_lower_rt_callable_shader_record_ptr, nir_metadata_none,
      shader_record_ptr);
   if (!progress ||
       vtvk_rt_shader_has_intrinsic(
          shader, nir_intrinsic_load_shader_record_ptr))
      return false;

   nir_fixup_deref_modes(shader);
   nir_opt_dce(shader);
   *shader_record_ptr_out = shader_record_ptr;
   return true;
}

static bool
vtvk_lower_rt_entry_shader_record_ptr(nir_builder *b,
                                      nir_intrinsic_instr *intrin,
                                      void *data)
{
   (void)data;
   if (intrin->intrinsic != nir_intrinsic_load_shader_record_ptr)
      return false;

   b->cursor = nir_before_instr(&intrin->instr);
   nir_def *frame = vt_nir_rt_v5_current_continuation_frame(b);
   nir_def *record_ptr = nir_load_global(
      b, 1, 64,
      nir_iadd_imm(b, frame, VTVK_RT_ENTRY_RECORD_PTR_OFFSET),
      .align_mul = 8);
   nir_def_rewrite_uses(&intrin->def, record_ptr);
   nir_instr_remove(&intrin->instr);
   return true;
}

static const char *
vtvk_rt_entry_symbol_prefix(mesa_shader_stage stage)
{
   switch (stage) {
   case MESA_SHADER_MISS:
      return "miss";
   case MESA_SHADER_CLOSEST_HIT:
      return "closest_hit";
   case MESA_SHADER_ANY_HIT:
      return "any_hit";
   case MESA_SHADER_INTERSECTION:
      return "intersection";
   case MESA_SHADER_CALLABLE:
      return "callable";
   default:
      return nullptr;
   }
}

static bool
vtvk_prepare_rt_independent_entries(
   std::vector<vtvk_rt_callback_shader_entry> &entries,
   void *mem_ctx, std::vector<nir_shader *> *entry_shaders,
   std::vector<const char *> *entry_symbols, const char *symbol_suffix)
{
   for (size_t i = 0; i < entries.size(); i++) {
      vtvk_rt_callback_shader_entry &entry = entries[i];
      for (size_t j = 0; j < i; j++) {
         if (entries[j].shader == entry.shader) {
            entry.executable_index = entries[j].executable_index;
            entry.executable_symbol = entries[j].executable_symbol;
            break;
         }
      }
      if (entry.executable_index != UINT32_MAX)
         continue;

      const char *prefix = vtvk_rt_entry_symbol_prefix(entry.stage);
      if (!prefix)
         return false;
      const char *symbol = ralloc_asprintf(
         mem_ctx, "vt_%s_g%u%s", prefix, entry.group_index,
         symbol_suffix ? symbol_suffix : "");
      if (!symbol)
         return false;
      entry.executable_index = entry_shaders->size();
      entry.executable_symbol = symbol;
      entry_shaders->push_back(entry.shader);
      entry_symbols->push_back(symbol);

      vtvk_localize_rt_callback_halts(entry.shader);
      NIR_PASS(_, entry.shader, nir_shader_intrinsics_pass,
               vtvk_lower_rt_entry_shader_record_ptr,
               nir_metadata_none, NULL);
      if (vtvk_rt_shader_has_intrinsic(
             entry.shader, nir_intrinsic_load_shader_record_ptr)) {
         mesa_loge("ventus-vk: failed to bind shader record data for "
                   "independent entry %s", symbol);
         return false;
      }
   }
   return true;
}

static bool
vtvk_prepare_rt_callable_entries(
   std::vector<vtvk_rt_callback_shader_entry> &callables,
   void *mem_ctx, std::vector<nir_shader *> *entry_shaders,
   std::vector<const char *> *entry_symbols)
{
   for (const vtvk_rt_callback_shader_entry &entry : callables) {
      if (vtvk_rt_shader_has_intrinsic(
             entry.shader, nir_intrinsic_trace_ray)) {
         mesa_loge("ventus-vk: callable-originated traceRayEXT is not "
                   "supported");
         return false;
      }
   }
   return vtvk_prepare_rt_independent_entries(
      callables, mem_ctx, entry_shaders, entry_symbols, NULL);
}

static bool
vtvk_clone_rt_callable_depth_entries(
   const std::vector<vtvk_rt_callback_shader_entry> &source,
   unsigned depth, void *mem_ctx,
   std::vector<vtvk_rt_callback_shader_entry> *clones,
   std::vector<nir_shader *> *entry_shaders,
   std::vector<const char *> *entry_symbols)
{
   clones->clear();
   for (size_t i = 0; i < source.size(); ++i) {
      nir_shader *clone = NULL;
      for (size_t j = 0; j < i; ++j) {
         if (source[j].shader == source[i].shader) {
            clone = (*clones)[j].shader;
            break;
         }
      }
      if (!clone)
         clone = nir_shader_clone(mem_ctx, source[i].shader);
      if (!clone)
         return false;

      clones->push_back({source[i].group_index, source[i].stage, clone,
                         nullptr, UINT32_MAX, nullptr,
                         source[i].call_data_read_word_mask,
                         source[i].call_data_write_word_mask,
                         source[i].call_data_summary_complete});
   }

   const char *suffix = ralloc_asprintf(mem_ctx, "_depth%u", depth);
   return suffix && vtvk_prepare_rt_independent_entries(
                       *clones, mem_ctx, entry_shaders, entry_symbols, suffix);
}

static bool
vtvk_clone_rt_trace_depth_entries(
   const std::vector<vtvk_rt_callback_shader_entry> &source,
   unsigned depth, void *mem_ctx,
   std::vector<vtvk_rt_callback_shader_entry> *clones,
   std::vector<nir_shader *> *entry_shaders,
   std::vector<const char *> *entry_symbols)
{
   clones->clear();
   for (size_t i = 0; i < source.size(); ++i) {
      nir_shader *clone = NULL;
      for (size_t j = 0; j < i; ++j) {
         if (source[j].shader == source[i].shader) {
            clone = (*clones)[j].shader;
            break;
         }
      }
      if (!clone)
         clone = nir_shader_clone(mem_ctx, source[i].shader);
      if (!clone)
         return false;
      clones->push_back({source[i].group_index, source[i].stage, clone,
                         nullptr, UINT32_MAX, nullptr,
                         source[i].call_data_read_word_mask,
                         source[i].call_data_write_word_mask,
                         source[i].call_data_summary_complete});
   }

   const char *suffix = ralloc_asprintf(mem_ctx, "_trace_depth%u", depth);
   return suffix && vtvk_prepare_rt_independent_entries(
                       *clones, mem_ctx, entry_shaders, entry_symbols, suffix);
}

static nir_def *
vtvk_rt_global_load_u32(nir_builder *b, nir_def *base, uint32_t offset)
{
   return nir_load_global(b, 1, 32, nir_iadd_imm(b, base, offset),
                          .align_mul = 4);
}

static void
vtvk_rt_global_store_u32(nir_builder *b, nir_def *base, uint32_t offset,
                         nir_def *value)
{
   nir_store_global(b, nir_u2u32(b, value), nir_iadd_imm(b, base, offset),
                    .write_mask = 0x1, .align_mul = 4);
}

static nir_def *
vtvk_rt_load_call_data_word(nir_builder *b, uint32_t byte_offset)
{
   return nir_ventus_rt_load_pds_field(
      b, .base = VT_NIR_RT_PDS_FIELD_CALL_DATA,
      .range_base = byte_offset);
}

static void
vtvk_rt_store_call_data_word(nir_builder *b, uint32_t byte_offset,
                             nir_def *value)
{
   nir_ventus_rt_store_pds_field(
      b, nir_u2u32(b, value), .base = VT_NIR_RT_PDS_FIELD_CALL_DATA,
      .range_base = byte_offset);
}

static void
vtvk_rt_save_callable_words(nir_builder *b, uint32_t call_data_offset,
                            nir_def *frame, uint32_t frame_offset,
                            uint32_t word_mask)
{
   for (uint32_t word = 0; word < 4; ++word) {
      if ((word_mask & BITFIELD_BIT(word)) == 0)
         continue;
      const uint32_t offset = word * sizeof(uint32_t);
      vtvk_rt_global_store_u32(
         b, frame, frame_offset + offset,
         vtvk_rt_load_call_data_word(b, call_data_offset + offset));
   }
}

static void
vtvk_rt_restore_callable_words(nir_builder *b, nir_def *frame,
                               uint32_t frame_offset,
                               uint32_t call_data_offset,
                               uint32_t word_mask)
{
   for (uint32_t word = 0; word < 4; ++word) {
      if ((word_mask & BITFIELD_BIT(word)) == 0)
         continue;
      const uint32_t offset = word * sizeof(uint32_t);
      vtvk_rt_store_call_data_word(
         b, call_data_offset + offset,
         vtvk_rt_global_load_u32(b, frame, frame_offset + offset));
   }
}

struct vtvk_rt_callable_entry_lower_data {
   const std::vector<vtvk_rt_callback_shader_entry> *callables;
   std::vector<vtvk_rt_callable_payload_binding> payload_bindings;
   uint32_t max_frame_bytes;
   bool failed;
};

static uint32_t
vtvk_rt_callable_frame_bytes(uint32_t word_mask)
{
   uint32_t frame_end = VT_NIR_RT_FRAME_BODY_OFFSET;
   for (uint32_t word = 0; word < 4; ++word) {
      if (word_mask & BITFIELD_BIT(word))
         frame_end = VT_NIR_RT_FRAME_BODY_OFFSET +
                     (word + 1) * sizeof(uint32_t);
   }
   return ALIGN_POT(MAX2(frame_end, VT_NIR_RT_MIN_FRAME_BYTES),
                    VT_NIR_RT_STACK_FRAME_ALIGNMENT_BYTES);
}

static bool
vtvk_lower_rt_execute_callable_entry(nir_builder *b,
                                     nir_intrinsic_instr *intrin, void *data)
{
   if (intrin->intrinsic != nir_intrinsic_execute_callable)
      return false;

   vtvk_rt_callable_entry_lower_data *state =
      static_cast<vtvk_rt_callable_entry_lower_data *>(data);
   if (!state || !state->callables || state->callables->empty() ||
       !nir_src_is_deref(intrin->src[1])) {
      if (state)
         state->failed = true;
      return false;
   }

   nir_deref_instr *payload_deref = nir_src_as_deref(intrin->src[1]);
   if (payload_deref->deref_type != nir_deref_type_var) {
      state->failed = true;
      return false;
   }
   nir_variable *payload = payload_deref->var;
   int payload_location = -1;
   if (!vtvk_rt_prepare_callable_payload(
          b->shader, payload, &state->payload_bindings,
          &payload_location) ||
       !glsl_type_is_vector_or_scalar(payload->type) ||
       glsl_get_bit_size(payload->type) != 32) {
      state->failed = true;
      return false;
   }

   const unsigned components = glsl_get_vector_elements(payload->type);
   const uint32_t payload_offset =
      (uint32_t)payload_location * VT_RT_ABI_ALIGNMENT_BYTES;
   if (components == 0 || components > 4 ||
       payload_offset + components * sizeof(uint32_t) >
          VENTUS_RT_V5_CALL_DATA_CAPACITY_BYTES) {
      state->failed = true;
      return false;
   }

   uint32_t target_read_mask = 0;
   uint32_t target_write_mask = 0;
   bool target_summary_complete = true;
   for (const vtvk_rt_callback_shader_entry &entry : *state->callables) {
      target_read_mask |= entry.call_data_read_word_mask;
      target_write_mask |= entry.call_data_write_word_mask;
      target_summary_complete &= entry.call_data_summary_complete;
   }
   const uint32_t component_mask = BITFIELD_MASK(components);
   const uint32_t first_word = payload_offset / sizeof(uint32_t);
   uint32_t local_read_mask =
      (target_read_mask >> first_word) & component_mask;
   uint32_t local_write_mask =
      (target_write_mask >> first_word) & component_mask;
   if (!target_summary_complete)
      local_read_mask = local_write_mask = component_mask;
   const uint32_t local_access_mask = local_read_mask | local_write_mask;
   const uint32_t call_site_frame_bytes =
      vtvk_rt_callable_frame_bytes(local_access_mask);
   state->max_frame_bytes = MAX2(state->max_frame_bytes,
                                 call_site_frame_bytes);
   if (vtvk_env_enabled("VENTUS_VK_NIR_PROBE")) {
      mesa_logi("ventus-vk: RT Callable continuation site stage=%s "
                "call-data=%u read=0x%x write=0x%x access=0x%x frame=%u",
                _mesa_shader_stage_to_string(b->shader->info.stage),
                payload_offset, local_read_mask, local_write_mask,
                local_access_mask, call_site_frame_bytes);
   }

   b->cursor = nir_before_instr(&intrin->instr);
   nir_def *base = vtvk_load_rt_dispatch_arg(
      b, VT_NIR_RT_DISPATCH_ARG_CALLABLE_SBT_ADDR, 64);
   nir_def *size = vtvk_load_rt_dispatch_arg(
      b, VT_NIR_RT_DISPATCH_ARG_CALLABLE_SBT_SIZE, 64);
   nir_def *stride = vtvk_load_rt_dispatch_arg(
      b, VT_NIR_RT_DISPATCH_ARG_CALLABLE_SBT_STRIDE, 64);
   nir_def *offset =
      nir_imul(b, nir_u2u64(b, intrin->src[0].ssa), stride);
   nir_def *record_end = nir_iadd_imm(b, offset, 32);
   nir_def *valid_record = nir_iand(
      b, nir_ine_imm(b, base, 0),
      nir_iand(b, nir_uge(b, stride, nir_imm_int64(b, 32)),
               nir_uge(b, size, record_end)));

   nir_push_if(b, valid_record);
   nir_def *record = nir_iadd(b, base, offset);
   nir_def *entry_ref =
      nir_load_global(b, 1, 32, nir_iadd_imm(b, record, 4), .align_mul = 4);
   nir_def *valid_entry = nir_imm_false(b);
   for (const vtvk_rt_callback_shader_entry &entry : *state->callables) {
      valid_entry = nir_ior(
         b, valid_entry, nir_ieq_imm(b, entry_ref, entry.group_index));
   }

   nir_push_if(b, valid_entry);
   struct vt_nir_rt_stack_cursor cursor;
   vt_nir_rt_v5_load_stack_cursor(b, &cursor);
   nir_def *stack_available = nir_ult(
      b, cursor.stack_pointer, cursor.frame_capacity);
   nir_def *callable_available = nir_ult(
      b, cursor.callable_depth, cursor.callable_capacity);
   nir_push_if(b, nir_iand(b, stack_available, callable_available));
   nir_def *frame = cursor.next_frame;
   vtvk_rt_save_callable_words(
      b, payload_offset, frame, VTVK_RT_CALLABLE_PARENT_DATA_OFFSET,
      local_access_mask);
   vt_nir_rt_v5_publish_stack_push(
      b, &cursor, nir_imm_int(b, VT_NIR_RT_FRAME_CALLABLE),
      nir_imm_int(b, 0), nir_iadd_imm(b, record, 32), true);

   nir_def *call_data = nir_load_var(b, payload);
   for (unsigned component = 0; component < components; ++component) {
      if ((local_access_mask & BITFIELD_BIT(component)) == 0)
         continue;
      vtvk_rt_store_call_data_word(
         b, payload_offset + component * sizeof(uint32_t),
         nir_channel(b, call_data, component));
   }
   for (const vtvk_rt_callback_shader_entry &entry : *state->callables) {
      nir_push_if(b, nir_ieq_imm(b, entry_ref, entry.group_index));
      nir_ventus_rt_call_shader_entry(
         b, nir_imm_int(b, entry.executable_index));
      nir_pop_if(b, NULL);
   }
   nir_def *returned_components[4];
   for (unsigned component = 0; component < components; ++component) {
      returned_components[component] =
         (local_write_mask & BITFIELD_BIT(component)) ?
            vtvk_rt_load_call_data_word(
               b, payload_offset + component * sizeof(uint32_t)) :
            nir_channel(b, call_data, component);
   }
   nir_def *returned_data = nir_vec(b, returned_components, components);
   vtvk_rt_restore_callable_words(
      b, frame, VTVK_RT_CALLABLE_PARENT_DATA_OFFSET, payload_offset,
      local_access_mask);
   vt_nir_rt_v5_publish_stack_pop(b, true);
   nir_store_var(b, payload, returned_data, BITFIELD_MASK(components));
   nir_push_else(b, NULL);
   nir_terminate(b);
   nir_pop_if(b, NULL);
   nir_push_else(b, NULL);
   nir_terminate(b);
   nir_pop_if(b, NULL);
   nir_push_else(b, NULL);
   nir_terminate(b);
   nir_pop_if(b, NULL);

   nir_instr_remove(&intrin->instr);
   return true;
}

static bool
vtvk_lower_rt_callable_entry_invocations(
   nir_shader **shaders, uint32_t shader_count,
   const std::vector<vtvk_rt_callback_shader_entry> &callables,
   bool *uses_callable_entry, uint32_t *max_frame_bytes)
{
   if (!shaders || !uses_callable_entry || !max_frame_bytes ||
       callables.empty())
      return true;

   for (uint32_t i = 0; i < shader_count; i++) {
      nir_shader *shader = shaders[i];
      if (!shader)
         continue;

      vtvk_rt_callable_entry_lower_data state = {
         .callables = &callables,
         .failed = false,
      };
      const bool progress = nir_shader_intrinsics_pass(
         shader, vtvk_lower_rt_execute_callable_entry, nir_metadata_none,
         &state);
      if (state.failed ||
          vtvk_rt_shader_has_intrinsic(shader,
                                       nir_intrinsic_execute_callable))
         return false;
      if (!progress)
         continue;

      *uses_callable_entry = true;
      *max_frame_bytes = MAX2(*max_frame_bytes, state.max_frame_bytes);
      nir_fixup_deref_modes(shader);
      nir_opt_dead_cf(shader);
      nir_opt_dce(shader);
      nir_remove_dead_derefs(shader);
      nir_function_impl *impl = nir_shader_get_entrypoint(shader);
      nir_index_blocks(impl);
      nir_index_ssa_defs(impl);
      nir_validate_shader(shader,
                          "after Ventus single-ELF callable entry lowering");
   }
   return true;
}

static bool
vtvk_lower_rt_callable_depth_overflow(nir_builder *b,
                                      nir_intrinsic_instr *intrin,
                                      void *data)
{
   (void)data;
   if (intrin->intrinsic != nir_intrinsic_execute_callable)
      return false;

   b->cursor = nir_before_instr(&intrin->instr);
   nir_terminate(b);
   nir_instr_remove(&intrin->instr);
   return true;
}

static bool
vtvk_lower_rt_callable_depth_overflows(nir_shader **shaders,
                                       uint32_t shader_count)
{
   for (uint32_t i = 0; i < shader_count; ++i) {
      nir_shader *shader = shaders[i];
      if (!shader)
         continue;
      NIR_PASS(_, shader, nir_shader_intrinsics_pass,
               vtvk_lower_rt_callable_depth_overflow, nir_metadata_none,
               NULL);
      if (vtvk_rt_shader_has_intrinsic(
             shader, nir_intrinsic_execute_callable))
         return false;
      nir_opt_dead_cf(shader);
      nir_opt_dce(shader);
   }
   return true;
}

static bool
vtvk_lower_rt_trace_depth_overflow(nir_builder *b,
                                   nir_intrinsic_instr *intrin, void *data)
{
   (void)data;
   if (intrin->intrinsic != nir_intrinsic_trace_ray)
      return false;

   b->cursor = nir_before_instr(&intrin->instr);
   nir_terminate(b);
   nir_instr_remove(&intrin->instr);
   return true;
}

static bool
vtvk_lower_rt_trace_depth_overflows(
   std::vector<vtvk_rt_callback_shader_entry> &entries)
{
   std::vector<nir_shader *> shaders;
   for (const vtvk_rt_callback_shader_entry &entry : entries) {
      if (std::find(shaders.begin(), shaders.end(), entry.shader) ==
          shaders.end())
         shaders.push_back(entry.shader);
   }
   for (nir_shader *shader : shaders) {
      NIR_PASS(_, shader, nir_shader_intrinsics_pass,
               vtvk_lower_rt_trace_depth_overflow, nir_metadata_none, NULL);
      if (vtvk_rt_shader_has_intrinsic(shader, nir_intrinsic_trace_ray))
         return false;
      nir_opt_dead_cf(shader);
      nir_opt_dce(shader);
   }
   return true;
}

static bool
vtvk_inline_rt_callable_shader(nir_builder *b, nir_shader *shader,
                               nir_variable *payload, int payload_location,
                               nir_variable *callable_shader_record_ptr,
                               nir_variable *selected_record_ptr)
{
   if (!b || !shader || !payload)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   if (!impl)
      return false;

   nir_opt_dead_cf(shader);
   nir_lower_returns(shader);
   nir_opt_dce(shader);

   struct hash_table *var_remap = _mesa_pointer_hash_table_create(b->shader);
   if (!var_remap)
      return false;

   unsigned incoming_payload_count = 0;
   bool valid = true;
   nir_foreach_variable_in_shader(var, shader) {
      if (var->data.mode == nir_var_function_temp)
         continue;

      if (var->data.mode == nir_var_shader_call_data) {
         incoming_payload_count++;
         if (!var->data.explicit_location ||
             var->data.location != payload_location ||
             !vtvk_rt_callable_types_match(payload, var)) {
            valid = false;
            break;
         }
         _mesa_hash_table_insert(var_remap, var, payload);
         continue;
      }

      if (var == callable_shader_record_ptr) {
         if (!selected_record_ptr) {
            valid = false;
            break;
         }
         _mesa_hash_table_insert(var_remap, var, selected_record_ptr);
         continue;
      }

      nir_variable *clone = nir_variable_clone(var, b->shader);
      if (!clone) {
         valid = false;
         break;
      }
      nir_shader_add_variable(b->shader, clone);
      _mesa_hash_table_insert(var_remap, var, clone);
   }

   if (incoming_payload_count > 1)
      valid = false;

   if (valid) {
      b->shader->scratch_size =
         MAX2(b->shader->scratch_size, shader->scratch_size);
      nir_inline_function_impl(b, impl, NULL, var_remap);
   }
   _mesa_hash_table_destroy(var_remap, NULL);
   return valid;
}

static bool
vtvk_lower_rt_execute_callable(nir_builder *b,
                               nir_intrinsic_instr *intrin, void *data)
{
   if (intrin->intrinsic != nir_intrinsic_execute_callable)
      return false;

   vtvk_rt_execute_callable_lower_data *state =
      static_cast<vtvk_rt_execute_callable_lower_data *>(data);
   if (!state || !state->callables || state->callables->empty() ||
       !nir_src_is_deref(intrin->src[1])) {
      if (state)
         state->failed = true;
      return false;
   }

   nir_deref_instr *payload_deref = nir_src_as_deref(intrin->src[1]);
   if (payload_deref->deref_type != nir_deref_type_var) {
      state->failed = true;
      return false;
   }

   nir_variable *payload = payload_deref->var;
   int payload_location = -1;
   if (!vtvk_rt_prepare_callable_payload(
          b->shader, payload, &state->payload_bindings,
          &payload_location)) {
      state->failed = true;
      return false;
   }

   b->cursor = nir_before_instr(&intrin->instr);
   nir_def *base = vtvk_load_rt_dispatch_arg(
      b, VT_NIR_RT_DISPATCH_ARG_CALLABLE_SBT_ADDR, 64);
   nir_def *size = vtvk_load_rt_dispatch_arg(
      b, VT_NIR_RT_DISPATCH_ARG_CALLABLE_SBT_SIZE, 64);
   nir_def *stride = vtvk_load_rt_dispatch_arg(
      b, VT_NIR_RT_DISPATCH_ARG_CALLABLE_SBT_STRIDE, 64);
   nir_def *offset =
      nir_imul(b, nir_u2u64(b, intrin->src[0].ssa), stride);
   nir_def *record_end = nir_iadd_imm(b, offset, 32);
   nir_def *valid_record = nir_iand(
      b, nir_ine_imm(b, base, 0),
      nir_iand(b, nir_uge(b, stride, nir_imm_int64(b, 32)),
               nir_uge(b, size, record_end)));

   nir_push_if(b, valid_record);
   nir_def *record = nir_iadd(b, base, offset);
   nir_variable *selected_record_ptr =
      nir_variable_create(b->shader, nir_var_shader_temp, glsl_uint64_t_type(),
                          "ventus_selected_callable_record_ptr");
   if (!selected_record_ptr) {
      state->failed = true;
      nir_pop_if(b, NULL);
      return false;
   }
   nir_store_var(b, selected_record_ptr, nir_iadd_imm(b, record, 32), 0x1);
   nir_def *group_index =
      nir_load_global(b, 1, 32, nir_iadd_imm(b, record, 4));
   for (const vtvk_rt_callback_shader_entry &entry : *state->callables) {
      nir_push_if(b, nir_ieq_imm(b, group_index, entry.group_index));
      if (!vtvk_inline_rt_callable_shader(
             b, entry.shader, payload, payload_location,
             entry.callable_shader_record_ptr, selected_record_ptr))
         state->failed = true;
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);

   nir_instr_remove(&intrin->instr);
   return true;
}

static bool
vtvk_lower_rt_callable_invocations(
   nir_shader **shaders, uint32_t shader_count,
   std::vector<vtvk_rt_callback_shader_entry> &callables)
{
   for (size_t i = 0; i < callables.size(); i++) {
      vtvk_rt_callback_shader_entry &entry = callables[i];
      if (vtvk_rt_shader_has_intrinsic(
             entry.shader, nir_intrinsic_execute_callable)) {
         mesa_loge("ventus-vk: nested callable shader invocation is not "
                   "supported");
         return false;
      }
      if (vtvk_rt_shader_has_intrinsic(
             entry.shader, nir_intrinsic_trace_ray)) {
         mesa_loge("ventus-vk: callable-originated traceRayEXT is not "
                   "supported");
         return false;
      }

      entry.callable_shader_record_ptr = nullptr;
      for (size_t j = 0; j < i; j++) {
         if (callables[j].shader == entry.shader) {
            entry.callable_shader_record_ptr =
               callables[j].callable_shader_record_ptr;
            break;
         }
      }
      if (entry.callable_shader_record_ptr)
         continue;

      if (!vtvk_prepare_rt_callable_shader_record_ptr(
             entry.shader, &entry.callable_shader_record_ptr)) {
         mesa_loge("ventus-vk: failed to bind callable shader record data");
         return false;
      }
   }

   for (uint32_t i = 0; i < shader_count; i++) {
      nir_shader *shader = shaders[i];
      if (!shader || shader->info.stage == MESA_SHADER_CALLABLE)
         continue;

      vtvk_rt_execute_callable_lower_data state = {
         .callables = &callables,
         .failed = false,
      };
      const bool progress = nir_shader_intrinsics_pass(
         shader, vtvk_lower_rt_execute_callable, nir_metadata_none, &state);
      if (state.failed ||
          vtvk_rt_shader_has_intrinsic(
             shader, nir_intrinsic_execute_callable))
         return false;

      if (progress) {
         nir_fixup_deref_modes(shader);
         nir_opt_dead_cf(shader);
         nir_opt_dce(shader);
         nir_remove_dead_derefs(shader);
         nir_function_impl *impl = nir_shader_get_entrypoint(shader);
         nir_index_blocks(impl);
         nir_index_ssa_defs(impl);
         nir_validate_shader(shader,
                             "after Ventus executeCallableEXT lowering");
      }
   }

   return true;
}

/* A procedural report stays inside one compact intersection callback.  The
 * matching any-hit code is inlined at the report site, so locals and control
 * flow naturally continue to a later reportIntersectionEXT in the same
 * shader.  The committed-record validity remembers an earlier accepted report
 * while CALLBACK_DECISION carries the final shader-return effect. */
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
   nir_def *committed_status = vt_nir_load_hit_record_word(
      b, config, layout, VT_RT_HIT_RECORD_STATUS, 32);
   nir_def *committed_tmax = vt_nir_load_hit_record_word(
      b, config, layout, VT_RT_HIT_RECORD_HIT_T, 32);
   nir_def *ray_tmax = vt_nir_load_ray_state_word(
      b, config, layout, VT_RT_SLOT_TMAX);
   nir_def *committed_valid = nir_ieq_imm(
      b, committed_status, VT_RT_HIT_RECORD_STATUS_VALID);
   nir_def *current_tmax = nir_bcsel(
      b, committed_valid,
      committed_tmax, ray_tmax);
   nir_def *tmin = vt_nir_load_ray_state_word(
      b, config, layout, VT_RT_SLOT_TMIN);
   nir_def *terminated = nir_ieq_imm(
      b, vt_nir_load_control_word(
            b, config, layout, VT_RT_CONTROL_CALLBACK_DECISION, 32),
      VT_RT_CALLBACK_TERMINATE);
   nir_def *ray_ref = vt_nir_rt_global_callback_ray_ref(b);
   nir_def *candidate_attributes = vt_nir_rt_global_load_completion_addr(
      b, ray_ref, VT_RT_GLOBAL_COMPLETION_CANDIDATE_HIT_ATTRIBUTE_ADDR_LO);
   nir_def *committed_attributes = vt_nir_rt_global_load_completion_addr(
      b, ray_ref, VT_RT_GLOBAL_COMPLETION_HIT_ATTRIBUTE_ADDR_LO);
   nir_def *in_range = nir_iand(
      b, nir_flt(b, hit_t, current_tmax), nir_fge(b, hit_t, tmin));
   nir_def *eligible = nir_iand(b, in_range, nir_inot(b, terminated));

   /* Preserve the AABB metadata populated by RTcore, changing only the
    * report-specific fields.  Its geometry is non-opaque unless the ray or
    * the record says otherwise. */
   vt_nir_store_candidate_hit_record_word(
      b, config, layout, VT_RT_HIT_RECORD_STATUS,
      nir_bcsel(b, eligible,
                nir_imm_int(b, VT_RT_HIT_RECORD_STATUS_VALID),
                nir_imm_int(b, VT_RT_HIT_RECORD_STATUS_INVALID)));
   vt_nir_store_candidate_hit_record_word(
      b, config, layout, VT_RT_HIT_RECORD_HIT_T, hit_t);
   vt_nir_store_candidate_hit_record_word(
      b, config, layout, VT_RT_HIT_RECORD_HIT_KIND, hit_kind);
   vt_nir_store_candidate_hit_record_word(
      b, config, layout, VT_RT_HIT_RECORD_FRONT_FACE,
      nir_ieq_imm(b, hit_kind, 0xfe));
   vt_nir_store_control_word(
      b, config, layout, VT_RT_CONTROL_CALLBACK_DECISION,
      nir_bcsel(b, terminated, nir_imm_int(b, VT_RT_CALLBACK_TERMINATE),
                nir_bcsel(b, eligible, nir_imm_int(b, VT_RT_CALLBACK_ACCEPT),
                          nir_bcsel(b, committed_valid,
                                    nir_imm_int(b, VT_RT_CALLBACK_ACCEPT),
                                    nir_imm_int(b, VT_RT_CALLBACK_IGNORE)))));

   nir_def *flags = vt_nir_load_ray_state_word(
      b, config, layout, VT_RT_SLOT_FLAGS);
   nir_def *force_opaque = nir_ine_imm(b, nir_iand_imm(b, flags, 1), 0);
   nir_def *force_non_opaque =
      nir_ine_imm(b, nir_iand_imm(b, flags, 1u << 1), 0);
   nir_def *geometry_opaque = nir_ine_imm(
      b, vt_nir_load_candidate_hit_record_word(
            b, config, layout, VT_RT_HIT_RECORD_OPAQUE, 32), 0);
   nir_def *opaque = nir_ior(
      b, force_opaque, nir_iand(b, geometry_opaque, nir_inot(b, force_non_opaque)));

   if (state->any_hit) {
      nir_push_if(b, nir_iand(b, eligible, nir_inot(b, opaque)));
      vtvk_inline_rt_callback_shader(b, state->any_hit);
      nir_pop_if(b, NULL);
   }

   nir_def *decision = vt_nir_load_control_word(
      b, config, layout, VT_RT_CONTROL_CALLBACK_DECISION, 32);
   nir_def *accepted = nir_iand(
      b, eligible,
      nir_ior(b, nir_ieq_imm(b, decision, VT_RT_CALLBACK_ACCEPT),
              nir_ieq_imm(b, decision, VT_RT_CALLBACK_TERMINATE)));

   nir_push_if(b, accepted);
   for (uint32_t word = 0; word < VT_RT_HIT_RECORD_WORD_COUNT; word++) {
      const enum vt_rt_hit_record_word hit_word =
         static_cast<enum vt_rt_hit_record_word>(word);
      vt_nir_store_hit_record_word(
         b, config, layout, hit_word,
         vt_nir_load_candidate_hit_record_word(b, config, layout, hit_word, 32));
   }
   for (uint32_t word = 0; word < 2; word++) {
      nir_def *offset = nir_imm_intN_t(b, word * sizeof(uint32_t), 64);
      nir_store_global(b, nir_load_global(
                              b, 1, 32, nir_iadd(b, candidate_attributes, offset)),
                       nir_iadd(b, committed_attributes, offset), 0x1);
   }
   nir_pop_if(b, NULL);

   vt_nir_store_control_word(
      b, config, layout, VT_RT_CONTROL_CALLBACK_DECISION,
      nir_bcsel(b, nir_ieq_imm(b, decision, VT_RT_CALLBACK_TERMINATE),
                nir_imm_int(b, VT_RT_CALLBACK_TERMINATE),
                nir_bcsel(b, nir_ior(b, committed_valid, accepted),
                          nir_imm_int(b, VT_RT_CALLBACK_ACCEPT),
                          nir_imm_int(b, VT_RT_CALLBACK_IGNORE))));

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

static void
vtvk_emit_rt_callback_shader_with_index(
   nir_builder *b,
   const std::vector<vtvk_rt_callback_shader_entry> &entries,
   nir_def *index, nir_def *record_ptr, bool independent_entry)
{
   if (!independent_entry) {
      vtvk_inline_rt_callback_shader_with_index(b, entries, index);
      return;
   }

   nir_def *frame = vt_nir_rt_v5_current_continuation_frame(b);
   nir_def *previous_record_ptr = nir_load_global(
      b, 1, 64,
      nir_iadd_imm(b, frame, VTVK_RT_ENTRY_RECORD_PTR_OFFSET),
      .align_mul = 8);
   nir_store_global(
      b, record_ptr,
      nir_iadd_imm(b, frame, VTVK_RT_ENTRY_RECORD_PTR_OFFSET),
      .write_mask = 0x1, .align_mul = 8);
   for (const vtvk_rt_callback_shader_entry &entry : entries) {
      nir_push_if(b, nir_ieq_imm(b, index, entry.group_index));
      nir_ventus_rt_call_shader_entry(
         b, nir_imm_int(b, entry.executable_index));
      nir_pop_if(b, NULL);
   }
   nir_store_global(
      b, previous_record_ptr,
      nir_iadd_imm(b, frame, VTVK_RT_ENTRY_RECORD_PTR_OFFSET),
      .write_mask = 0x1, .align_mul = 8);
}

struct vtvk_rt_v5_intersection_report_lower_data {
   const struct vt_nir_rt_config *config;
   const std::vector<vtvk_rt_callback_shader_entry> *any_hit;
   uint32_t intersection_group_index;
   bool statically_bound_any_hit;
};

/* An independent intersection entry must preserve the Vulkan-visible order
 * of reportIntersection calls.  Every accepted report invokes its matching
 * any-hit entry synchronously and updates S6 before the intersection shader
 * continues, so a later report observes the new nearest t. */
static bool
vtvk_lower_rt_v5_intersection_report_entry(nir_builder *b,
                                            nir_intrinsic_instr *intrin,
                                            void *data)
{
   if (intrin->intrinsic != nir_intrinsic_report_ray_intersection)
      return false;

   const vtvk_rt_v5_intersection_report_lower_data *state =
      static_cast<const vtvk_rt_v5_intersection_report_lower_data *>(data);
   const struct vt_nir_rt_config *config = state->config;
   const enum vt_rt_shader_layout layout = VT_RT_LAYOUT_INTERSECTION;
   b->cursor = nir_before_instr(&intrin->instr);

   nir_def *pds = nir_u2u64(
      b, nir_ventus_rt_hot_pds_address(b, nir_imm_int(b, 0)));
   nir_def *staging = nir_ventus_rt_load_pds_field(
      b, .base = VT_NIR_RT_PDS_FIELD_CALLBACK_STAGING, .range_base = 0);
   nir_def *already_terminated = nir_ine_imm(
      b, nir_iand_imm(b, staging,
                      VENTUS_RT_V5_SHADER_RETURN_TERMINATE), 0);
   nir_def *committed_valid = nir_ieq_imm(
      b, vt_nir_load_hit_record_word(
            b, config, layout, VT_RT_HIT_RECORD_STATUS, 32),
      VT_RT_HIT_RECORD_STATUS_VALID);
   nir_def *committed_t = vt_nir_load_hit_record_word(
      b, config, layout, VT_RT_HIT_RECORD_HIT_T, 32);
   nir_def *ray_tmax = vt_nir_load_ray_state_word(
      b, config, layout, VT_RT_SLOT_TMAX);
   nir_def *current_tmax = nir_bcsel(
      b, committed_valid, committed_t, ray_tmax);
   nir_def *tmin = vt_nir_load_ray_state_word(
      b, config, layout, VT_RT_SLOT_TMIN);
   nir_def *hit_t = intrin->src[0].ssa;
   nir_def *hit_kind = intrin->src[1].ssa;
   nir_def *in_range = nir_iand(
      b, nir_fisfinite(b, hit_t),
      nir_iand(b, nir_fge(b, hit_t, tmin),
               nir_fge(b, current_tmax, hit_t)));
   nir_def *eligible = nir_iand(b, in_range, nir_inot(b, already_terminated));

   vt_nir_store_candidate_hit_record_word(
      b, config, layout, VT_RT_HIT_RECORD_HIT_T, hit_t);
   vt_nir_store_candidate_hit_record_word(
      b, config, layout, VT_RT_HIT_RECORD_HIT_KIND, hit_kind);
   vt_nir_store_control_word(
      b, config, layout, VT_RT_CONTROL_CALLBACK_DECISION,
      nir_bcsel(b, eligible, nir_imm_int(b, VT_RT_CALLBACK_ACCEPT),
                nir_imm_int(b, VT_RT_CALLBACK_IGNORE)));

   nir_def *procedural_any_hit = vt_nir_load_candidate_hit_record_word(
      b, config, layout,
      VT_RT_HIT_RECORD_NEED_SOFTWARE_OPACITY_TEST, 1);
   nir_def *invoke_any_hit = state->statically_bound_any_hit ?
      eligible : nir_iand(b, eligible, procedural_any_hit);
   nir_push_if(b, invoke_any_hit);
   if (state->any_hit && !state->any_hit->empty()) {
      nir_def *frame = vt_nir_rt_v5_current_continuation_frame(b);
      nir_def *record_ptr = nir_load_global(
         b, 1, 64,
         nir_iadd_imm(b, frame, VTVK_RT_ENTRY_RECORD_PTR_OFFSET),
         .align_mul = 8);
      nir_def *group_index = state->statically_bound_any_hit ?
         nir_imm_int(b, state->intersection_group_index) :
         vtvk_load_rt_shader_record_group_index(b, record_ptr);
      vtvk_emit_rt_callback_shader_with_index(
         b, *state->any_hit, group_index, record_ptr, true);
   }
   nir_pop_if(b, NULL);

   nir_def *decision = vt_nir_load_control_word(
      b, config, layout, VT_RT_CONTROL_CALLBACK_DECISION, 32);
   nir_def *accepted = nir_iand(
      b, eligible,
      nir_ior(b, nir_ieq_imm(b, decision, VT_RT_CALLBACK_ACCEPT),
              nir_ieq_imm(b, decision, VT_RT_CALLBACK_TERMINATE)));
   nir_def *terminated =
      nir_ieq_imm(b, decision, VT_RT_CALLBACK_TERMINATE);
   nir_def *has_terminated = nir_ior(b, already_terminated, terminated);

   nir_push_if(b, accepted);
   vt_nir_rt_v5_publish_candidate_commit(
      b, config, pds, nir_imm_true(b));
   nir_pop_if(b, NULL);

   nir_def *previous_commit = nir_ine_imm(
      b, nir_iand_imm(b, staging,
                      VENTUS_RT_V5_COMPILER_CALLBACK_COMMITTED), 0);
   nir_def *has_commit = nir_ior(b, previous_commit, accepted);
   nir_ventus_rt_store_pds_field(
      b,
      nir_ior(
         b,
         nir_bcsel(
            b, has_commit,
            nir_imm_int(b, VENTUS_RT_V5_COMPILER_CALLBACK_COMMITTED),
            nir_imm_int(b, 0)),
            nir_bcsel(
               b, has_terminated,
               nir_imm_int(b, VENTUS_RT_V5_SHADER_RETURN_TERMINATE),
               nir_imm_int(b, 0))),
      .base = VT_NIR_RT_PDS_FIELD_CALLBACK_STAGING, .range_base = 0);

   nir_def_rewrite_uses(&intrin->def, accepted);
   nir_instr_remove(&intrin->instr);
   return true;
}

static bool
vtvk_lower_rt_v5_intersection_entry_reports(
   std::vector<vtvk_rt_callback_shader_entry> &intersection,
   const std::vector<vtvk_rt_callback_shader_entry> &any_hit,
   const struct vt_nir_rt_config *config)
{
   for (size_t i = 0; i < intersection.size(); i++) {
      nir_shader *shader = intersection[i].shader;
      bool already_lowered = false;
      for (size_t j = 0; j < i; j++) {
         if (intersection[j].shader == shader) {
            already_lowered = true;
            break;
         }
      }
      if (already_lowered)
         continue;

      size_t shader_group_count = 0;
      for (const vtvk_rt_callback_shader_entry &entry : intersection) {
         if (entry.shader == shader)
            shader_group_count++;
      }

      bool statically_bound_any_hit = false;
      if (shader_group_count == 1) {
         for (const vtvk_rt_callback_shader_entry &entry : any_hit) {
            if (entry.group_index == intersection[i].group_index) {
               statically_bound_any_hit = true;
               break;
            }
         }
      }

      const vtvk_rt_v5_intersection_report_lower_data data = {
         .config = config,
         .any_hit = &any_hit,
         .intersection_group_index = intersection[i].group_index,
         .statically_bound_any_hit = statically_bound_any_hit,
      };
      nir_shader_intrinsics_pass(
         shader, vtvk_lower_rt_v5_intersection_report_entry,
         nir_metadata_control_flow, (void *)&data);
      if (vtvk_rt_shader_has_intrinsic(
             shader, nir_intrinsic_report_ray_intersection))
         return false;
   }
   return true;
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
   vtvk_emit_rt_callback_shader_with_index(
      b, cb->shaders->miss, index, record_ptr,
      cb->independent_entries);
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
       !result || !result->hit_shader_record_ptr)
      return;

   nir_def *index =
      vtvk_load_rt_shader_record_group_index(b, result->hit_shader_record_ptr);
   vtvk_emit_rt_callback_shader_with_index(
      b, cb->shaders->closest_hit, index, result->hit_shader_record_ptr,
      cb->independent_entries);
}

static void
vtvk_emit_rt_any_hit(nir_builder *b, const struct vt_nir_rt_trace_params *params,
                     const struct vt_nir_rt_result *result, void *data)
{
   (void)params;
   const struct vtvk_rt_pipeline_callback_data *cb =
      static_cast<const struct vtvk_rt_pipeline_callback_data *>(data);

   if (!cb || !cb->shaders || cb->shaders->any_hit.empty() ||
       !result || !result->hit_shader_record_ptr)
      return;

   nir_def *index =
      vtvk_load_rt_shader_record_group_index(b, result->hit_shader_record_ptr);
   vtvk_emit_rt_callback_shader_with_index(
      b, cb->shaders->any_hit, index, result->hit_shader_record_ptr,
      cb->independent_entries);
}

static void
vtvk_emit_rt_intersection(nir_builder *b, const struct vt_nir_rt_trace_params *params,
                          const struct vt_nir_rt_result *result, void *data)
{
   (void)params;
   const struct vtvk_rt_pipeline_callback_data *cb =
      static_cast<const struct vtvk_rt_pipeline_callback_data *>(data);

   if (!cb || !cb->shaders || cb->shaders->intersection.empty() ||
       !result || !result->hit_shader_record_ptr)
      return;

   nir_def *index =
      vtvk_load_rt_shader_record_group_index(b, result->hit_shader_record_ptr);
   vtvk_emit_rt_callback_shader_with_index(
      b, cb->shaders->intersection, index, result->hit_shader_record_ptr,
      cb->independent_entries);
}
static bool
vtvk_rt_stage_supported(mesa_shader_stage stage)
{
   return stage == MESA_SHADER_RAYGEN ||
          stage == MESA_SHADER_MISS ||
          stage == MESA_SHADER_CLOSEST_HIT ||
          stage == MESA_SHADER_ANY_HIT ||
          stage == MESA_SHADER_INTERSECTION ||
          stage == MESA_SHADER_CALLABLE;
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
   NIR_PASS(_, *nir_out, vt_nir_lower_rt_push_constants);
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

static bool
vtvk_rt_stack_size_is_dynamic(
   const VkRayTracingPipelineCreateInfoKHR *info)
{
   if (!info || !info->pDynamicState)
      return false;

   for (uint32_t i = 0; i < info->pDynamicState->dynamicStateCount; i++) {
      if (info->pDynamicState->pDynamicStates[i] ==
          VK_DYNAMIC_STATE_RAY_TRACING_PIPELINE_STACK_SIZE_KHR)
         return true;
   }
   return false;
}

static uint32_t
vtvk_rt_group_shader_index(
   const VkRayTracingShaderGroupCreateInfoKHR *group,
   VkShaderGroupShaderKHR group_shader)
{
   switch (group_shader) {
   case VK_SHADER_GROUP_SHADER_GENERAL_KHR:
      return group->generalShader;
   case VK_SHADER_GROUP_SHADER_CLOSEST_HIT_KHR:
      return group->closestHitShader;
   case VK_SHADER_GROUP_SHADER_ANY_HIT_KHR:
      return group->anyHitShader;
   case VK_SHADER_GROUP_SHADER_INTERSECTION_KHR:
      return group->intersectionShader;
   default:
      return VK_SHADER_UNUSED_KHR;
   }
}

static bool
vtvk_rt_finalize_stack_sizes(
   struct vtvk_pipeline *pipeline,
   const VkRayTracingPipelineCreateInfoKHR *info,
   const VkDeviceSize stage_stack_sizes[MESA_VULKAN_SHADER_STAGES])
{
   static const VkShaderGroupShaderKHR group_shaders[] = {
      VK_SHADER_GROUP_SHADER_GENERAL_KHR,
      VK_SHADER_GROUP_SHADER_CLOSEST_HIT_KHR,
      VK_SHADER_GROUP_SHADER_ANY_HIT_KHR,
      VK_SHADER_GROUP_SHADER_INTERSECTION_KHR,
   };

   for (uint32_t group_index = 0; group_index < info->groupCount;
        group_index++) {
      const VkRayTracingShaderGroupCreateInfoKHR *group =
         &info->pGroups[group_index];
      for (uint32_t slot = 0; slot < ARRAY_SIZE(group_shaders); slot++) {
         const uint32_t shader_index =
            vtvk_rt_group_shader_index(group, group_shaders[slot]);
         if (shader_index == VK_SHADER_UNUSED_KHR ||
             shader_index >= info->stageCount)
            continue;

         const mesa_shader_stage stage =
            vk_to_mesa_shader_stage(info->pStages[shader_index].stage);
         if (stage < MESA_VULKAN_SHADER_STAGES) {
            pipeline->rt_group_stack_sizes[
               group_index * ARRAY_SIZE(group_shaders) + slot] =
                  stage_stack_sizes[stage];
         }
      }
   }

   const uint64_t raygen = stage_stack_sizes[MESA_SHADER_RAYGEN];
   const uint64_t closest = stage_stack_sizes[MESA_SHADER_CLOSEST_HIT];
   const uint64_t miss = stage_stack_sizes[MESA_SHADER_MISS];
   const uint64_t intersection = stage_stack_sizes[MESA_SHADER_INTERSECTION];
   const uint64_t any_hit = stage_stack_sizes[MESA_SHADER_ANY_HIT];
   const uint64_t callable = stage_stack_sizes[MESA_SHADER_CALLABLE];
   const uint64_t depth = info->maxPipelineRayRecursionDepth;
   const uint64_t first_trace =
      MAX3(closest, miss, intersection + any_hit);
   const uint64_t deeper_trace = MAX2(closest, miss);
   const uint64_t stack_size =
      raygen + MIN2(1ull, depth) * first_trace +
      (depth > 1 ? depth - 1 : 0) * deeper_trace + 2 * callable;
   if (stack_size > UINT32_MAX)
      return false;

   /* A non-dynamic pipeline must publish a nonzero default.  The value one
    * mirrors Mesa common behavior for a pipeline whose shaders require no
    * application-visible stack bytes. */
   pipeline->rt_vulkan_default_stack_size =
      stack_size ? (uint32_t)stack_size : 1;
   pipeline->rt_stack_size_dynamic = vtvk_rt_stack_size_is_dynamic(info);
   return true;
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
   out->callable.clear();

   auto add_shader = [&](uint32_t shader_index, uint32_t group_index,
                         mesa_shader_stage required_stage,
                         std::vector<vtvk_rt_callback_shader_entry> *dst) {
      if (shader_index == VK_SHADER_UNUSED_KHR || shader_index >= stage_count ||
          shader_index >= info->stageCount)
         return;

      mesa_shader_stage shader_stage =
         vk_to_mesa_shader_stage(info->pStages[shader_index].stage);
      if (shader_stage == required_stage) {
         dst->push_back(
            {group_index, required_stage, shaders[shader_index], nullptr,
             UINT32_MAX, nullptr});
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
            else if (stage == MESA_SHADER_CALLABLE)
               add_shader(group->generalShader, group_index,
                          MESA_SHADER_CALLABLE, &out->callable);
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
   if (!pipeline->callback_kernel_paths || !pipeline->callback_kernel_entries ||
       !pipeline->callback_kernel_groups || !pipeline->callback_kernel_stages)
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
      if (!vt_nir_lower_rt_global_callback(shader, &config)) {
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
         module, pipeline->rt_uses_rv64,
         pipeline->retain_generated_artifacts, false, false, &path, &entry);
      delete module;
      if (!emitted)
         return false;
      pipeline->callback_kernel_paths[i] = path;
      pipeline->callback_kernel_entries[i] = entry;
      pipeline->callback_kernel_groups[i] = callbacks[i].group_index;
      pipeline->callback_kernel_stages[i] = callbacks[i].stage;
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
   VkResult profile_result = vtvk_rt_execution_profile_parse(
      info->pNext, VTVK_MAX_BOUND_DESCRIPTOR_SETS,
      VTVK_MAX_DESCRIPTOR_BINDINGS, &profile_config);
   if (profile_result != VK_SUCCESS) {
      mesa_loge("ventus-vk: invalid Ventus RT execution-profile contract");
      return vk_error(device, profile_result);
   }

   struct vtvk_pipeline *pipeline =
      (struct vtvk_pipeline *)vk_pipeline_zalloc(
         &device->vk, &vtvk_pipeline_ops, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
         pipeline_flags, pAllocator, sizeof(*pipeline));
   if (!pipeline)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   pipeline->nir_mem_ctx = ralloc_context(nullptr);
   pipeline->shader_count = info->stageCount;
   pipeline->max_ray_recursion_depth = info->maxPipelineRayRecursionDepth;
   pipeline->rt_execution_profile = profile_config.profile;
   pipeline->rt_stage_role = profile_config.stage_role;
   pipeline->rt_state_descriptor_set = profile_config.state_descriptor_set;
   pipeline->rt_state_descriptor_binding =
      profile_config.state_descriptor_binding;
   pipeline->rt_uses_rv64 = device->rt_uses_rv64;
   pipeline->retain_generated_artifacts =
      vtvk_env_enabled("VENTUS_VK_RETAIN_SHADER_ARTIFACTS");
   const bool legacy_v4_requested =
      vtvk_env_enabled("VENTUS_VK_RT_LEGACY_V4");
   const bool v5_megakernel =
      profile_config.profile == VK_VENTUS_RT_EXECUTION_PROFILE_MEGAKERNEL &&
      !legacy_v4_requested;
   const bool single_elf_multi_entry = v5_megakernel;
   pipeline->rt_v5_enabled = v5_megakernel;
   if (legacy_v4_requested &&
       profile_config.profile != VK_VENTUS_RT_EXECUTION_PROFILE_MEGAKERNEL) {
      mesa_loge("ventus-vk: the legacy V4 compatibility path only applies "
                "to the MegaKernel execution profile");
      vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
   }
   pipeline->rt_single_elf_multi_entry = single_elf_multi_entry;
   pipeline->shaders =
      (nir_shader **)vk_zalloc2(&device->vk.alloc, pAllocator,
                                sizeof(nir_shader *) * info->stageCount, 8,
                                VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   pipeline->group_handles =
      (uint8_t *)vk_zalloc2(&device->vk.alloc, pAllocator,
                            32 * info->groupCount, 8,
                            VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   pipeline->rt_group_stack_sizes =
      (VkDeviceSize *)vk_zalloc2(
         &device->vk.alloc, pAllocator,
         sizeof(VkDeviceSize) * 4 * info->groupCount, 8,
         VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);

   if (!pipeline->nir_mem_ctx || !pipeline->shaders ||
       !pipeline->group_handles || !pipeline->rt_group_stack_sizes) {
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
   vtvk_rt_pipeline_callback_data cb_data = {
      .shaders = &cb_shaders,
      .independent_entries = single_elf_multi_entry,
   };
   std::vector<nir_shader *> linked_entry_shaders;
   std::vector<const char *> linked_entry_symbols;
   std::vector<vtvk_rt_callback_shader_entry> linked_entries;
   std::vector<std::vector<vtvk_rt_callback_shader_entry>>
      callable_depth_layers;
   std::vector<nir_shader *> callable_depth_shaders;
   std::vector<vtvk_rt_callback_shaders> trace_depth_layers;
   std::vector<vtvk_rt_pipeline_callback_data> trace_depth_callback_data;
   std::vector<nir_shader *> trace_depth_shaders;
   bool uses_callable_entry = false;
   bool uses_candidate_entry = false;
   std::vector<vtvk_rt_resident_trace_shader> resident_trace_shaders;
   std::vector<struct vt_nir_rt_resident_continuation_info>
      resident_continuations;
   uint32_t callable_frame_bytes = 0;
   VkDeviceSize stage_stack_sizes[MESA_VULKAN_SHADER_STAGES] = {};
   bool stage_uses_callable[MESA_VULKAN_SHADER_STAGES] = {};

   for (uint32_t s = 0; s < pipeline->shader_count; s++) {
      nir_shader *shader = pipeline->shaders[s];
      if (shader && shader->info.stage < MESA_VULKAN_SHADER_STAGES &&
          vtvk_rt_shader_has_intrinsic(
             shader, nir_intrinsic_execute_callable)) {
         stage_uses_callable[shader->info.stage] = true;
      }
   }

   vtvk_rt_infer_missing_payload_locations(pipeline->shaders,
                                           pipeline->shader_count);
   if (!vtvk_rt_compute_entry_call_data_summaries(&cb_shaders)) {
      mesa_loge("ventus-vk: RT entry call-data summary is not statically "
                "resolvable");
      vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
   }

   uint32_t ray_payload_size = 0;
   uint32_t callable_data_size = 0;
   if (!vt_nir_rt_compute_pipeline_call_data_sizes(
          pipeline->shaders, pipeline->shader_count, &ray_payload_size,
          &callable_data_size)) {
      mesa_loge("ventus-vk: RT pipeline call-data layout is not statically "
                "resolvable");
      vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
   }
   pipeline->rt_payload_size_bytes = ray_payload_size;
   pipeline->rt_callable_data_size_bytes = callable_data_size;

   if (single_elf_multi_entry) {
      if (!vtvk_prepare_rt_callable_entries(
             cb_shaders.callable, pipeline->nir_mem_ctx,
             &linked_entry_shaders, &linked_entry_symbols)) {
         mesa_loge("ventus-vk: independent callable-entry lowering failed");
         vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
         return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
      }

      for (unsigned depth = 2;
           depth <= VT_NIR_RT_BASE_CALLABLE_DEPTH; ++depth) {
         callable_depth_layers.emplace_back();
         if (!vtvk_clone_rt_callable_depth_entries(
                cb_shaders.callable, depth, pipeline->nir_mem_ctx,
                &callable_depth_layers.back(), &linked_entry_shaders,
                &linked_entry_symbols)) {
            mesa_loge("ventus-vk: failed to prepare Callable depth %u", depth);
            vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
            return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
         }
      }

      auto collect_unique_shaders = [](const auto &entries,
                                       std::vector<nir_shader *> *out) {
         for (const vtvk_rt_callback_shader_entry &entry : entries) {
            if (std::find(out->begin(), out->end(), entry.shader) == out->end())
               out->push_back(entry.shader);
         }
      };
      std::vector<nir_shader *> root_callable_callers;
      for (uint32_t s = 0; s < pipeline->shader_count; ++s) {
         nir_shader *shader = pipeline->shaders[s];
         if (shader && shader->info.stage != MESA_SHADER_CALLABLE)
            root_callable_callers.push_back(shader);
      }
      if (!vtvk_lower_rt_callable_entry_invocations(
             root_callable_callers.data(), root_callable_callers.size(),
             cb_shaders.callable, &uses_callable_entry,
             &callable_frame_bytes)) {
         mesa_loge("ventus-vk: root Callable invocation lowering failed");
         vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
         return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
      }

      std::vector<nir_shader *> depth_callers;
      collect_unique_shaders(cb_shaders.callable, &depth_callers);
      for (const auto &depth_entries : callable_depth_layers) {
         if (!vtvk_lower_rt_callable_entry_invocations(
                depth_callers.data(), depth_callers.size(), depth_entries,
                &uses_callable_entry, &callable_frame_bytes)) {
            mesa_loge("ventus-vk: nested Callable invocation lowering failed");
            vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
            return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
         }
         depth_callers.clear();
         collect_unique_shaders(depth_entries, &depth_callers);
         collect_unique_shaders(depth_entries, &callable_depth_shaders);
      }
      if (!vtvk_lower_rt_callable_depth_overflows(
             depth_callers.data(), depth_callers.size())) {
         mesa_loge("ventus-vk: Callable depth overflow lowering failed");
         vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
         return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
      }

      if (!vtvk_prepare_rt_independent_entries(
             cb_shaders.miss, pipeline->nir_mem_ctx,
             &linked_entry_shaders, &linked_entry_symbols, NULL) ||
          !vtvk_prepare_rt_independent_entries(
             cb_shaders.closest_hit, pipeline->nir_mem_ctx,
             &linked_entry_shaders, &linked_entry_symbols, NULL) ||
          !vtvk_prepare_rt_independent_entries(
             cb_shaders.any_hit, pipeline->nir_mem_ctx,
             &linked_entry_shaders, &linked_entry_symbols, NULL) ||
          !vtvk_prepare_rt_independent_entries(
             cb_shaders.intersection, pipeline->nir_mem_ctx,
             &linked_entry_shaders, &linked_entry_symbols, NULL)) {
         mesa_loge("ventus-vk: independent RT-entry preparation failed");
         vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
         return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
      }

      if (pipeline->max_ray_recursion_depth > 1) {
         trace_depth_layers.reserve(pipeline->max_ray_recursion_depth - 1);
         for (unsigned depth = 2;
              depth <= pipeline->max_ray_recursion_depth; ++depth) {
            vtvk_rt_callback_shaders layer = cb_shaders;
            layer.raygen = NULL;
            layer.miss.clear();
            layer.closest_hit.clear();
            if (!vtvk_clone_rt_trace_depth_entries(
                   cb_shaders.miss, depth, pipeline->nir_mem_ctx,
                   &layer.miss, &linked_entry_shaders,
                   &linked_entry_symbols) ||
                !vtvk_clone_rt_trace_depth_entries(
                   cb_shaders.closest_hit, depth, pipeline->nir_mem_ctx,
                   &layer.closest_hit, &linked_entry_shaders,
                   &linked_entry_symbols)) {
               mesa_loge("ventus-vk: failed to prepare trace depth %u", depth);
               vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
               return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
            }
            collect_unique_shaders(layer.miss, &trace_depth_shaders);
            collect_unique_shaders(layer.closest_hit, &trace_depth_shaders);
            trace_depth_layers.push_back(std::move(layer));
         }
      }
      trace_depth_callback_data.reserve(trace_depth_layers.size());
      for (const vtvk_rt_callback_shaders &layer : trace_depth_layers)
         trace_depth_callback_data.push_back({&layer, true});

      uses_candidate_entry = !cb_shaders.any_hit.empty() ||
                             !cb_shaders.intersection.empty();
      linked_entries.insert(linked_entries.end(), cb_shaders.callable.begin(),
                            cb_shaders.callable.end());
      linked_entries.insert(linked_entries.end(), cb_shaders.miss.begin(),
                            cb_shaders.miss.end());
      linked_entries.insert(linked_entries.end(),
                            cb_shaders.closest_hit.begin(),
                            cb_shaders.closest_hit.end());
      linked_entries.insert(linked_entries.end(), cb_shaders.any_hit.begin(),
                            cb_shaders.any_hit.end());
      linked_entries.insert(linked_entries.end(),
                            cb_shaders.intersection.begin(),
                            cb_shaders.intersection.end());
   } else if (!vtvk_lower_rt_callable_invocations(
                 pipeline->shaders, pipeline->shader_count,
                 cb_shaders.callable)) {
      mesa_loge("ventus-vk: executeCallableEXT lowering failed");
      vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
   }

   struct vt_nir_rt_abi_layout rt_abi = vt_nir_rt_default_abi_layout();
   bool triangle_barycentrics = false;
   uint32_t hit_attribute_rest = 0;
   const uint32_t rt_payload_size = MAX2(ray_payload_size,
                                         callable_data_size);
   if (!vtvk_rt_compute_hit_attribute_rest(
          pipeline->shaders, pipeline->shader_count, &triangle_barycentrics,
          &hit_attribute_rest) ||
       !vt_nir_rt_abi_layout_set_hit_attrib_size(
          &rt_abi, hit_attribute_rest, triangle_barycentrics) ||
       !vt_nir_rt_abi_layout_set_payload_size(&rt_abi, rt_payload_size) ||
       !vt_nir_rt_abi_layout_set_recursion_depth(
          &rt_abi, pipeline->max_ray_recursion_depth)) {
      mesa_loge("ventus-vk: RT frame layout exceeds MegaKernel ABI v4 limits");
      vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
   }

   if (v5_megakernel) {
      struct vt_nir_rt_config resident_config = vt_nir_rt_default_config();
      vt_nir_rt_config_apply_abi_layout(&resident_config, &rt_abi);
      resident_config.use_v5_pds = true;
      resident_config.enable_monolithic_shader_callbacks = true;
      resident_config.emit_miss =
         cb_shaders.miss.empty() ? NULL : vtvk_emit_rt_miss;
      resident_config.emit_closest_hit =
         cb_shaders.closest_hit.empty() ? NULL : vtvk_emit_rt_closest_hit;
      auto add_resident_trace_shader = [&resident_trace_shaders](
                                          nir_shader *shader,
                                          vtvk_rt_pipeline_callback_data
                                             *callbacks) {
         if (!shader || !vtvk_rt_shader_has_intrinsic(
                           shader, nir_intrinsic_trace_ray))
            return;
         for (const auto &existing : resident_trace_shaders) {
            if (existing.shader == shader)
               return;
         }
         resident_trace_shaders.push_back({shader, callbacks});
      };
      add_resident_trace_shader(cb_shaders.raygen, &cb_data);
      if (single_elf_multi_entry) {
         if (pipeline->max_ray_recursion_depth <= 1) {
            if (!vtvk_lower_rt_trace_depth_overflows(cb_shaders.miss) ||
                !vtvk_lower_rt_trace_depth_overflows(
                   cb_shaders.closest_hit)) {
               mesa_loge("ventus-vk: root trace-depth overflow lowering failed");
               vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
               return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
            }
         } else {
            vtvk_rt_pipeline_callback_data *next_callbacks =
               &trace_depth_callback_data.front();
            for (const auto &entry : cb_shaders.miss)
               add_resident_trace_shader(entry.shader, next_callbacks);
            for (const auto &entry : cb_shaders.closest_hit)
               add_resident_trace_shader(entry.shader, next_callbacks);
         }

         for (size_t layer_index = 0;
              layer_index < trace_depth_layers.size(); ++layer_index) {
            vtvk_rt_callback_shaders &layer =
               trace_depth_layers[layer_index];
            const bool final_depth =
               layer_index + 1 == trace_depth_layers.size();
            if (final_depth) {
               if (!vtvk_lower_rt_trace_depth_overflows(layer.miss) ||
                   !vtvk_lower_rt_trace_depth_overflows(layer.closest_hit)) {
                  mesa_loge("ventus-vk: final trace-depth overflow lowering "
                            "failed");
                  vtvk_pipeline_destroy(&device->vk, &pipeline->vk,
                                        pAllocator);
                  return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
               }
               continue;
            }
            vtvk_rt_pipeline_callback_data *next_callbacks =
               &trace_depth_callback_data[layer_index + 1];
            for (const auto &entry : layer.miss)
               add_resident_trace_shader(entry.shader, next_callbacks);
            for (const auto &entry : layer.closest_hit)
               add_resident_trace_shader(entry.shader, next_callbacks);
         }
      }

      for (const auto &resident_shader : resident_trace_shaders) {
         struct vt_nir_rt_parent_state_ssa_info parent_state = {};
         if (!vt_nir_opt_rt_parent_state_ssa(
                resident_shader.shader, &resident_config,
                &parent_state)) {
            mesa_loge("ventus-vk: V5 parent-state SSA preparation failed "
                      "for %s",
                      _mesa_shader_stage_to_string(
                         resident_shader.shader->info.stage));
            vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
            return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
         }
         if (parent_state.unresolved_access_count != 0) {
            mesa_loge("ventus-vk: V5 parent-state SSA rejected %u raw "
                      "carrier accesses (mask=0x%x) in %s",
                      parent_state.unresolved_access_count,
                      parent_state.unresolved_access_mask,
                      _mesa_shader_stage_to_string(
                         resident_shader.shader->info.stage));
            vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
            return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
         }
      }

      for (const auto &resident_shader : resident_trace_shaders) {
         resident_config.callback_data = resident_shader.callbacks;
         if (!vt_nir_prepare_rt_resident_continuations(
                resident_shader.shader, &resident_config)) {
            mesa_loge("ventus-vk: V5 resident trace continuation preparation "
                      "failed for %s",
                      _mesa_shader_stage_to_string(
                         resident_shader.shader->info.stage));
            vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
            return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
         }
      }
   }

   const bool global_wavefront =
      pipeline->rt_execution_profile ==
         VK_VENTUS_RT_EXECUTION_PROFILE_STAGED_DISPATCH;

   {
      struct vt_nir_rt_config config = vt_nir_rt_default_config();
      vt_nir_rt_config_apply_abi_layout(&config, &rt_abi);
      config.use_v5_pds = v5_megakernel;

      if (single_elf_multi_entry &&
          !vtvk_lower_rt_v5_intersection_entry_reports(
             cb_shaders.intersection, cb_shaders.any_hit, &config)) {
         mesa_loge("ventus-vk: independent intersection report lowering failed");
         vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
         return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
      }

      if (global_wavefront) {
         vt_nir_rt_config_apply_global_cps_layout(&config, &rt_abi);
         config.execution_model = VT_RT_EXECUTION_WAVEFRONT_GLOBAL_WORKER;
         config.defer_intersection_reports = true;
      }

      for (uint32_t s = 0; s < pipeline->shader_count; s++) {
         vtvk_rt_lower_abi_with_probe(pipeline->shaders[s], &config);
         nir_validate_shader(pipeline->shaders[s], "after Ventus RT lowering");
         if (vtvk_env_enabled("VENTUS_VK_DUMP_NIR"))
            nir_print_shader(pipeline->shaders[s], stderr);
      }
      for (nir_shader *shader : callable_depth_shaders) {
         vtvk_rt_lower_abi_with_probe(shader, &config);
         nir_validate_shader(shader,
                             "after Ventus Callable-depth RT lowering");
         if (vtvk_env_enabled("VENTUS_VK_DUMP_NIR"))
            nir_print_shader(shader, stderr);
      }
      for (nir_shader *shader : trace_depth_shaders) {
         vtvk_rt_lower_abi_with_probe(shader, &config);
         nir_validate_shader(shader,
                             "after Ventus trace-depth RT lowering");
         if (vtvk_env_enabled("VENTUS_VK_DUMP_NIR"))
            nir_print_shader(shader, stderr);
      }
   }

   if (v5_megakernel) {
      resident_continuations.resize(resident_trace_shaders.size());
      uint32_t frame_stride = callable_frame_bytes;
      for (size_t i = 0; i < resident_trace_shaders.size(); i++) {
         nir_shader *shader = resident_trace_shaders[i].shader;
         struct vt_nir_rt_resident_continuation_info &resident =
            resident_continuations[i];
         if (!vt_nir_lower_rt_resident_continuations(
                shader,
                pipeline->max_ray_recursion_depth, ray_payload_size,
                callable_data_size, true,
                &resident, pipeline->nir_mem_ctx)) {
            mesa_loge("ventus-vk: V5 resident trace continuation lowering "
                      "failed for %s",
                      _mesa_shader_stage_to_string(shader->info.stage));
            vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
            return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
         }
         pipeline->rt_v5_feature_flags |= resident.feature_flags;
         if (vtvk_env_enabled("VENTUS_VK_NIR_PROBE")) {
            for (uint32_t site_index = 0;
                 site_index < resident.site_count; ++site_index) {
               const struct vt_nir_rt_continuation_site &site =
                  resident.sites[site_index];
               mesa_logi("ventus-vk: RT trace continuation site stage=%s "
                         "call=%u control=%u live-ssa=%u "
                         "ssa-base=%u ssa-offset=%u "
                         "return=call-data+%u:%u frame=%u",
                         _mesa_shader_stage_to_string(shader->info.stage),
                         site.call_index, VT_NIR_RT_FRAME_BODY_OFFSET,
                         site.ssa_spill_bytes,
                         resident.scratch_base_bytes,
                         resident.scratch_offset_bytes,
                         site.return_data_offset, site.return_data_bytes,
                         site.frame_size_bytes);
            }
         }
         pipeline->rt_v5_max_trace_frame_bytes = MAX2(
            pipeline->rt_v5_max_trace_frame_bytes,
            resident.max_trace_frame_bytes);
         stage_stack_sizes[shader->info.stage] = MAX2(
            stage_stack_sizes[shader->info.stage],
            (VkDeviceSize)resident.max_trace_frame_bytes);
         frame_stride = MAX2(frame_stride,
                             resident.state_frame_stride_bytes);
      }

      if (uses_callable_entry) {
         pipeline->rt_v5_feature_flags |=
            VENTUS_RT_RESOURCE_USES_CALLABLE_CONTINUATION;
         pipeline->rt_v5_max_callable_frame_bytes =
            callable_frame_bytes;
      }
      frame_stride = ALIGN_POT(frame_stride, 32);
      pipeline->rt_v5_continuation_frame_stride = frame_stride;
      pipeline->rt_v5_callable_depth_capacity = uses_callable_entry ?
         VT_NIR_RT_BASE_CALLABLE_DEPTH : 0;
      /* One extra physical frame is reserved for a depth-faulting trace call.
       * Its generated helper must spill before the semantic depth check, but
       * it is never part of the valid active-continuation capacity. */
      if (!vt_nir_rt_v5_compute_continuation_stack_layout(
             frame_stride, pipeline->max_ray_recursion_depth,
             pipeline->rt_v5_callable_depth_capacity, true,
             &pipeline->rt_v5_continuation_frame_capacity,
             &pipeline->rt_v5_continuation_stack_bytes)) {
         mesa_loge("ventus-vk: V5 continuation stack resource overflow");
         vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
         return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
      }
      for (uint32_t stage = 0; stage < MESA_VULKAN_SHADER_STAGES; stage++) {
         if (stage_uses_callable[stage]) {
            stage_stack_sizes[stage] = MAX2(
               stage_stack_sizes[stage],
               (VkDeviceSize)callable_frame_bytes);
         }
      }
      if (!vtvk_rt_finalize_stack_sizes(pipeline, info, stage_stack_sizes)) {
         mesa_loge("ventus-vk: Vulkan RT pipeline stack size overflow");
         vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
         return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
      }
      if (vtvk_env_enabled("VENTUS_VK_NIR_PROBE")) {
         mesa_logi("ventus-vk: RT continuation layout max-trace=%u "
                   "max-callable=%u stride=%u capacity=%u stack=%u",
                   pipeline->rt_v5_max_trace_frame_bytes,
                   pipeline->rt_v5_max_callable_frame_bytes,
                   pipeline->rt_v5_continuation_frame_stride,
                   pipeline->rt_v5_continuation_frame_capacity,
                   pipeline->rt_v5_continuation_stack_bytes);
      }
      if (!vt_nir_rt_v5_initialize_continuation_stack(
             cb_shaders.raygen, frame_stride,
             pipeline->rt_v5_continuation_frame_capacity,
             pipeline->rt_v5_callable_depth_capacity)) {
         mesa_loge("ventus-vk: V5 continuation stack initialization failed");
         vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
         return vk_error(device, VK_ERROR_INITIALIZATION_FAILED);
      }
   }

   if (vtvk_env_enabled("VENTUS_VK_NIR_PROBE")) {
      mesa_logi("ventus-vk: RT ABI v4 hitattr=%u bary=%u payload-union=%u "
                "ray-payload=%u callable-data=%u depth=%u spill=%u "
                "private-base=%u V5-continuation=%u",
                rt_abi.hit_attrib_size_bytes,
                rt_abi.hit_attrib_barycentrics ? 1 : 0,
                rt_abi.payload_size_bytes, ray_payload_size,
                callable_data_size, rt_abi.max_ray_recursion_depth,
                rt_abi.payload_spill_size_bytes, rt_abi.private_base_bytes,
                pipeline->rt_v5_continuation_stack_bytes);
   }

   {
      struct vt_nir_rt_config rt_io_config = vt_nir_rt_default_config();
      vt_nir_rt_config_apply_abi_layout(&rt_io_config, &rt_abi);
      rt_io_config.use_v5_pds = v5_megakernel;
      if (global_wavefront) {
         vt_nir_rt_config_apply_global_cps_layout(&rt_io_config, &rt_abi);
         rt_io_config.execution_model = VT_RT_EXECUTION_WAVEFRONT_GLOBAL_WORKER;
      }

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
         if (!vt_nir_lower_rt_enqueue(cb_shaders.raygen)) {
            mesa_loge("ventus-vk: global wavefront producer did not publish "
                      "an enqueue operation");
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
            vt_nir_lower_rt_enqueue(pipeline->resume_shaders[i]);
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
         rt_io_config.use_v5_resident_continuations = v5_megakernel;
         rt_io_config.use_v5_independent_candidate_entries =
            single_elf_multi_entry && uses_candidate_entry;
         rt_io_config.v5_payload_size_bytes = ray_payload_size;
         rt_io_config.emit_miss = v5_megakernel || cb_shaders.miss.empty() ?
            NULL : vtvk_emit_rt_miss;
         rt_io_config.emit_closest_hit =
            v5_megakernel || cb_shaders.closest_hit.empty() ?
               NULL : vtvk_emit_rt_closest_hit;
         rt_io_config.emit_any_hit =
            cb_shaders.any_hit.empty() ? NULL : vtvk_emit_rt_any_hit;
         rt_io_config.emit_intersection =
            cb_shaders.intersection.empty() ? NULL : vtvk_emit_rt_intersection;
         rt_io_config.callback_data = &cb_data;

         if (v5_megakernel) {
            for (size_t i = 0; i < resident_trace_shaders.size(); i++) {
               nir_shader *shader = resident_trace_shaders[i].shader;
               struct vt_nir_rt_config shader_config = rt_io_config;
               shader_config.callback_data =
                  resident_trace_shaders[i].callbacks;
               shader_config.v5_continuation_scratch_bytes =
                  resident_continuations[i].scratch_bytes_per_thread;
               shader_config.v5_continuation_state_frame_stride_bytes =
                  pipeline->rt_v5_continuation_frame_stride;
               vtvk_rt_log_nir_stats(
                  shader, "entry before vt_nir_lower_rt_io_monolithic");
               NIR_PASS(_, shader, vt_nir_lower_rt_io_monolithic,
                        &shader_config);
               if (!vt_nir_materialize_rt_resident_continuations(
                      shader, &shader_config, &resident_continuations[i])) {
                  mesa_loge("ventus-vk: V5 resident continuation "
                            "materialization failed for %s",
                            _mesa_shader_stage_to_string(shader->info.stage));
                  vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
                  return vk_error(device, VK_ERROR_INITIALIZATION_FAILED);
               }
               vtvk_rt_log_nir_stats(
                  shader, "entry after vt_nir_lower_rt_io_monolithic");
            }
         } else {
            vtvk_rt_log_nir_stats(
               cb_shaders.raygen,
               "raygen before vt_nir_lower_rt_io_monolithic");
            NIR_PASS(_, cb_shaders.raygen, vt_nir_lower_rt_io_monolithic,
                     &rt_io_config);
            vtvk_rt_log_nir_stats(
               cb_shaders.raygen,
               "raygen after vt_nir_lower_rt_io_monolithic");
         }
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

   if (single_elf_multi_entry) {
      for (nir_shader *entry_shader : linked_entry_shaders) {
         NIR_PASS(_, entry_shader, vt_nir_lower_descriptors);
         struct vt_nir_rt_canonicalize_error entry_error = {};
         if (vt_nir_canonicalize_rt_for_llvm(entry_shader, &entry_error) !=
             VT_NIR_RT_CANONICALIZE_SUCCESS) {
            mesa_loge("ventus-vk: independent RT entry has non-SSA function "
                      "temporary %s",
                      entry_error.variable_name ? entry_error.variable_name :
                                                  "<unnamed>");
            vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
            return vk_error(device, VK_ERROR_INITIALIZATION_FAILED);
         }
         nir_validate_shader(entry_shader,
                             "after Ventus independent RT entry canonicalization");
      }
      /* Translate callees before raygen so direct entry calls resolve to an
       * existing function in the shared LLVM module. */
      linked_entry_shaders.push_back(cb_shaders.raygen);
      linked_entry_symbols.push_back("main");
   } else {
      /* Callback stages have been inlined into raygen.  Translating their
       * original NIR as well would compile dead entry points. */
      linked_entry_shaders.push_back(cb_shaders.raygen);
   }

   pipeline->per_lane_scratch_size =
      vt_nir_pipeline_compute_scratch(linked_entry_shaders.data(),
                                      linked_entry_shaders.size(),
                                      &pipeline->scratch_info);
   pipeline->per_lane_scratch_size = v5_megakernel ? 0 :
      MAX2(pipeline->per_lane_scratch_size, rt_abi.total_size_bytes);
   pipeline->hit_attribute_size_bytes = rt_abi.hit_attrib_size_bytes;

   struct vt_nir_to_llvm_options llvm_options = {
      .module_name = "ventus_rt_pipeline",
      .target_triple = pipeline->rt_uses_rv64 ? "riscv64-unknown-elf" :
                                                "riscv32-unknown-elf",
      .data_layout = pipeline->rt_uses_rv64 ?
         "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128-A5-G1" :
         "e-m:e-p:32:32-i64:64-n32-S128-A5-G1",
      .require_ssa_function_temps = true,
      .texture_explicit_quad = vtvk_env_enabled("VENTUS_VK_TEXTURE_EXPLICIT_QUAD"),
      .global_rt_wavefront = global_wavefront,
      .private_base_bytes = global_wavefront ? 0 :
         (v5_megakernel ? VENTUS_RT_V5_PDS_SIZE_BYTES +
                             pipeline->rt_v5_continuation_stack_bytes :
                          rt_abi.private_base_bytes),
      .emit_rt_resource_v4 = v5_megakernel,
      .rt_feature_flags = v5_megakernel ?
         pipeline->rt_v5_feature_flags : 0,
      .rt_pds_layout_id = v5_megakernel ? VENTUS_RT_PDS_LAYOUT_V5 : 0,
      .rt_private_layout_id = v5_megakernel ?
         VENTUS_RT_PRIVATE_LAYOUT_V5 : 0,
      .rt_hot_pds_offset_per_thread = 0,
      .rt_hot_pds_stride_bytes = v5_megakernel ?
         VENTUS_RT_V5_PDS_SIZE_BYTES : 0,
      .rt_continuation_offset_per_thread = v5_megakernel ?
         VENTUS_RT_V5_PDS_SIZE_BYTES : 0,
      .rt_continuation_bytes_per_thread = v5_megakernel ?
         pipeline->rt_v5_continuation_stack_bytes : 0,
      .rt_payload_size_bytes = v5_megakernel ? ray_payload_size : 0,
      .rt_callable_data_size_bytes = v5_megakernel ? callable_data_size : 0,
      .entry_point_names = single_elf_multi_entry ?
         linked_entry_symbols.data() : NULL,
      .entry_point_count = single_elf_multi_entry ?
         (unsigned)linked_entry_symbols.size() : 0,
   };

   /* A mip-linear sampler is immaterial when every texture instruction uses
    * LOD0 and minLod is zero. Preserve existing RT base-level workloads while
    * rejecting this sampler for explicit nonzero/dynamic LOD pipelines. */
   /* Separate global callbacks/resume ELFs are not all in this list. */
   pipeline->texture_lod0_only = !global_wavefront;
   for (nir_shader *shader : linked_entry_shaders) {
      nir_foreach_function_impl(impl, shader) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_tex)
                  continue;
               const nir_tex_instr *tex = nir_instr_as_tex(instr);
               bool lod0 = tex->op == nir_texop_tex;
               if (tex->op == nir_texop_txl) {
                  for (unsigned src = 0; src < tex->num_srcs; ++src)
                     if (tex->src[src].src_type == nir_tex_src_lod)
                        lod0 = nir_src_is_const(tex->src[src].src) &&
                           nir_src_as_float(tex->src[src].src) == 0.0f;
               }
               pipeline->texture_lod0_only &= lod0;
            }
         }
      }
   }
   pipeline->llvm_module =
      vt_nir_to_llvm_translate(linked_entry_shaders.data(),
                               linked_entry_shaders.size(),
                               &llvm_options);
   if (!pipeline->llvm_module) {
      vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
      return vk_error(device, VK_ERROR_INITIALIZATION_FAILED);
   }
   if (!vtvk_maybe_emit_rt_kernel(pipeline)) {
      vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
      return vk_error(device, VK_ERROR_INITIALIZATION_FAILED);
   }
   if (!vtvk_resolve_rt_shader_entries(pipeline, linked_entries)) {
      vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
      return vk_error(device, VK_ERROR_INITIALIZATION_FAILED);
   }
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
             "Ventus lowering -> LLVM; abi=%s stages=%u groups=%u scratch=%u "
             "max_depth=%u llvm_functions=%zu",
             v5_megakernel ? "V5" : "V4",
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
