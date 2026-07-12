#include "vtvk_pipeline.h"

#include <cerrno>
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
vtvk_rt_finalize_abi(nir_shader *shader,
                     const struct vt_nir_rt_config *config)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   if (!impl)
      return;

   enum vt_rt_shader_layout layout = vt_nir_layout_for_stage(shader->info.stage);
   shader->scratch_size =
      MAX2(shader->scratch_size, config->scratchpad[layout].total_size_bytes);

   nir_shader_gather_info(shader, impl);
}

static void
vtvk_rt_lower_abi_with_probe(nir_shader *shader,
                             const struct vt_nir_rt_config *config)
{
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
   if (!lib_dir || !lib_dir[0])
      lib_dir = "/home/guanys/ventus-env/llvm/build/lib";
   return "LD_LIBRARY_PATH=" + vtvk_shell_quote(lib_dir) +
          ":${LD_LIBRARY_PATH:-} ";
}

static bool
vtvk_dump_rt_llvm_module_to(struct vtvk_pipeline *pipeline,
                            const std::string &ll_path,
                            const char *label)
{
   std::string text;
   llvm::raw_string_ostream text_os(text);
   pipeline->llvm_module->print(text_os, nullptr);
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
vtvk_dump_rt_llvm_module(struct vtvk_pipeline *pipeline, std::string *ll_path)
{
   std::string base = vtvk_rt_artifact_base();
   *ll_path = base + ".ll";
   return vtvk_dump_rt_llvm_module_to(pipeline, *ll_path, "pre-lowering");
}

static bool
vtvk_find_crt0(std::string *path)
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

   static const char *candidates[] = {
      "/home/guanys/ventus-env/llvm/install/lib/crt0.o",
      "/home/guanys/ventus-env/llvm/build-libclc/lib/crt0.o",
      "/home/guanys/ventus-env/llvm/build-libclc/riscv32/lib/CMakeFiles/ctr0_obj.dir/crt0.S.o",
   };

   for (const char *candidate : candidates) {
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
vtvk_dump_rt_emit_llvm_module(struct vtvk_pipeline *pipeline,
                              const std::string &ll_path,
                              std::string *emit_ll_path)
{
   *emit_ll_path = ll_path.substr(0, ll_path.size() - 3) + ".emit.ll";
   return vtvk_dump_rt_llvm_module_to(pipeline, *emit_ll_path, "ELF-input");
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

static void
vtvk_maybe_emit_rt_kernel(struct vtvk_pipeline *pipeline)
{
   if (!vtvk_env_enabled("VENTUS_VK_DUMP_LLVM") &&
       !vtvk_env_enabled("VENTUS_VK_EMIT_ELF"))
      return;

   std::string ll_path;
   if (!vtvk_dump_rt_llvm_module(pipeline, &ll_path) ||
       !vtvk_env_enabled("VENTUS_VK_EMIT_ELF"))
      return;

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
      return;
   if (!vtvk_find_crt0(&crt0_path))
      return;

   if (!vtvk_dump_rt_emit_llvm_module(pipeline, ll_path, &emit_ll_path))
      return;

   std::string tool_env = vtvk_llvm_tool_env_prefix();
   std::string llc_cmd = tool_env + vtvk_shell_quote(llc) +
                         " -mtriple=riscv32-unknown-elf"
                         " -target-abi=ilp32"
                         " -mattr=+m,+a,+f,+d,+v,+zfh"
                         " -filetype=obj " +
                         vtvk_shell_quote(emit_ll_path.c_str()) + " -o " +
                         vtvk_shell_quote(obj_path.c_str());
   if (!vtvk_run_tool(llc_cmd, "llc"))
      return;

   std::string link_cmd;
   if (!vtvk_rt_elf_link_command(tool_env, ld_lld, ld_path, elf_path, crt0_path,
                                 obj_path, &link_cmd))
      return;
   if (!vtvk_run_tool(link_cmd, "link"))
      return;

   uint64_t kernel_entry = 0;
   if (!vtvk_read_elf_symbol(elf_path, "main", &kernel_entry))
      return;

   free(pipeline->generated_kernel_path);
   pipeline->generated_kernel_path = strdup(elf_path.c_str());
   pipeline->generated_kernel_entry = kernel_entry;
   mesa_logi("ventus-vk: emitted RT RISC-V kernel to %s entry=0x%" PRIx64,
             elf_path.c_str(), kernel_entry);
}

struct vtvk_rt_callback_shader_entry {
   uint32_t group_index;
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
   return var && var->data.explicit_location &&
          (var->data.mode == nir_var_shader_temp ||
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
   std::vector<vtvk_rt_payload_location_candidate> candidates;
   for (uint32_t i = 0; i < shader_count; i++)
      vtvk_rt_collect_payload_location_candidates(shaders[i], &candidates);

   if (candidates.empty())
      return;

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
         if (vtvk_env_enabled("VENTUS_VK_NIR_PROBE")) {
            mesa_logi("ventus-vk: inferred RT payload location %d for %s",
                      location, var->name ? var->name : "<unnamed>");
         }
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

   for (size_t i = 0; i < entries.size(); i++) {
      nir_push_if(b, nir_ieq_imm(b, index, entries[i].group_index));
      vtvk_inline_rt_callback_shader(b, entries[i].shader);
      if (i + 1 < entries.size())
         nir_push_else(b, NULL);
   }
   for (size_t i = 0; i < entries.size(); i++)
      nir_pop_if(b, NULL);
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
       !result || !result->sbt_index)
      return;

   nir_def *index =
      vtvk_load_rt_shader_record_group_index(b, result->hit_shader_record_ptr);
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
       !result || !result->sbt_index)
      return;

   nir_def *index =
      vtvk_load_rt_shader_record_group_index(b, result->hit_shader_record_ptr);
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
       !result || !result->sbt_index)
      return;

   nir_def *index =
      vtvk_load_rt_shader_record_group_index(b, result->hit_shader_record_ptr);
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
         dst->push_back({group_index, shaders[shader_index]});
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

static VkResult
vtvk_create_one_rt_pipeline(VkDevice _device,
                            VkPipelineCreateFlags2KHR pipeline_flags,
                            const VkRayTracingPipelineCreateInfoKHR *info,
                            const VkAllocationCallbacks *pAllocator,
                            VkPipeline *out)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);

   struct vtvk_pipeline *pipeline =
      (struct vtvk_pipeline *)vk_pipeline_zalloc(
         &device->vk, &vtvk_pipeline_ops, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
         pipeline_flags, pAllocator, sizeof(*pipeline));
   if (!pipeline)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   pipeline->nir_mem_ctx = ralloc_context(nullptr);
   pipeline->shader_count = info->stageCount;

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

   {
      struct vt_nir_rt_config config = vt_nir_rt_default_config();
      struct vt_nir_rt_abi_layout abi = vt_nir_rt_default_abi_layout();
      vt_nir_rt_config_apply_abi_layout(&config, &abi);

      for (uint32_t s = 0; s < pipeline->shader_count; s++) {
         vtvk_rt_lower_abi_with_probe(pipeline->shaders[s], &config);
         nir_validate_shader(pipeline->shaders[s], "after Ventus RT lowering");
         if (vtvk_env_enabled("VENTUS_VK_DUMP_NIR"))
            nir_print_shader(pipeline->shaders[s], stderr);
      }
   }

   {
      struct vt_nir_rt_config rt_io_config = vt_nir_rt_default_config();
      struct vt_nir_rt_abi_layout abi = vt_nir_rt_default_abi_layout();
      vt_nir_rt_config_apply_abi_layout(&rt_io_config, &abi);
      rt_io_config.enable_monolithic_shader_callbacks = true;
      rt_io_config.emit_miss = cb_shaders.miss.empty() ? NULL : vtvk_emit_rt_miss;
      rt_io_config.emit_closest_hit =
         cb_shaders.closest_hit.empty() ? NULL : vtvk_emit_rt_closest_hit;
      rt_io_config.emit_any_hit =
         cb_shaders.any_hit.empty() ? NULL : vtvk_emit_rt_any_hit;
      rt_io_config.emit_intersection =
         cb_shaders.intersection.empty() ? NULL : vtvk_emit_rt_intersection;

      vtvk_rt_pipeline_callback_data cb_data = {
         .shaders = &cb_shaders,
      };
      rt_io_config.callback_data = &cb_data;

      vtvk_rt_log_nir_stats(cb_shaders.raygen,
                            "raygen before vt_nir_lower_rt_io_monolithic");
      NIR_PASS(_, cb_shaders.raygen, vt_nir_lower_rt_io_monolithic,
               &rt_io_config);
      vtvk_rt_log_nir_stats(cb_shaders.raygen,
                            "raygen after vt_nir_lower_rt_io_monolithic");
      NIR_PASS(_, cb_shaders.raygen, vt_nir_lower_descriptors);
      vtvk_rt_log_nir_stats(cb_shaders.raygen,
                            "raygen after monolithic vt_nir_lower_descriptors");
   }

   pipeline->per_lane_scratch_size =
      vt_nir_pipeline_compute_scratch(pipeline->shaders, pipeline->shader_count,
                                      &pipeline->scratch_info);

   struct vt_nir_to_llvm_options llvm_options = {
      .module_name = "ventus_rt_pipeline",
      .target_triple = "riscv32-unknown-elf",
      .data_layout = "e-m:e-p:32:32-i64:64-n32-S128",
   };

   pipeline->llvm_module =
      vt_nir_to_llvm_translate(pipeline->shaders, pipeline->shader_count,
                               &llvm_options);
   if (!pipeline->llvm_module) {
      vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
      return vk_error(device, VK_ERROR_INITIALIZATION_FAILED);
   }
   vtvk_maybe_emit_rt_kernel(pipeline);

   vtvk_fill_group_handles(pipeline, info);

   mesa_logi("ventus-vk: RT pipeline compiled via SPIR-V -> NIR -> "
             "Ventus lowering -> LLVM; stages=%u groups=%u scratch=%u llvm_functions=%zu",
             info->stageCount, info->groupCount,
             pipeline->per_lane_scratch_size,
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
