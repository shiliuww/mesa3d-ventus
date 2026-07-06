#include "vtvk_private.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "compiler/spirv/nir_spirv.h"
#include "nir/nir.h"
#include "util/hash_table.h"
#include "util/log.h"
#include "util/ralloc.h"
#include "vk_pipeline.h"
#include "vk_util.h"
#include "vtvk_driver_bridge.h"
#include "vtvk_entrypoints.h"
#include "ventus/common/ventus_rt_bvh.h"
#include "ventus/compiler/vt_nir.h"
#include "ventus/llvm/ventus_nir_to_llvm.h"

#include <cfloat>

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

struct vtvk_aabb {
   float min[3];
   float max[3];
};

struct vtvk_build_node {
   uint32_t ref;
   struct vtvk_aabb bounds;
};

struct vtvk_pipeline {
   struct vk_pipeline vk;

   void *nir_mem_ctx;
   nir_shader **shaders;
   uint32_t shader_count;

   uint8_t *group_handles;
   uint32_t group_count;

   uint32_t per_lane_scratch_size;
   struct vt_nir_pipeline_scratch_info scratch_info;

   llvm::Module *llvm_module;
   char *generated_kernel_path;
   uint64_t generated_kernel_entry;
};

static std::vector<struct vtvk_acceleration_structure *> vtvk_as_registry;

static void *
vtvk_handle_to_ptr(uint64_t handle)
{
   return (void *)(uintptr_t)handle;
}

static uint64_t
vtvk_ptr_to_handle(void *ptr)
{
   return (uint64_t)(uintptr_t)ptr;
}

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

static void
vtvk_aabb_empty(struct vtvk_aabb *aabb)
{
   for (unsigned i = 0; i < 3; i++) {
      aabb->min[i] = FLT_MAX;
      aabb->max[i] = -FLT_MAX;
   }
}

static void
vtvk_aabb_include_point(struct vtvk_aabb *aabb, const float p[3])
{
   for (unsigned i = 0; i < 3; i++) {
      aabb->min[i] = std::min(aabb->min[i], p[i]);
      aabb->max[i] = std::max(aabb->max[i], p[i]);
   }
}

static void
vtvk_aabb_include_aabb(struct vtvk_aabb *dst, const struct vtvk_aabb *src)
{
   vtvk_aabb_include_point(dst, src->min);
   vtvk_aabb_include_point(dst, src->max);
}

static void
vtvk_store_u16(uint8_t *base, uint32_t offset, uint16_t value)
{
   memcpy(base + offset, &value, sizeof(value));
}

static void
vtvk_store_u32(uint8_t *base, uint32_t offset, uint32_t value)
{
   memcpy(base + offset, &value, sizeof(value));
}

static void
vtvk_store_u64(uint8_t *base, uint32_t offset, uint64_t value)
{
   memcpy(base + offset, &value, sizeof(value));
}

static void
vtvk_store_f32(uint8_t *base, uint32_t offset, float value)
{
   memcpy(base + offset, &value, sizeof(value));
}

static uint32_t
vtvk_load_u32(const uint8_t *base, uint32_t offset)
{
   uint32_t value;
   memcpy(&value, base + offset, sizeof(value));
   return value;
}

static uint64_t
vtvk_load_u64(const uint8_t *base, uint32_t offset)
{
   uint64_t value;
   memcpy(&value, base + offset, sizeof(value));
   return value;
}

static float
vtvk_load_f32(const uint8_t *base, uint32_t offset)
{
   float value;
   memcpy(&value, base + offset, sizeof(value));
   return value;
}

static struct vtvk_aabb
vtvk_header_aabb(const uint8_t *as_data)
{
   struct vtvk_aabb aabb;
   aabb.min[0] = vtvk_load_f32(as_data, VENTUS_AS_HEADER_ROOT_AABB_MIN_X);
   aabb.min[1] = vtvk_load_f32(as_data, VENTUS_AS_HEADER_ROOT_AABB_MIN_Y);
   aabb.min[2] = vtvk_load_f32(as_data, VENTUS_AS_HEADER_ROOT_AABB_MIN_Z);
   aabb.max[0] = vtvk_load_f32(as_data, VENTUS_AS_HEADER_ROOT_AABB_MAX_X);
   aabb.max[1] = vtvk_load_f32(as_data, VENTUS_AS_HEADER_ROOT_AABB_MAX_Y);
   aabb.max[2] = vtvk_load_f32(as_data, VENTUS_AS_HEADER_ROOT_AABB_MAX_Z);
   return aabb;
}

static void
vtvk_write_header(uint8_t *dst, enum ventus_as_type type, uint32_t root_ref,
                  uint32_t node_count, uint32_t node_region_size,
                  uint32_t primitive_count, uint32_t instance_count,
                  const struct vtvk_aabb *root_aabb)
{
   memset(dst, 0, VENTUS_AS_HEADER_SIZE);
   vtvk_store_u32(dst, VENTUS_AS_HEADER_MAGIC, VENTUS_AS_MAGIC);
   vtvk_store_u16(dst, VENTUS_AS_HEADER_VERSION, VENTUS_AS_VERSION);
   vtvk_store_u16(dst, VENTUS_AS_HEADER_HEADER_SIZE, VENTUS_AS_HEADER_SIZE);
   vtvk_store_u32(dst, VENTUS_AS_HEADER_TYPE, type);
   vtvk_store_u32(dst, VENTUS_AS_HEADER_ROOT_NODE_REF, root_ref);
   vtvk_store_u32(dst, VENTUS_AS_HEADER_NODE_COUNT, node_count);
   vtvk_store_u32(dst, VENTUS_AS_HEADER_NODE_REGION_OFFSET,
                  VENTUS_AS_HEADER_SIZE);
   vtvk_store_u32(dst, VENTUS_AS_HEADER_NODE_REGION_SIZE, node_region_size);
   vtvk_store_u32(dst, VENTUS_AS_HEADER_PRIMITIVE_REGION_OFFSET,
                  VENTUS_AS_HEADER_SIZE);
   vtvk_store_u32(dst, VENTUS_AS_HEADER_PRIMITIVE_COUNT, primitive_count);
   vtvk_store_u32(dst, VENTUS_AS_HEADER_INSTANCE_REGION_OFFSET,
                  VENTUS_AS_HEADER_SIZE);
   vtvk_store_u32(dst, VENTUS_AS_HEADER_INSTANCE_COUNT, instance_count);
   vtvk_store_f32(dst, VENTUS_AS_HEADER_ROOT_AABB_MIN_X, root_aabb->min[0]);
   vtvk_store_f32(dst, VENTUS_AS_HEADER_ROOT_AABB_MIN_Y, root_aabb->min[1]);
   vtvk_store_f32(dst, VENTUS_AS_HEADER_ROOT_AABB_MIN_Z, root_aabb->min[2]);
   vtvk_store_f32(dst, VENTUS_AS_HEADER_ROOT_AABB_MAX_X, root_aabb->max[0]);
   vtvk_store_f32(dst, VENTUS_AS_HEADER_ROOT_AABB_MAX_Y, root_aabb->max[1]);
   vtvk_store_f32(dst, VENTUS_AS_HEADER_ROOT_AABB_MAX_Z, root_aabb->max[2]);
}

static void
vtvk_write_box4(uint8_t *dst, uint32_t offset,
                const std::vector<struct vtvk_build_node> &children)
{
   uint8_t *node = dst + offset;
   memset(node, 0, VENTUS_AS_BOX4_NODE_SIZE);
   for (uint32_t i = 0; i < 4; i++)
      vtvk_store_u32(node, VENTUS_BOX4_CHILD_REF + i * 4,
                     VENTUS_AS_INVALID_NODE);

   for (uint32_t i = 0; i < children.size(); i++) {
      vtvk_store_u32(node, VENTUS_BOX4_CHILD_REF + i * 4, children[i].ref);
      vtvk_store_f32(node, VENTUS_BOX4_MIN_X + i * 4, children[i].bounds.min[0]);
      vtvk_store_f32(node, VENTUS_BOX4_MIN_Y + i * 4, children[i].bounds.min[1]);
      vtvk_store_f32(node, VENTUS_BOX4_MIN_Z + i * 4, children[i].bounds.min[2]);
      vtvk_store_f32(node, VENTUS_BOX4_MAX_X + i * 4, children[i].bounds.max[0]);
      vtvk_store_f32(node, VENTUS_BOX4_MAX_Y + i * 4, children[i].bounds.max[1]);
      vtvk_store_f32(node, VENTUS_BOX4_MAX_Z + i * 4, children[i].bounds.max[2]);
   }
}

static uint32_t
vtvk_next_node_offset(uint32_t *cursor, uint32_t size)
{
   uint32_t offset = *cursor;
   *cursor += size;
   return offset;
}

static struct vtvk_build_node
vtvk_build_box_tree(uint8_t *dst, uint32_t *cursor,
                    std::vector<struct vtvk_build_node> leaves,
                    uint32_t *node_count)
{
   if (leaves.size() == 1)
      return leaves[0];

   while (leaves.size() > 1) {
      std::vector<struct vtvk_build_node> next;
      for (size_t i = 0; i < leaves.size(); i += 4) {
         std::vector<struct vtvk_build_node> children;
         struct vtvk_aabb bounds;
         vtvk_aabb_empty(&bounds);

         for (size_t j = i; j < leaves.size() && j < i + 4; j++) {
            children.push_back(leaves[j]);
            vtvk_aabb_include_aabb(&bounds, &leaves[j].bounds);
         }

         uint32_t offset = vtvk_next_node_offset(cursor, VENTUS_AS_BOX4_NODE_SIZE);
         vtvk_write_box4(dst, offset, children);
         (*node_count)++;

         next.push_back({
            ventus_bvh_make_node_ref(offset, VENTUS_BVH_NODE_BOX4),
            bounds,
         });
      }
      leaves = next;
   }

   return leaves[0];
}

struct vtvk_acceleration_structure *
vtvk_find_as_by_address(VkDeviceAddress address)
{
   for (struct vtvk_acceleration_structure *as : vtvk_as_registry) {
      if (as && (as->address == address ||
                 (as->snapshot &&
                  address >= (VkDeviceAddress)(uintptr_t)as->snapshot &&
                  address < (VkDeviceAddress)(uintptr_t)as->snapshot +
                               as->snapshot_size) ||
                 (as->data && (VkDeviceAddress)(uintptr_t)as->data == address) ||
                 (as->data && address >= (VkDeviceAddress)(uintptr_t)as->data &&
                  address < (VkDeviceAddress)(uintptr_t)as->data + as->size)))
         return as;
   }
   return nullptr;
}

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

static struct vtvk_pipeline *
vtvk_pipeline_from_handle(VkPipeline pipeline)
{
   VK_FROM_HANDLE(vk_pipeline, vk_pipeline, pipeline);
   return vk_pipeline ? container_of(vk_pipeline, struct vtvk_pipeline, vk) : nullptr;
}

static void
vtvk_pipeline_destroy(struct vk_device *device,
                      struct vk_pipeline *vk_pipeline,
                      const VkAllocationCallbacks *pAllocator)
{
   struct vtvk_pipeline *pipeline =
      container_of(vk_pipeline, struct vtvk_pipeline, vk);

   delete pipeline->llvm_module;
   free(pipeline->generated_kernel_path);
   ralloc_free(pipeline->nir_mem_ctx);
   vk_free2(&device->alloc, pAllocator, pipeline->shaders);
   vk_free2(&device->alloc, pAllocator, pipeline->group_handles);
   vk_pipeline_free(device, pAllocator, vk_pipeline);
}

static const struct vk_pipeline_ops vtvk_pipeline_ops = {
   .destroy = vtvk_pipeline_destroy,
};

VKAPI_ATTR void VKAPI_CALL
vtvk_DestroyPipeline(VkDevice _device,
                     VkPipeline _pipeline,
                     const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   VK_FROM_HANDLE(vk_pipeline, pipeline, _pipeline);

   if (!pipeline)
      return;

   pipeline->ops->destroy(&device->vk, pipeline, pAllocator);
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_CreateGraphicsPipelines(VkDevice _device,
                             VkPipelineCache pipelineCache,
                             uint32_t createInfoCount,
                             const VkGraphicsPipelineCreateInfo *pCreateInfos,
                             const VkAllocationCallbacks *pAllocator,
                             VkPipeline *pPipelines)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   (void)pipelineCache;

   VkResult final_result = VK_SUCCESS;
   for (uint32_t i = 0; i < createInfoCount; i++) {
      VkPipelineCreateFlags2KHR flags =
         vk_graphics_pipeline_create_flags(&pCreateInfos[i]);
      struct vtvk_pipeline *pipeline =
         (struct vtvk_pipeline *)vk_pipeline_zalloc(
            &device->vk, &vtvk_pipeline_ops, VK_PIPELINE_BIND_POINT_GRAPHICS,
            flags, pAllocator, sizeof(*pipeline));

      if (!pipeline) {
         pPipelines[i] = VK_NULL_HANDLE;
         final_result = vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

         if (pCreateInfos[i].flags &
             VK_PIPELINE_CREATE_EARLY_RETURN_ON_FAILURE_BIT) {
            for (uint32_t j = i + 1; j < createInfoCount; j++)
               pPipelines[j] = VK_NULL_HANDLE;
            break;
         }
         continue;
      }

      for (uint32_t s = 0; s < pCreateInfos[i].stageCount; s++)
         pipeline->vk.stages |= pCreateInfos[i].pStages[s].stage;

      pPipelines[i] = vk_pipeline_to_handle(&pipeline->vk);
   }

   return final_result;
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

   struct vt_nir_rt_config config = vt_nir_rt_default_config();
   struct vt_nir_rt_abi_layout abi = vt_nir_rt_default_abi_layout();
   vt_nir_rt_config_apply_abi_layout(&config, &abi);

   NIR_PASS(_, *nir_out, vt_nir_lower_rt_abi, &config);
   nir_validate_shader(*nir_out, "after Ventus RT lowering");
   if (vtvk_env_enabled("VENTUS_VK_DUMP_NIR"))
      nir_print_shader(*nir_out, stderr);

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

      NIR_PASS(_, cb_shaders.raygen, vt_nir_lower_rt_io_monolithic,
               &rt_io_config);
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

static const VkAccelerationStructureGeometryKHR *
vtvk_build_geometry(const VkAccelerationStructureBuildGeometryInfoKHR *info,
                    uint32_t index)
{
   return info->pGeometries ? &info->pGeometries[index] : info->ppGeometries[index];
}

static void
vtvk_read_vec3(const uint8_t *base, VkDeviceSize stride, uint32_t index,
               float out[3])
{
   const uint8_t *ptr = base + (VkDeviceSize)index * stride;
   memcpy(out, ptr, sizeof(float) * 3);
}

static void
vtvk_transform_point(const VkTransformMatrixKHR *m, const float in[3],
                     float out[3])
{
   for (unsigned row = 0; row < 3; row++) {
      out[row] = m->matrix[row][0] * in[0] +
                 m->matrix[row][1] * in[1] +
                 m->matrix[row][2] * in[2] +
                 m->matrix[row][3];
   }
}

static struct vtvk_aabb
vtvk_transform_aabb(const VkTransformMatrixKHR *m, const struct vtvk_aabb *in)
{
   struct vtvk_aabb out;
   vtvk_aabb_empty(&out);

   for (unsigned x = 0; x < 2; x++) {
      for (unsigned y = 0; y < 2; y++) {
         for (unsigned z = 0; z < 2; z++) {
            float p[3] = {
               x ? in->max[0] : in->min[0],
               y ? in->max[1] : in->min[1],
               z ? in->max[2] : in->min[2],
            };
            float tp[3];
            vtvk_transform_point(m, p, tp);
            vtvk_aabb_include_point(&out, tp);
         }
      }
   }

   return out;
}

static void
vtvk_write_triangle_leaf(uint8_t *dst, uint32_t offset, const float v0[3],
                         const float v1[3], const float v2[3],
                         uint32_t primitive_id, uint32_t geometry_id,
                         uint32_t sbt_record_offset, uint32_t flags,
                         uint64_t primitive_addr)
{
   uint8_t *node = dst + offset;
   memset(node, 0, VENTUS_AS_TRIANGLE_NODE_SIZE);
   memcpy(node + VENTUS_TRIANGLE_V0, v0, sizeof(float) * 3);
   memcpy(node + VENTUS_TRIANGLE_V1, v1, sizeof(float) * 3);
   memcpy(node + VENTUS_TRIANGLE_V2, v2, sizeof(float) * 3);
   vtvk_store_u32(node, VENTUS_TRIANGLE_PRIMITIVE_ID, primitive_id);
   vtvk_store_u32(node, VENTUS_TRIANGLE_GEOMETRY_ID, geometry_id);
   vtvk_store_u32(node, VENTUS_TRIANGLE_SBT_RECORD_OFFSET, sbt_record_offset);
   vtvk_store_u32(node, VENTUS_TRIANGLE_FLAGS, flags);
   vtvk_store_u64(node, VENTUS_TRIANGLE_PRIMITIVE_ADDR_LO, primitive_addr);
}

static void
vtvk_write_instance_leaf(uint8_t *dst, uint32_t offset,
                         const VkAccelerationStructureInstanceKHR *instance,
                         uint32_t instance_id)
{
   uint8_t *node = dst + offset;
   memset(node, 0, VENTUS_AS_INSTANCE_NODE_SIZE);
   vtvk_store_u64(node, VENTUS_INSTANCE_BLAS_ADDR_LO,
                  instance->accelerationStructureReference);
   vtvk_store_u32(node, VENTUS_INSTANCE_CUSTOM_INDEX,
                  instance->instanceCustomIndex);
   vtvk_store_u32(node, VENTUS_INSTANCE_MASK, instance->mask);
   vtvk_store_u32(node, VENTUS_INSTANCE_SBT_RECORD_OFFSET,
                  instance->instanceShaderBindingTableRecordOffset);
   vtvk_store_u32(node, VENTUS_INSTANCE_FLAGS, instance->flags);
   vtvk_store_u32(node, VENTUS_INSTANCE_INSTANCE_ID, instance_id);

   memcpy(node + VENTUS_INSTANCE_OBJECT_TO_WORLD, instance->transform.matrix,
          sizeof(instance->transform.matrix));

   float identity[3][4] = {
      { 1.0f, 0.0f, 0.0f, 0.0f },
      { 0.0f, 1.0f, 0.0f, 0.0f },
      { 0.0f, 0.0f, 1.0f, 0.0f },
   };
   memcpy(node + VENTUS_INSTANCE_WORLD_TO_OBJECT, identity, sizeof(identity));
}

static VkResult
vtvk_build_blas(struct vtvk_device *device,
                struct vtvk_acceleration_structure *as,
                const VkAccelerationStructureBuildGeometryInfoKHR *info,
                const VkAccelerationStructureBuildRangeInfoKHR *range)
{
   if (info->geometryCount != 1)
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);

   const VkAccelerationStructureGeometryKHR *geom = vtvk_build_geometry(info, 0);
   if (geom->geometryType != VK_GEOMETRY_TYPE_TRIANGLES_KHR)
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);

   const VkAccelerationStructureGeometryTrianglesDataKHR *triangles =
      &geom->geometry.triangles;
   if (triangles->vertexFormat != VK_FORMAT_R32G32B32_SFLOAT ||
       triangles->indexType != VK_INDEX_TYPE_UINT32)
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);

   uint32_t primitive_count = range ? range->primitiveCount : 0;
   if (!primitive_count)
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);

   uint8_t *dst = (uint8_t *)as->data;
   if (!dst)
      return vk_error(device, VK_ERROR_UNKNOWN);

   uint8_t *vertex_base = (uint8_t *)vtvk_buffer_host_ptr_from_addr(
      triangles->vertexData.deviceAddress, triangles->vertexStride *
      (triangles->maxVertex + 1));
   uint32_t *index_base = (uint32_t *)vtvk_buffer_host_ptr_from_addr(
      triangles->indexData.deviceAddress, sizeof(uint32_t) * primitive_count * 3);
   if (!vertex_base || !index_base)
      return vk_error(device, VK_ERROR_UNKNOWN);

   memset(dst, 0, as->size);
   uint32_t cursor = VENTUS_AS_HEADER_SIZE;
   uint32_t node_count = 0;
   std::vector<struct vtvk_build_node> leaves;

   for (uint32_t p = 0; p < primitive_count; p++) {
      uint32_t i0 = index_base[p * 3 + 0];
      uint32_t i1 = index_base[p * 3 + 1];
      uint32_t i2 = index_base[p * 3 + 2];
      float v0[3], v1[3], v2[3];
      vtvk_read_vec3(vertex_base, triangles->vertexStride, i0, v0);
      vtvk_read_vec3(vertex_base, triangles->vertexStride, i1, v1);
      vtvk_read_vec3(vertex_base, triangles->vertexStride, i2, v2);

      struct vtvk_aabb bounds;
      vtvk_aabb_empty(&bounds);
      vtvk_aabb_include_point(&bounds, v0);
      vtvk_aabb_include_point(&bounds, v1);
      vtvk_aabb_include_point(&bounds, v2);

      uint32_t offset =
         vtvk_next_node_offset(&cursor, VENTUS_AS_TRIANGLE_NODE_SIZE);
      vtvk_write_triangle_leaf(dst, offset, v0, v1, v2, p, 0, 0,
                               geom->flags & VK_GEOMETRY_OPAQUE_BIT_KHR ? 1 : 0,
                               triangles->vertexData.deviceAddress +
                                  (VkDeviceAddress)p * triangles->vertexStride);
      node_count++;
      leaves.push_back({
         ventus_bvh_make_node_ref(offset, VENTUS_BVH_NODE_TRIANGLE),
         bounds,
      });
   }

   struct vtvk_build_node root =
      vtvk_build_box_tree(dst, &cursor, leaves, &node_count);
   vtvk_write_header(dst, VENTUS_AS_TYPE_BLAS, root.ref, node_count,
                     cursor - VENTUS_AS_HEADER_SIZE, primitive_count, 0,
                     &root.bounds);
   as->address = (VkDeviceAddress)(uintptr_t)as->data;
   return VK_SUCCESS;
}

static VkResult
vtvk_build_tlas(struct vtvk_device *device,
                struct vtvk_acceleration_structure *as,
                const VkAccelerationStructureBuildGeometryInfoKHR *info,
                const VkAccelerationStructureBuildRangeInfoKHR *range)
{
   if (info->geometryCount != 1)
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);

   const VkAccelerationStructureGeometryKHR *geom = vtvk_build_geometry(info, 0);
   if (geom->geometryType != VK_GEOMETRY_TYPE_INSTANCES_KHR)
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);

   uint32_t instance_count = range ? range->primitiveCount : 0;
   if (!instance_count)
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);

   const VkAccelerationStructureGeometryInstancesDataKHR *instances =
      &geom->geometry.instances;
   if (instances->arrayOfPointers)
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);

   VkAccelerationStructureInstanceKHR *instance_base =
      (VkAccelerationStructureInstanceKHR *)vtvk_buffer_host_ptr_from_addr(
         instances->data.deviceAddress,
         sizeof(VkAccelerationStructureInstanceKHR) * instance_count);
   if (!instance_base)
      return vk_error(device, VK_ERROR_UNKNOWN);

   uint8_t *dst = (uint8_t *)as->data;
   if (!dst)
      return vk_error(device, VK_ERROR_UNKNOWN);

   memset(dst, 0, as->size);
   uint32_t cursor = VENTUS_AS_HEADER_SIZE;
   uint32_t node_count = 0;
   std::vector<struct vtvk_build_node> leaves;

   for (uint32_t i = 0; i < instance_count; i++) {
      const VkAccelerationStructureInstanceKHR *inst = &instance_base[i];
      const uint8_t *blas = (const uint8_t *)(uintptr_t)inst->accelerationStructureReference;
      if (!blas || vtvk_load_u32(blas, VENTUS_AS_HEADER_MAGIC) != VENTUS_AS_MAGIC)
         return vk_error(device, VK_ERROR_UNKNOWN);

      struct vtvk_aabb blas_bounds = vtvk_header_aabb(blas);
      struct vtvk_aabb bounds = vtvk_transform_aabb(&inst->transform,
                                                    &blas_bounds);
      uint32_t offset =
         vtvk_next_node_offset(&cursor, VENTUS_AS_INSTANCE_NODE_SIZE);
      vtvk_write_instance_leaf(dst, offset, inst, i);
      if (vtvk_env_enabled("VENTUS_VK_DEBUG_AS"))
         mesa_logi("ventus-vk: TLAS instance %u node=0x%x blas_ref=0x%" PRIx64
                   " stored=0x%" PRIx64,
                   i, offset,
                   (uint64_t)inst->accelerationStructureReference,
                   vtvk_load_u64(dst + offset, VENTUS_INSTANCE_BLAS_ADDR_LO));
      node_count++;
      leaves.push_back({
         ventus_bvh_make_node_ref(offset, VENTUS_BVH_NODE_INSTANCE),
         bounds,
      });
   }

   struct vtvk_build_node root =
      vtvk_build_box_tree(dst, &cursor, leaves, &node_count);
   vtvk_write_header(dst, VENTUS_AS_TYPE_TLAS, root.ref, node_count,
                     cursor - VENTUS_AS_HEADER_SIZE, 0, instance_count,
                     &root.bounds);
   as->address = (VkDeviceAddress)(uintptr_t)as->data;
   return VK_SUCCESS;
}

static VkResult
vtvk_build_as(struct vtvk_device *device,
              const VkAccelerationStructureBuildGeometryInfoKHR *info,
              const VkAccelerationStructureBuildRangeInfoKHR *const *ranges)
{
   struct vtvk_acceleration_structure *as =
      (struct vtvk_acceleration_structure *)
         vtvk_handle_to_ptr((uint64_t)info->dstAccelerationStructure);
   if (!as)
      return vk_error(device, VK_ERROR_UNKNOWN);

   as->data = vtvk_buffer_host_ptr(as->buffer, as->offset, as->size);
   if (!as->data)
      return vk_error(device, VK_ERROR_UNKNOWN);

   const VkAccelerationStructureBuildRangeInfoKHR *range =
      ranges ? ranges[0] : NULL;

   VkResult result;
   if (info->type == VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR)
      result = vtvk_build_blas(device, as, info, range);
   else if (info->type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR)
      result = vtvk_build_tlas(device, as, info, range);
   else
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);

   if (result != VK_SUCCESS)
      return result;

   free(as->snapshot);
   as->snapshot = (uint8_t *)malloc(as->size);
   if (!as->snapshot) {
      as->snapshot_size = 0;
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   memcpy(as->snapshot, as->data, as->size);
   as->snapshot_size = as->size;
   as->address = (VkDeviceAddress)(uintptr_t)as->snapshot;
   if (vtvk_env_enabled("VENTUS_VK_DEBUG_AS") &&
       info->type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR) {
      uint32_t root_ref =
         vtvk_load_u32(as->snapshot, VENTUS_AS_HEADER_ROOT_NODE_REF);
      uint32_t root_offset = ventus_bvh_node_ref_offset(root_ref);
      mesa_logi("ventus-vk: TLAS snapshot=%p size=%" PRIu64
                " root=0x%x root_blas=0x%" PRIx64,
                (void *)as->snapshot, (uint64_t)as->snapshot_size, root_ref,
                root_offset < as->snapshot_size ?
                   vtvk_load_u64(as->snapshot + root_offset,
                                 VENTUS_INSTANCE_BLAS_ADDR_LO) : 0);
   }
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdTraceRaysKHR(VkCommandBuffer commandBuffer,
                     const VkStridedDeviceAddressRegionKHR *pRaygenShaderBindingTable,
                     const VkStridedDeviceAddressRegionKHR *pMissShaderBindingTable,
                     const VkStridedDeviceAddressRegionKHR *pHitShaderBindingTable,
                     const VkStridedDeviceAddressRegionKHR *pCallableShaderBindingTable,
                     uint32_t width,
                     uint32_t height,
                     uint32_t depth)
{
   static bool logged;

   VkPipeline pipeline_handle =
      vtvk_cmd_buffer_bound_rt_pipeline(commandBuffer);
   struct vtvk_pipeline *pipeline =
      vtvk_pipeline_from_handle(pipeline_handle);
   VkDescriptorSet descriptor_set =
      vtvk_cmd_buffer_bound_descriptor_set(commandBuffer, 0);
   VkAccelerationStructureKHR tlas_handle =
      vtvk_descriptor_set_acceleration_structure(descriptor_set, 0);
   struct vtvk_acceleration_structure *tlas =
      (struct vtvk_acceleration_structure *)
         vtvk_handle_to_ptr((uint64_t)tlas_handle);

   VkExtent3D image_extent = {};
   VkFormat image_format = VK_FORMAT_UNDEFINED;
   VkImage storage_image = vtvk_descriptor_set_image(descriptor_set, 1);
   void *storage_image_data =
      vtvk_image_host_ptr(storage_image, &image_extent, &image_format);

   VkDeviceSize camera_range = 0;
   void *camera_data =
      vtvk_descriptor_set_buffer_ptr(descriptor_set, 2, &camera_range);

   if (!pipeline || !tlas || !tlas->address || !storage_image_data ||
       !camera_data) {
      mesa_loge("ventus-vk: vkCmdTraceRaysKHR missing Spike dispatch state "
                "pipeline=%p tlas=0x%" PRIx64 " image=%p camera=%p",
                (void *)pipeline, tlas ? (uint64_t)tlas->address : 0,
                storage_image_data, camera_data);
      return;
   }

   if (!logged) {
      logged = true;
      mesa_logi("ventus-vk: vkCmdTraceRaysKHR collected Spike dispatch state "
                "%ux%ux%u tlas=0x%" PRIx64 " image=%ux%u format=%u "
                "scratch=%u sbt={raygen:0x%" PRIx64 ",miss:0x%" PRIx64
                ",hit:0x%" PRIx64 "}",
                width, height, depth, (uint64_t)tlas->address,
                image_extent.width, image_extent.height, image_format,
                pipeline->per_lane_scratch_size,
                pRaygenShaderBindingTable ?
                   (uint64_t)pRaygenShaderBindingTable->deviceAddress : 0,
                pMissShaderBindingTable ?
                   (uint64_t)pMissShaderBindingTable->deviceAddress : 0,
                pHitShaderBindingTable ?
                   (uint64_t)pHitShaderBindingTable->deviceAddress : 0);
   }

   struct vtvk_driver_trace_rays_info trace = {};
   trace.tlas = tlas;
   trace.storage_image_data = storage_image_data;
   trace.image_extent = image_extent;
   trace.image_format = image_format;
   trace.camera_data = camera_data;
   trace.camera_range = camera_range;
   trace.raygen_sbt = pRaygenShaderBindingTable;
   trace.miss_sbt = pMissShaderBindingTable;
   trace.hit_sbt = pHitShaderBindingTable;
   trace.callable_sbt = pCallableShaderBindingTable;
   trace.width = width;
   trace.height = height;
   trace.depth = depth;
   trace.per_lane_scratch_size = pipeline->per_lane_scratch_size;
   trace.kernel_path = pipeline->generated_kernel_path;
   trace.kernel_entry = pipeline->generated_kernel_entry;

   vtvk_driver_trace_rays(&trace);
}

VKAPI_ATTR VkDeviceAddress VKAPI_CALL
vtvk_GetAccelerationStructureDeviceAddressKHR(
   VkDevice device,
   const VkAccelerationStructureDeviceAddressInfoKHR *pInfo)
{
   (void)device;
   struct vtvk_acceleration_structure *as =
      (struct vtvk_acceleration_structure *)
         vtvk_handle_to_ptr((uint64_t)pInfo->accelerationStructure);
   if (as && as->snapshot)
      return (VkDeviceAddress)(uintptr_t)as->snapshot;
   if (as && !as->address) {
      as->data = vtvk_buffer_host_ptr(as->buffer, as->offset, as->size);
      as->address = (VkDeviceAddress)(uintptr_t)as->data;
   }
   return as ? as->address : 0;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_CreateAccelerationStructureKHR(
   VkDevice _device,
   const VkAccelerationStructureCreateInfoKHR *pCreateInfo,
   const VkAllocationCallbacks *pAllocator,
   VkAccelerationStructureKHR *pAccelerationStructure)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc =
      pAllocator ? pAllocator : &device->vk.alloc;

   struct vtvk_acceleration_structure *as =
      (struct vtvk_acceleration_structure *)
         vk_zalloc(alloc, sizeof(*as), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!as)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   as->type = pCreateInfo->type;
   as->buffer = pCreateInfo->buffer;
   as->offset = pCreateInfo->offset;
   as->size = pCreateInfo->size;
   as->data = vtvk_buffer_host_ptr(as->buffer, as->offset, as->size);
   as->address = (VkDeviceAddress)(uintptr_t)as->data;

   *pAccelerationStructure =
      (VkAccelerationStructureKHR)vtvk_ptr_to_handle(as);
   vtvk_as_registry.push_back(as);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_DestroyAccelerationStructureKHR(
   VkDevice _device,
   VkAccelerationStructureKHR accelerationStructure,
   const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc =
      pAllocator ? pAllocator : &device->vk.alloc;
   struct vtvk_acceleration_structure *as =
      (struct vtvk_acceleration_structure *)
         vtvk_handle_to_ptr((uint64_t)accelerationStructure);
   vtvk_as_registry.erase(std::remove(vtvk_as_registry.begin(),
                                      vtvk_as_registry.end(), as),
                          vtvk_as_registry.end());
   if (as)
      free(as->snapshot);
   vk_free(alloc, as);
}

VKAPI_ATTR void VKAPI_CALL
vtvk_GetAccelerationStructureBuildSizesKHR(
   VkDevice device,
   VkAccelerationStructureBuildTypeKHR buildType,
   const VkAccelerationStructureBuildGeometryInfoKHR *pBuildInfo,
   const uint32_t *pMaxPrimitiveCounts,
   VkAccelerationStructureBuildSizesInfoKHR *pSizeInfo)
{
   (void)device;
   (void)buildType;
   uint64_t primitive_count = 1;
   if (pMaxPrimitiveCounts && pBuildInfo && pBuildInfo->geometryCount)
      primitive_count = pMaxPrimitiveCounts[0] ? pMaxPrimitiveCounts[0] : 1;

   uint64_t leaf_size =
      pBuildInfo && pBuildInfo->type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR
         ? VENTUS_AS_INSTANCE_NODE_SIZE : VENTUS_AS_TRIANGLE_NODE_SIZE;
   uint64_t internal_count = primitive_count > 1 ? (primitive_count + 2) / 3 : 0;
   uint64_t internal_bytes = internal_count * VENTUS_AS_BOX4_NODE_SIZE;
   uint64_t primitive_bytes = primitive_count * leaf_size;

   pSizeInfo->accelerationStructureSize =
      VENTUS_AS_HEADER_SIZE + primitive_bytes + internal_bytes + 256;
   pSizeInfo->updateScratchSize = 2048 + primitive_bytes;
   pSizeInfo->buildScratchSize = 4096 + primitive_bytes;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdBuildAccelerationStructuresKHR(
   VkCommandBuffer commandBuffer,
   uint32_t infoCount,
   const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
   const VkAccelerationStructureBuildRangeInfoKHR *const *ppBuildRangeInfos)
{
   VK_FROM_HANDLE(vk_command_buffer, vk_cmd, commandBuffer);
   struct vtvk_device *device =
      container_of(vk_cmd->base.device, struct vtvk_device, vk);

   for (uint32_t i = 0; i < infoCount; i++) {
      VkResult result =
         vtvk_build_as(device, &pInfos[i], ppBuildRangeInfos + i);
      if (result != VK_SUCCESS) {
         mesa_loge("ventus-vk: AS build failed for Ventus BVH ABI v1 (%d)",
                   result);
         return;
      }
   }

   mesa_logi("ventus-vk: AS build wrote Ventus BVH ABI v1 objects=%u",
             infoCount);
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdBuildAccelerationStructuresIndirectKHR(
   VkCommandBuffer commandBuffer,
   uint32_t infoCount,
   const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
   const VkDeviceAddress *pIndirectDeviceAddresses,
   const uint32_t *pIndirectStrides,
   const uint32_t *const *ppMaxPrimitiveCounts)
{
   (void)commandBuffer;
   (void)infoCount;
   (void)pInfos;
   (void)pIndirectDeviceAddresses;
   (void)pIndirectStrides;
   (void)ppMaxPrimitiveCounts;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_BuildAccelerationStructuresKHR(
   VkDevice device,
   VkDeferredOperationKHR deferredOperation,
   uint32_t infoCount,
   const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
   const VkAccelerationStructureBuildRangeInfoKHR *const *ppBuildRangeInfos)
{
   (void)device;
   (void)deferredOperation;
   VK_FROM_HANDLE(vtvk_device, vdevice, device);

   for (uint32_t i = 0; i < infoCount; i++) {
      VkResult result =
         vtvk_build_as(vdevice, &pInfos[i], ppBuildRangeInfos + i);
      if (result != VK_SUCCESS)
         return result;
   }
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_CopyAccelerationStructureKHR(
   VkDevice device,
   VkDeferredOperationKHR deferredOperation,
   const VkCopyAccelerationStructureInfoKHR *pInfo)
{
   (void)device;
   (void)deferredOperation;
   (void)pInfo;
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_CopyAccelerationStructureToMemoryKHR(
   VkDevice device,
   VkDeferredOperationKHR deferredOperation,
   const VkCopyAccelerationStructureToMemoryInfoKHR *pInfo)
{
   (void)device;
   (void)deferredOperation;
   (void)pInfo;
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_CopyMemoryToAccelerationStructureKHR(
   VkDevice device,
   VkDeferredOperationKHR deferredOperation,
   const VkCopyMemoryToAccelerationStructureInfoKHR *pInfo)
{
   (void)device;
   (void)deferredOperation;
   (void)pInfo;
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_WriteAccelerationStructuresPropertiesKHR(
   VkDevice device,
   uint32_t accelerationStructureCount,
   const VkAccelerationStructureKHR *pAccelerationStructures,
   VkQueryType queryType,
   size_t dataSize,
   void *pData,
   size_t stride)
{
   (void)device;
   (void)accelerationStructureCount;
   (void)pAccelerationStructures;
   (void)queryType;
   (void)dataSize;
   (void)pData;
   (void)stride;
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdCopyAccelerationStructureKHR(
   VkCommandBuffer commandBuffer,
   const VkCopyAccelerationStructureInfoKHR *pInfo)
{
   (void)commandBuffer;
   (void)pInfo;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdCopyAccelerationStructureToMemoryKHR(
   VkCommandBuffer commandBuffer,
   const VkCopyAccelerationStructureToMemoryInfoKHR *pInfo)
{
   (void)commandBuffer;
   (void)pInfo;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdCopyMemoryToAccelerationStructureKHR(
   VkCommandBuffer commandBuffer,
   const VkCopyMemoryToAccelerationStructureInfoKHR *pInfo)
{
   (void)commandBuffer;
   (void)pInfo;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdWriteAccelerationStructuresPropertiesKHR(
   VkCommandBuffer commandBuffer,
   uint32_t accelerationStructureCount,
   const VkAccelerationStructureKHR *pAccelerationStructures,
   VkQueryType queryType,
   VkQueryPool queryPool,
   uint32_t firstQuery)
{
   (void)commandBuffer;
   (void)accelerationStructureCount;
   (void)pAccelerationStructures;
   (void)queryType;
   (void)queryPool;
   (void)firstQuery;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_GetDeviceAccelerationStructureCompatibilityKHR(
   VkDevice device,
   const VkAccelerationStructureVersionInfoKHR *pVersionInfo,
   VkAccelerationStructureCompatibilityKHR *pCompatibility)
{
   (void)device;
   (void)pVersionInfo;
   *pCompatibility = VK_ACCELERATION_STRUCTURE_COMPATIBILITY_INCOMPATIBLE_KHR;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_GetRayTracingCaptureReplayShaderGroupHandlesKHR(
   VkDevice device,
   VkPipeline pipeline,
   uint32_t firstGroup,
   uint32_t groupCount,
   size_t dataSize,
   void *pData)
{
   return vtvk_GetRayTracingShaderGroupHandlesKHR(device, pipeline, firstGroup,
                                                  groupCount, dataSize, pData);
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdTraceRaysIndirectKHR(
   VkCommandBuffer commandBuffer,
   const VkStridedDeviceAddressRegionKHR *pRaygenShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pMissShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pHitShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pCallableShaderBindingTable,
   VkDeviceAddress indirectDeviceAddress)
{
   (void)commandBuffer;
   (void)pRaygenShaderBindingTable;
   (void)pMissShaderBindingTable;
   (void)pHitShaderBindingTable;
   (void)pCallableShaderBindingTable;
   (void)indirectDeviceAddress;
}

VKAPI_ATTR VkDeviceSize VKAPI_CALL
vtvk_GetRayTracingShaderGroupStackSizeKHR(
   VkDevice device,
   VkPipeline pipeline,
   uint32_t group,
   VkShaderGroupShaderKHR groupShader)
{
   (void)device;
   (void)pipeline;
   (void)group;
   (void)groupShader;
   return 0;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdSetRayTracingPipelineStackSizeKHR(VkCommandBuffer commandBuffer,
                                          uint32_t pipelineStackSize)
{
   (void)commandBuffer;
   (void)pipelineStackSize;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_CreateDeferredOperationKHR(VkDevice device,
                                const VkAllocationCallbacks *pAllocator,
                                VkDeferredOperationKHR *pDeferredOperation)
{
   (void)device;
   (void)pAllocator;
   *pDeferredOperation = VK_NULL_HANDLE;
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_DestroyDeferredOperationKHR(VkDevice device,
                                 VkDeferredOperationKHR operation,
                                 const VkAllocationCallbacks *pAllocator)
{
   (void)device;
   (void)operation;
   (void)pAllocator;
}

VKAPI_ATTR uint32_t VKAPI_CALL
vtvk_GetDeferredOperationMaxConcurrencyKHR(VkDevice device,
                                           VkDeferredOperationKHR operation)
{
   (void)device;
   (void)operation;
   return 0;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_GetDeferredOperationResultKHR(VkDevice device,
                                   VkDeferredOperationKHR operation)
{
   (void)device;
   (void)operation;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_DeferredOperationJoinKHR(VkDevice device,
                              VkDeferredOperationKHR operation)
{
   (void)device;
   (void)operation;
   return VK_SUCCESS;
}

VKAPI_ATTR VkDeviceAddress VKAPI_CALL
vtvk_GetBufferDeviceAddressKHR(VkDevice device,
                               const VkBufferDeviceAddressInfo *pInfo)
{
   (void)device;
   return pInfo ? vtvk_buffer_get_device_address(pInfo->buffer) : 0;
}

VKAPI_ATTR uint64_t VKAPI_CALL
vtvk_GetBufferOpaqueCaptureAddressKHR(VkDevice device,
                                      const VkBufferDeviceAddressInfo *pInfo)
{
   (void)device;
   (void)pInfo;
   return 0;
}

VKAPI_ATTR uint64_t VKAPI_CALL
vtvk_GetDeviceMemoryOpaqueCaptureAddressKHR(
   VkDevice device,
   const VkDeviceMemoryOpaqueCaptureAddressInfo *pInfo)
{
   (void)device;
   (void)pInfo;
   return 0;
}
