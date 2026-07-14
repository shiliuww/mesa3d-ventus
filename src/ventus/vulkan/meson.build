vtvk_entrypoints = custom_target(
  'vtvk_entrypoints',
  input : [vk_entrypoints_gen, vk_api_xml],
  output : ['vtvk_entrypoints.h', 'vtvk_entrypoints.c'],
  command : [
    prog_python, '@INPUT0@', '--xml', '@INPUT1@', '--proto', '--weak',
    '--out-h', '@OUTPUT0@', '--out-c', '@OUTPUT1@', '--prefix', 'vtvk',
    '--beta', with_vulkan_beta.to_string()
  ],
  depend_files : vk_entrypoints_gen_depend_files,
)

libvtvk_files = files(
  'vtvk_acceleration_structure.cpp',
  'vtvk_cmd_buffer.c',
  'vtvk_descriptors.c',
  'vtvk_device.c',
  'vtvk_driver_bridge.cpp',
  'vtvk_instance.c',
  'vtvk_memory.c',
  'vtvk_objects.c',
  'vtvk_physical_device.c',
  'vtvk_pipeline.cpp',
  'vtvk_pipeline_graphics.cpp',
  'vtvk_pipeline_rt.cpp',
  'vtvk_render_compat.c',
  'vtvk_rt_unsupported.cpp',
  'vtvk_shader.c',
  'vtvk_sync.c',
  'vtvk_trace.cpp',
)

libvulkan_ventus = shared_library(
  'vulkan_ventus',
  [libvtvk_files, libventus_nir_files, libventus_llvm_files, vtvk_entrypoints],
  include_directories : [
    inc_include,
    inc_src,
    inc_vulkan_util,
  ],
  dependencies : [
    dep_m,
    dep_thread,
    dep_llvm,
    idep_mesautil,
    idep_nir,
    idep_vtn,
    idep_vulkan_runtime,
    idep_vulkan_wsi,
  ],
  c_args : [
    no_override_init_args,
    soong_compat_c_args,
    '-DVENTUS_RT_RESIDENT_WARPS=' + get_option('ventus_rt_resident_warps').to_string(),
  ],
  cpp_args : [
    '-DVENTUS_RT_RESIDENT_WARPS=' + get_option('ventus_rt_resident_warps').to_string(),
  ],
  link_args : [vulkan_icd_link_args, ld_args_bsymbolic, ld_args_gc_sections, ld_args_build_id],
  link_depends : vulkan_icd_link_depends,
  gnu_symbol_visibility : 'hidden',
  install : true,
)

if with_symbols_check
  test(
    'vtvk symbols check',
    symbols_check,
    args : [
      '--lib', libvulkan_ventus,
      '--symbols-file', vulkan_icd_symbols,
      symbols_check_args,
    ],
    suite : ['ventus'],
  )
endif

icd_file_name = libname_prefix + 'vulkan_ventus.' + libname_suffix

ventus_icd = custom_target(
  'ventus_icd',
  input : [vk_icd_gen, vk_api_xml],
  output : 'ventus_icd.' + vulkan_manifest_suffix,
  command : [
    prog_python, '@INPUT0@',
    '--api-version', '1.3', '--xml', '@INPUT1@',
    '--sizeof-pointer', sizeof_pointer,
    '--icd-lib-path', vulkan_icd_lib_path,
    '--icd-filename', icd_file_name,
    '--out', '@OUTPUT@',
  ],
  build_by_default : true,
  install_dir : with_vulkan_icd_dir,
  install_tag : 'runtime',
  install : true,
)

_dev_icdname = 'ventus_devenv_icd.@0@.json'.format(host_machine.cpu())
_dev_icd = custom_target(
  'ventus_devenv_icd',
  input : [vk_icd_gen, vk_api_xml],
  output : _dev_icdname,
  command : [
    prog_python, '@INPUT0@',
    '--api-version', '1.3', '--xml', '@INPUT1@',
    '--sizeof-pointer', sizeof_pointer,
    '--icd-lib-path', meson.current_build_dir(),
    '--icd-filename', icd_file_name,
    '--out', '@OUTPUT@',
  ],
  build_by_default : true,
)

devenv.append('VK_DRIVER_FILES', _dev_icd.full_path())
