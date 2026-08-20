vt_rt_profile_test = executable(
  'vt_rt_profile_test',
  [
    'vt_rt_profile_test.c',
    '../vulkan/vtvk_rt_profile.c',
  ],
  include_directories : [inc_include, inc_src],
  build_by_default : false,
)

vt_kernel_metadata_layout_test = executable(
  'vt_kernel_metadata_layout_test',
  'vt_kernel_metadata_layout_test.c',
  include_directories : [inc_src],
  build_by_default : false,
)

vt_kernel_resource_v4_test = executable(
  'vt_kernel_resource_v4_test',
  'vt_kernel_resource_v4_test.c',
  include_directories : [inc_src],
  build_by_default : false,
)

test(
  'vt_rt_profile',
  vt_rt_profile_test,
  suite : ['ventus'],
)

test(
  'vt_kernel_metadata_layout',
  vt_kernel_metadata_layout_test,
  suite : ['ventus'],
)

test(
  'vt_kernel_resource_v4',
  vt_kernel_resource_v4_test,
  suite : ['ventus'],
)

if prog_glslang.found()
  vt_rt_minimal_raygen_spv = custom_target(
    'vt_rt_minimal_raygen.spv',
    input : 'vt_rt_minimal_raygen.rgen',
    output : 'vt_rt_minimal_raygen.spv',
    command : [
      prog_glslang, '-V', '--target-env', 'vulkan1.2',
      '-o', '@OUTPUT@', '@INPUT@',
      glslang_quiet, glslang_depfile,
    ],
    depfile : 'vt_rt_minimal_raygen.spv.d',
  )

  vt_rt_flagged_raygen_spv = custom_target(
    'vt_rt_flagged_raygen.spv',
    input : 'vt_rt_flagged_raygen.rgen',
    output : 'vt_rt_flagged_raygen.spv',
    command : [
      prog_glslang, '-V', '--target-env', 'vulkan1.2',
      '-o', '@OUTPUT@', '@INPUT@',
      glslang_quiet, glslang_depfile,
    ],
    depfile : 'vt_rt_flagged_raygen.spv.d',
  )

  vt_rt_cps_loop_raygen_spv = custom_target(
    'vt_rt_cps_loop_raygen.spv',
    input : 'vt_rt_cps_loop_raygen.rgen',
    output : 'vt_rt_cps_loop_raygen.spv',
    command : [
      prog_glslang, '-V', '--target-env', 'vulkan1.2',
      '-o', '@OUTPUT@', '@INPUT@',
      glslang_quiet, glslang_depfile,
    ],
    depfile : 'vt_rt_cps_loop_raygen.spv.d',
  )

  vt_rt_minimal_loop_test = executable(
    'vt_rt_minimal_loop_test',
    [
      'vt_rt_minimal_loop_test.c',
      libventus_nir_files,
    ],
    include_directories : [inc_include, inc_src],
    dependencies : [idep_vtn, idep_nir, idep_mesautil],
    build_by_default : false,
  )

  vt_rt_image_smoke_test = executable(
    'vt_rt_image_smoke_test',
    [
      'vt_rt_image_smoke_test.c',
      libventus_nir_files,
    ],
    include_directories : [inc_include, inc_src],
    dependencies : [idep_vtn, idep_nir, idep_mesautil, dep_m],
    build_by_default : false,
  )

  vt_rt_flagged_image_smoke_test = executable(
    'vt_rt_flagged_image_smoke_test',
    [
      'vt_rt_flagged_image_smoke_test.c',
      libventus_nir_files,
    ],
    include_directories : [inc_include, inc_src],
    dependencies : [idep_vtn, idep_nir, idep_mesautil, dep_m],
    build_by_default : false,
  )

  vt_nir_rt_canonicalize_test = executable(
    'vt_nir_rt_canonicalize_test',
    [
      'vt_nir_rt_canonicalize_test.c',
      libventus_nir_files,
    ],
    include_directories : [inc_include, inc_src],
    dependencies : [idep_nir, idep_mesautil],
    build_by_default : false,
  )

  vt_nir_to_llvm_phi_test = executable(
    'vt_nir_to_llvm_phi_test',
    [
      'vt_nir_to_llvm_phi_test.cpp',
      libventus_llvm_files,
    ],
    include_directories : [inc_include, inc_src],
    dependencies : [dep_llvm, idep_nir, idep_mesautil],
    build_by_default : false,
  )

  vt_rt_global_abi_test = executable(
    'vt_rt_global_abi_test',
    [
      'vt_rt_global_abi_test.c',
      libventus_nir_files,
    ],
    include_directories : [inc_include, inc_src],
    dependencies : [idep_nir, idep_mesautil],
    build_by_default : false,
  )

  vt_rt_global_queue_nir_test = executable(
    'vt_rt_global_queue_nir_test',
    [
      'vt_rt_global_queue_nir_test.c',
      libventus_nir_files,
    ],
    include_directories : [inc_include, inc_src],
    dependencies : [idep_nir, idep_mesautil],
    build_by_default : false,
  )

  vt_rt_abi_v4_layout_test = executable(
    'vt_rt_abi_v4_layout_test',
    [
      'vt_rt_abi_v4_layout_test.c',
      libventus_nir_files,
    ],
    include_directories : [inc_include, inc_src],
    dependencies : [idep_nir, idep_mesautil],
    build_by_default : false,
  )

  test(
    'vt_rt_minimal_loop',
    vt_rt_minimal_loop_test,
    args : [vt_rt_minimal_raygen_spv, vt_rt_cps_loop_raygen_spv],
    suite : ['ventus'],
  )

  test(
    'vt_rt_image_smoke',
    vt_rt_image_smoke_test,
    args : [
      vt_rt_minimal_raygen_spv,
      meson.current_build_dir() / 'vt_rt_smoke.ppm',
    ],
    suite : ['ventus'],
  )

  test(
    'vt_rt_flagged_image_smoke',
    vt_rt_flagged_image_smoke_test,
    args : [
      vt_rt_flagged_raygen_spv,
      meson.current_build_dir() / 'vt_rt_flagged_smoke.ppm',
    ],
    suite : ['ventus'],
  )

  test(
    'vt_nir_rt_canonicalize',
    vt_nir_rt_canonicalize_test,
    suite : ['ventus'],
  )

  test(
    'vt_nir_to_llvm_phi',
    vt_nir_to_llvm_phi_test,
    suite : ['ventus'],
  )

  test(
    'vt_rt_global_abi',
    vt_rt_global_abi_test,
    suite : ['ventus'],
  )

  test(
    'vt_rt_global_queue_nir',
    vt_rt_global_queue_nir_test,
    suite : ['ventus'],
  )

  test(
    'vt_rt_abi_v4_layout',
    vt_rt_abi_v4_layout_test,
    suite : ['ventus'],
  )
endif
