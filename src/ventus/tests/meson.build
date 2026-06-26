vt_rt_minimal_loop_test = executable(
  'vt_rt_minimal_loop_test',
  [
    'vt_rt_minimal_loop_test.c',
    libventus_nir_files,
  ],
  include_directories : [inc_include, inc_src],
  dependencies : [idep_nir, idep_mesautil],
  build_by_default : false,
)

test(
  'vt_rt_minimal_loop',
  vt_rt_minimal_loop_test,
  suite : ['ventus'],
)
