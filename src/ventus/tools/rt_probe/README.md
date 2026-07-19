# Ventus RT Driver Probe

`probe_write_pixel.ll` is a minimal RV32 Spike-driver kernel for isolating the
Mesa -> driver -> Spike bridge from generated raygen code. Generated RT ELF
execution is now the full-app baseline, so this probe is diagnostic evidence,
not the current rendering path.

It requests the kernel metadata pointer through
`llvm.riscv.ventus.kernel.metadata()`, loads `args.image_addr`, and writes four
diagnostic pixels:

- pixel 0: fixed RGBA test pattern
- pixel 1: low 32 bits of the dispatch args device address
- pixel 2: low 32 bits of the storage image device address
- pixel 3: dispatch width

Build it with:

```sh
bash mesa/src/ventus/tools/rt_probe/build_probe.sh
```

Run `raytracingbasic` with `VENTUS_VK_RAYGEN_ELF` pointing at the printed ELF.
