# Direct Get algorithms

- `urma_aiv/`: URMA AIV Direct Get, including its Host transport binding,
  launcher, ABI and kernel.
- `ubmem_aiv/`: UBMEM AIV Direct Get kernel. The UBMEM provider supplies its
  mapping and execution resources.
- `common/kernel/`: builds both AIV entry points in one translation unit to
  preserve CANN's single definition of device globals. This packaging does not
  merge the algorithms' resource ownership.

The shared build continues to produce `bin/kvaf_execute_batch_get_kernels.o`
and `bin/libkvaf_execute_batch_get_host.so`; runtime entry names and ABI are
unchanged by the directory split. Build through the repository's `build.sh`.
