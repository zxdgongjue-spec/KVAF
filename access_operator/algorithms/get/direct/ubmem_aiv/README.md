# UBMEM AIV Direct Get

`kernel/execute_batch_get_ubmem.asc` reads the mapped CPU DDR allocation and
copies requested tokens to the caller's HBM output. Mapping and execution
resources are supplied by the UBMEM provider.

The kernel is compiled by `../common/kernel/` together with the URMA AIV
Direct Get entry point. Its installed object name remains
`kvaf_execute_batch_get_kernels.o`.
