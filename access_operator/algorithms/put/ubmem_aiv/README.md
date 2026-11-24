# UBMEM AIV Put

`kernel/execute_batch_put_ubmem.asc` copies caller input from HBM into the
mapped CPU DDR allocation using the destinations prepared by Runtime.

The kernel is compiled by `../common/aiv_kernel/` together with the URMA AIV
Put entry point. Its installed object name remains
`kvaf_execute_batch_put_kernels.o`. Mapping and execution resources are
supplied by the UBMEM provider.
