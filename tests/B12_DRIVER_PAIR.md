# B12 local-host driver pair check

`KVAF_B12_USE_DRIVER_PAIR=1` selects the Host/NPU EIDs reported by
`/proc/asdrv_ub/pair_info` for the active physical NPU. Default `0` preserves
V9's candidate enumeration. This switch is for same-host B12 reproduction;
it must not be enabled when the peer service lives on another host.

CPU candidate EIDs are obtained from temporary 4096-byte allocation replies.
Every successful temporary allocation is released, including the matched
candidate, before NPU queues are constructed. No match or failed release
stops initialization. Probe and full benchmark use this same route.

Returned NPU2 evidence: the CLOS-first pair failed one WQE with CQ 2/2 using
both the RA token and the ACL process token. `non_pin=0` failed registration
with return 528101. These results do not prove the driver pair works. The
next test retains V9 RA token and `non_pin=1`, changing only EID selection.

Host regression test: `kvaf_peer_candidate_selection_host`. Real CQ and Get
bandwidth still require NPU validation.
