"""Source data and request plans shared by BatchGet benchmarks."""


def token_pattern(torch, token_ids, byte_ids):
    # Encode a source token's identity across eight bytes. Full-shaped operands
    # also work with torch_npu versions without broadcasted bitwise operations.
    shifts = ((byte_ids % 8) * 8).expand(token_ids.numel(), -1).contiguous()
    ids = token_ids.unsqueeze(1).expand_as(shifts).contiguous()
    encoded = ((ids >> shifts) & 0xFF).to(torch.uint8)
    pattern = ((byte_ids * 17 + (byte_ids >> 3)) & 0xFF).to(torch.uint8)
    return encoded ^ pattern.expand_as(encoded).contiguous()


def sort_request_plan_prefix(
    torch: object,
    address_table: object,
    indices: object,
    offsets: object,
    output_indices: object,
    active_count: int,
) -> tuple[object, object, object]:
    """Sort active requests by resolved table entry and offset."""
    if active_count <= 1:
        return indices, offsets, output_indices

    active_indices = indices[:active_count]
    active_offsets = offsets[:active_count]
    active_output_indices = output_indices[:active_count]
    entry_addresses = address_table.index_select(
        0, active_indices.to(torch.int64)
    )
    # Every valid table entry owns one non-overlapping, fixed-size block, so
    # sorting the resolved remote byte address is equivalent to sorting first
    # by the actual slot ID and then by the offset inside that slot.
    sort_keys = entry_addresses + active_offsets.to(torch.int64)
    _, permutation = torch.sort(sort_keys)
    sorted_indices = active_indices.index_select(0, permutation)
    sorted_offsets = active_offsets.index_select(0, permutation)
    sorted_output_indices = active_output_indices.index_select(0, permutation)
    indices[:active_count].copy_(sorted_indices)
    offsets[:active_count].copy_(sorted_offsets)
    output_indices[:active_count].copy_(sorted_output_indices)
    return indices, offsets, output_indices


def generate_request_plan(torch, args, device, request_ids, address_table,
                          contiguous_source_token_ids=None):
    """Shared steady-state source/output distribution for transport comparisons."""
    if args.source_access_pattern == "random":
        indices = torch.randint(0, args.table_capacity, (args.request_length,),
                                dtype=torch.int32, device=device)
        token_slots = torch.randint(0, args.tokens_per_block, (args.request_length,),
                                    dtype=torch.int32, device=device)
    else:
        source_token_ids = (request_ids % (args.table_capacity * args.tokens_per_block)
                            if contiguous_source_token_ids is None else contiguous_source_token_ids)
        if args.source_access_pattern == "shuffled-contiguous":
            original = source_token_ids
            source_token_ids = source_token_ids.clone()
            permutation = torch.randperm(args.miss_count, device=device)
            source_token_ids[:args.miss_count].copy_(
                original[:args.miss_count].index_select(0, permutation))
        indices = (source_token_ids // args.tokens_per_block).to(torch.int32)
        token_slots = (source_token_ids % args.tokens_per_block).to(torch.int32)
    offsets = token_slots * args.token_bytes
    output_indices = (request_ids.to(torch.int32) if args.output_access_pattern == "contiguous"
                      else torch.randperm(args.request_length, device=device).to(torch.int32))
    if args.sort_input_indices:
        indices, offsets, output_indices = sort_request_plan_prefix(
            torch, address_table, indices, offsets, output_indices, args.miss_count)
    return indices, offsets, output_indices


def refresh_request_buffers(plan, fixed_inputs):
    for destination, source in zip(fixed_inputs, plan):
        destination.copy_(source)
