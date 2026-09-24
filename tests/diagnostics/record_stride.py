"""Source adaptation for padded remote records; transport payload remains 656B."""
def apply(source, stride):
    if stride not in (656, 704, 768, 1024):
        raise ValueError('stride must be 656,704,768,1024')
    if stride == 656:
        return source
    replacements = [
        ('    block_bytes = token_bytes * tokens_per_block\n',
         f'    record_stride = {stride}\n'
         '    if token_bytes != 656 or tokens_per_block != 128:\n'
         '        raise ValueError("Padded test requires 656 bytes and 128 logical tokens")\n'
         '    physical_tokens = (record_stride*tokens_per_block + token_bytes-1)//token_bytes\n'
         '    physical_tokens = ((physical_tokens+3)//4)*4 # block starts 64B aligned\n'
         '    block_bytes = physical_tokens * token_bytes\n'),
        ('        batch_get_tokens_per_block=tokens_per_block,',
         '        batch_get_tokens_per_block=physical_tokens,'),
        ('        populate_source = entry_pattern.reshape(1, block_bytes).repeat(populate_entries, 1)',
         '        padded = torch.zeros(block_bytes, dtype=torch.uint8, device=device)\n'
         '        for token in range(tokens_per_block):\n'
         '            padded[token*record_stride:token*record_stride+token_bytes] = entry_pattern[token]\n'
         '        populate_source = padded.reshape(1, block_bytes).repeat(populate_entries, 1)'),
        ('preflight_token_slots * token_bytes', 'preflight_token_slots * record_stride'),
        ('offsets = token_slots * token_bytes', 'offsets = token_slots * record_stride'),
    ]
    for old,new in replacements:
        # block_bytes also occurs in argument validation; apply to both sites.
        if old not in source:
            raise RuntimeError('Unsupported stride benchmark layout: '+old)
        source = source.replace(old,new)
    return source
