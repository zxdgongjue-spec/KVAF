"""Deterministic source data for BatchGet benchmark validation."""


def token_pattern(torch, token_ids, byte_ids):
    # Encode a source token's identity across eight bytes. Full-shaped operands
    # also work with torch_npu versions without broadcasted bitwise operations.
    shifts = ((byte_ids % 8) * 8).expand(token_ids.numel(), -1).contiguous()
    ids = token_ids.unsqueeze(1).expand_as(shifts).contiguous()
    encoded = ((ids >> shifts) & 0xFF).to(torch.uint8)
    pattern = ((byte_ids * 17 + (byte_ids >> 3)) & 0xFF).to(torch.uint8)
    return encoded ^ pattern.expand_as(encoded).contiguous()
