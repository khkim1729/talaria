import torch


def rotate_batch_3d(batch: torch.Tensor, label_type='rand'):
    device = batch.device
    if batch.ndim == 4:
        batch = batch.unsqueeze(0)

    B, C, D, H, W = batch.shape
    dims = (-2, -1)  # Axial Plane (H, W) 회전

    if label_type == 'rand':
        labels = torch.randint(0, 4, (B,), device=device)
        rotated_batch = batch
    elif label_type == 'expand':
        labels = torch.arange(4, device=device).repeat_interleave(B)
        rotated_batch = batch.repeat(4, 1, 1, 1, 1)
    else:
        labels = torch.full((B,), int(label_type), dtype=torch.long, device=device)
        rotated_batch = batch

    result = torch.empty_like(rotated_batch)

    for k in range(4):
        mask = (labels == k)
        if mask.any():
            if k == 0:
                result[mask] = rotated_batch[mask]
            else:
                # [안전장치] rot90 이후 메모리 불연속성 해결을 위해 .contiguous() 유지
                result[mask] = torch.rot90(rotated_batch[mask], k=k, dims=dims).contiguous()

    return result, labels