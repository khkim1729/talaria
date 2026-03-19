import torch

def tensor_rot_90_3d(x):
    # x shape: (B, C, D, H, W)
    # H와 W 축을 바꾼 뒤, W 축을 뒤집음 (Axial plane 기준 90도 회전)
    return x.transpose(-2, -1).flip(-1)

def tensor_rot_180_3d(x):
    # H와 W 축 모두 뒤집음
    return x.flip(-2, -1)

def tensor_rot_270_3d(x):
    # H와 W 축을 바꾼 뒤, H 축을 뒤집음
    return x.transpose(-2, -1).flip(-2)

def rotate_batch_3d(batch, label_type='rand'):
    """
    batch: 3D CT 볼륨 텐서 (B, C, D, H, W)
    """
    if label_type == 'rand':
        labels = torch.randint(4, (len(batch),), dtype=torch.long, device=batch.device)
    elif label_type == 'expand':
        # 하나의 입력에 대해 4가지 회전을 모두 생성 (Test-Time Training용)
        labels = torch.cat([
            torch.zeros(len(batch), dtype=torch.long),
            torch.zeros(len(batch), dtype=torch.long) + 1,
            torch.zeros(len(batch), dtype=torch.long) + 2,
            torch.zeros(len(batch), dtype=torch.long) + 3
        ]).to(batch.device)
        batch = batch.repeat((4, 1, 1, 1, 1))
    else:
        labels = torch.zeros((len(batch),), dtype=torch.long, device=batch.device) + label_type

    rotated_images = []
    for img, label in zip(batch, labels):
        img = img.unsqueeze(0) # (1, C, D, H, W)
        if label == 1:
            img = tensor_rot_90_3d(img)
        elif label == 2:
            img = tensor_rot_180_3d(img)
        elif label == 3:
            img = tensor_rot_270_3d(img)
        rotated_images.append(img)
        
    return torch.cat(rotated_images, dim=0), labels