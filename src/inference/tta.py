"""
TALARIA Test-Time Augmentation (TTA).

Augmentations applied at inference time:
    - Identity (no transform)
    - Flip along D, H, W axes
    - 90-degree rotations in axial plane

Each augmentation produces a set of predictions; they are averaged
via soft voting in soft_voting.py.
"""

import torch
from typing import List, Callable, Tuple, Dict


class TTTAdaptor:
    """Test-time training adaptor for unlabeled inference data."""

    def __init__(
        self,
        model: torch.nn.Module,
        steps: int = 1,
        lr: float = 1e-5,
        adapt_modules: List[str] = None,
        objective: str = 'entropy',
        reset_each_volume: bool = True,
        use_amp: bool = False,
        device: torch.device = None,
    ):
        self.model = model
        self.steps = max(0, steps)
        self.lr = lr
        self.adapt_modules = adapt_modules or ['bn', 'head']
        self.objective = objective
        self.reset_each_volume = reset_each_volume
        self.use_amp = use_amp
        self.device = device or torch.device('cpu')
        self._initial_state = None

        self.model.to(self.device)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp and self.device.type == 'cuda')

    def _is_adapt_param(self, name: str) -> bool:
        lower_name = name.lower()
        for module_key in self.adapt_modules:
            key = module_key.strip().lower()
            if key and key in lower_name:
                return True
        return False

    def _configure_trainable_params(self) -> List[torch.nn.Parameter]:
        params = []
        for name, param in self.model.named_parameters():
            should_adapt = self._is_adapt_param(name)
            param.requires_grad = should_adapt
            if should_adapt:
                params.append(param)
        if not params:
            raise ValueError(f'No trainable params selected for TTT modules: {self.adapt_modules}')
        return params

    def _entropy_objective(self, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        t_seg_prob = out['t_seg'].sigmoid().clamp(min=1e-6, max=1 - 1e-6)
        n_seg_prob = out['n_seg'].sigmoid().clamp(min=1e-6, max=1 - 1e-6)
        t_seg_ent = -(t_seg_prob * t_seg_prob.log() + (1 - t_seg_prob) * (1 - t_seg_prob).log()).mean()
        n_seg_ent = -(n_seg_prob * n_seg_prob.log() + (1 - n_seg_prob) * (1 - n_seg_prob).log()).mean()

        t_cls_prob = torch.softmax(out['t_cls'], dim=-1).clamp(min=1e-6, max=1 - 1e-6)
        n_cls_prob = torch.softmax(out['n_cls'], dim=-1).clamp(min=1e-6, max=1 - 1e-6)
        t_cls_ent = -(t_cls_prob * t_cls_prob.log()).sum(dim=-1).mean()
        n_cls_ent = -(n_cls_prob * n_cls_prob.log()).sum(dim=-1).mean()

        return t_seg_ent + n_seg_ent + t_cls_ent + n_cls_ent

    def _compute_loss(self, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.objective == 'entropy':
            return self._entropy_objective(out)
        raise ValueError(f'Unsupported TTT objective: {self.objective}')

    def _adapt_batch(self, optimizer: torch.optim.Optimizer, batch: torch.Tensor, prefix: str = 'volume') -> None:
        batch = batch.to(self.device)
        for step in range(self.steps):
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.use_amp and self.device.type == 'cuda'):
                out = self.model(batch, apply_manifold_mixup=False)
                loss = self._compute_loss(out)

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()

            print(f'[TTT] {prefix} step {step + 1}/{self.steps} loss={loss.item():.6f}')

    def adapt_volume(self, patches: List[torch.Tensor], scope: str = 'volume') -> None:
        if self.steps <= 0:
            return

        params = self._configure_trainable_params()
        optimizer = torch.optim.Adam(params, lr=self.lr)

        if self._initial_state is None:
            self._initial_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        elif self.reset_each_volume:
            self.model.load_state_dict(self._initial_state, strict=True)

        self.model.train()
        if scope == 'patch':
            for patch_idx, patch in enumerate(patches):
                self._adapt_batch(optimizer, patch, prefix=f'patch {patch_idx + 1}/{len(patches)}')
        else:
            for patch in patches:
                patch.requires_grad_(False)
            full_batch = torch.cat([patch for patch in patches], dim=0)
            self._adapt_batch(optimizer, full_batch, prefix='volume')

        self.model.eval()


# ---------------------------------------------------------------------------
# Augmentation functions
# ---------------------------------------------------------------------------

def identity(x: torch.Tensor) -> torch.Tensor:
    return x


def flip_d(x: torch.Tensor) -> torch.Tensor:
    return x.flip(2)


def flip_h(x: torch.Tensor) -> torch.Tensor:
    return x.flip(3)


def flip_w(x: torch.Tensor) -> torch.Tensor:
    return x.flip(4)


def flip_dh(x: torch.Tensor) -> torch.Tensor:
    return x.flip(2).flip(3)


def flip_dhw(x: torch.Tensor) -> torch.Tensor:
    return x.flip(2).flip(3).flip(4)


def rot90_hw(x: torch.Tensor) -> torch.Tensor:
    """90-degree rotation in the H-W (axial) plane."""
    return x.rot90(1, dims=[3, 4])


def rot180_hw(x: torch.Tensor) -> torch.Tensor:
    return x.rot90(2, dims=[3, 4])


def rot270_hw(x: torch.Tensor) -> torch.Tensor:
    return x.rot90(3, dims=[3, 4])


# Inverse functions (to de-augment predictions)
INVERSE = {
    identity:  identity,
    flip_d:    flip_d,
    flip_h:    flip_h,
    flip_w:    flip_w,
    flip_dh:   flip_dh,
    flip_dhw:  flip_dhw,
    rot90_hw:  rot270_hw,
    rot180_hw: rot180_hw,
    rot270_hw: rot90_hw,
}

# Default TTA set (6 flips + 3 rotations + identity = 10 transforms)
DEFAULT_TTA_TRANSFORMS = [
    identity,
    flip_d,
    flip_h,
    flip_w,
    flip_dh,
    flip_dhw,
    rot90_hw,
    rot180_hw,
    rot270_hw,
]


# ---------------------------------------------------------------------------
# TTA Predictor
# ---------------------------------------------------------------------------

class TTAPredictor:
    """
    Applies a set of test-time augmentations to a single patch,
    collects model outputs, and inverts augmentations on segmentation masks.

    Args:
        model:      TALARIANet in finetune mode
        transforms: list of augmentation functions to apply
        device:     torch device
    """

    def __init__(
        self,
        model: torch.nn.Module,
        transforms: List[Callable] = None,
        device: torch.device = None,
    ):
        self.model = model
        self.transforms = transforms or DEFAULT_TTA_TRANSFORMS
        self.device = device or torch.device('cpu')
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict_patch(self, patch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run TTA on a single 3D patch.

        Args:
            patch: (1, 1, D, H, W) — single patch tensor
        Returns:
            dict with averaged predictions:
                't_seg':  (1, 1, D, H, W) averaged tumor segmentation probability
                'n_seg':  (1, 1, D, H, W) averaged lymph node segmentation probability
                't_cls':  (1, 4) averaged T-stage logits
                'n_cls':  (1, 2) averaged N-stage logits
        """
        patch = patch.to(self.device)
        t_seg_acc = torch.zeros_like(patch)
        n_seg_acc = torch.zeros_like(patch)
        t_cls_acc = None
        n_cls_acc = None

        for aug in self.transforms:
            aug_patch = aug(patch)
            out = self.model(aug_patch, apply_manifold_mixup=False)

            # Invert spatial transforms on segmentation predictions
            inv = INVERSE[aug]
            t_seg_acc += inv(out['t_seg'].sigmoid())
            n_seg_acc += inv(out['n_seg'].sigmoid())

            if t_cls_acc is None:
                t_cls_acc = out['t_cls']
                n_cls_acc = out['n_cls']
            else:
                t_cls_acc = t_cls_acc + out['t_cls']
                n_cls_acc = n_cls_acc + out['n_cls']

        n = len(self.transforms)
        return {
            't_seg': t_seg_acc / n,
            'n_seg': n_seg_acc / n,
            't_cls': t_cls_acc / n,
            'n_cls': n_cls_acc / n,
        }

    @torch.no_grad()
    def predict_volume(
        self,
        patches: List[torch.Tensor],
        coords: List[Tuple[int, int, int]],
        volume_shape: Tuple[int, int, int],
        patch_size: int = 96,
    ) -> Dict[str, torch.Tensor]:
        """
        Run TTA over all patches of a volume and stitch results.

        Args:
            patches:      list of (1, 1, P, P, P) tensors
            coords:       list of (d, h, w) top-left coords
            volume_shape: (D, H, W) full volume shape
            patch_size:   P
        Returns:
            dict with full-resolution outputs
        """
        D, H, W = volume_shape
        P = patch_size
        t_seg_vol = torch.zeros(1, 1, D, H, W)
        n_seg_vol = torch.zeros(1, 1, D, H, W)
        weight_vol = torch.zeros(1, 1, D, H, W)
        t_cls_list = []
        n_cls_list = []

        for patch, (d, h, w) in zip(patches, coords):
            preds = self.predict_patch(patch)
            preds_cpu = {k: v.detach().cpu() for k, v in preds.items()}
            d_end = min(d + P, D)
            h_end = min(h + P, H)
            w_end = min(w + P, W)
            t_seg_vol[..., d:d_end, h:h_end, w:w_end] += preds_cpu['t_seg'][..., :d_end-d, :h_end-h, :w_end-w]
            n_seg_vol[..., d:d_end, h:h_end, w:w_end] += preds_cpu['n_seg'][..., :d_end-d, :h_end-h, :w_end-w]
            weight_vol[..., d:d_end, h:h_end, w:w_end] += 1.0
            t_cls_list.append(preds_cpu['t_cls'])
            n_cls_list.append(preds_cpu['n_cls'])

        weight_vol = weight_vol.clamp(min=1e-6)
        t_seg_vol /= weight_vol
        n_seg_vol /= weight_vol

        # Average classification logits across patches
        t_cls_avg = torch.stack(t_cls_list).mean(0)
        n_cls_avg = torch.stack(n_cls_list).mean(0)

        return {
            't_seg': t_seg_vol,
            'n_seg': n_seg_vol,
            't_cls': t_cls_avg,
            'n_cls': n_cls_avg,
        }
