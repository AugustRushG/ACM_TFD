import random
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision import transforms
from .variables import IMAGENET_MEAN, IMAGENET_STD
from typing import List

def _teacher_transform():
    """Weak view for the teacher: normalisation only (cropping is done by the frame reader)."""
    return transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_ssl_transform(crop_dim, use_randaugment=False, ra_num_ops=2, ra_magnitude=9, use_temporal_aware_aug=True):
    """Transforms for the unlabeled teacher/student views used by ssl_train.py.

    Returns (crop_transform, weak, strong). `weak` is applied to the teacher clip, `strong` to the
    student clip. With `use_temporal_aware_aug`, the student view additionally gets
    MotionGridMosaicRGB (shuffles high-motion patches) and RandomConsecutiveFrameMask
    (zeroes short runs of consecutive frames).
    """
    crop_transform = transforms.CenterCrop(crop_dim)
    weak = _teacher_transform()

    strong_transforms_list = []
    if use_randaugment:
        strong_transforms_list.append(transforms.ConvertImageDtype(torch.uint8))
        strong_transforms_list.append(
            transforms.RandAugment(num_ops=ra_num_ops, magnitude=ra_magnitude)
        )
    else:
        if use_temporal_aware_aug:
            strong_transforms_list.append(MotionGridMosaicRGB(
                grid=16,
                topk=9,
                p_frame=0.5,
                top_pixels_frac=0.02,
            ))
        strong_transforms_list.extend([
            # Geometric
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation(degrees=15)], p=0.5),
            # Colour / texture
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.5),
            transforms.RandomSolarize(threshold=0.5, p=0.2),
        ])
        if use_temporal_aware_aug:
            strong_transforms_list.append(RandomConsecutiveFrameMask(
                mask_len=3,
                num_masks=5,
                mask_value="zero",
                p=1.0,
            ))

    strong_transforms_list.append(transforms.ConvertImageDtype(torch.float32))
    strong_transforms_list.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    strong = transforms.Compose(strong_transforms_list)

    return crop_transform, weak, strong


def get_wacv_transform(crop_dim, use_randaugment=True, ra_num_ops=2, ra_magnitude=9):
    """Lighter student augmentation (enabled with --tma_augment): flip + colour jitter + blur."""
    crop_transform = transforms.CenterCrop(crop_dim)
    weak = _teacher_transform()

    strong_transforms_list = []
    if use_randaugment:
        strong_transforms_list.append(transforms.ConvertImageDtype(torch.uint8))
        strong_transforms_list.append(
            transforms.RandAugment(num_ops=ra_num_ops, magnitude=ra_magnitude)
        )
    else:
        strong_transforms_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.5),
        ])

    strong_transforms_list.extend([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    strong = transforms.Compose(strong_transforms_list)

    return crop_transform, weak, strong


def _get_img_transforms(
        is_eval,
        crop_dim,
        modality,
        same_transform,
        defer_transform=False,
        multi_crop=False
):
    crop_transform = None
    if crop_dim is not None:
        if multi_crop:
            assert is_eval
            crop_transform = ThreeCrop(crop_dim)
        elif is_eval:
            crop_transform = transforms.CenterCrop(crop_dim)
            # print('=>Using Resize instead')
            # crop_transform = transforms.Resize((crop_dim, crop_dim))
        elif same_transform:
            print('=> Using seeded crops!')
            crop_transform = SeedableRandomSquareCrop(crop_dim)
        elif crop_dim == -1:
            crop_transform = None
        else:
            crop_transform = transforms.RandomCrop(crop_dim)
    else:
        print('=> No crop: resizing frames to 224x398 (H x W)')
        crop_transform = transforms.Resize((224, 398))

    img_transforms = []
    if modality == 'rgb':
        if not is_eval:
            img_transforms.append(
                transforms.RandomHorizontalFlip())

            if not defer_transform:
                img_transforms.extend([
                    # Jittering separately is faster (low variance)
                    transforms.RandomApply(
                        nn.ModuleList([transforms.ColorJitter(hue=0.2)]),
                        p=0.25),
                    transforms.RandomApply(
                        nn.ModuleList([
                            transforms.ColorJitter(saturation=(0.7, 1.2))
                        ]), p=0.25),
                    transforms.RandomApply(
                        nn.ModuleList([
                            transforms.ColorJitter(brightness=(0.7, 1.2))
                        ]), p=0.25),
                    transforms.RandomApply(
                        nn.ModuleList([
                            transforms.ColorJitter(contrast=(0.7, 1.2))
                        ]), p=0.25),
                    transforms.RandomApply(
                        nn.ModuleList([transforms.GaussianBlur(5)]), p=0.25),
                    # transforms.RandomApply(
                    #     nn.ModuleList([transforms.RandomGridPatchMask()]), p=0.25)
                ])

        if not defer_transform:
            img_transforms.append(transforms.Normalize(
                mean=IMAGENET_MEAN, std=IMAGENET_STD))
    elif modality == 'bw':
        if not is_eval:
            img_transforms.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    nn.ModuleList([transforms.ColorJitter(hue=0.2)]), p=0.25)])
        img_transforms.append(transforms.Grayscale())

        if not defer_transform:
            if not is_eval:
                img_transforms.extend([
                    transforms.RandomApply(
                        nn.ModuleList([transforms.ColorJitter(brightness=0.3)]),
                        p=0.25),
                    transforms.RandomApply(
                        nn.ModuleList([transforms.ColorJitter(contrast=0.3)]),
                        p=0.25),
                    transforms.RandomApply(
                        nn.ModuleList([transforms.GaussianBlur(5)]), p=0.25),
                ])

            img_transforms.append(transforms.Normalize(
                mean=[0.5], std=[0.5]))

            if not is_eval:
                img_transforms.append(RandomGaussianNoise())
    elif modality == 'flow':
        assert not defer_transform

        img_transforms.append(transforms.Normalize(
            mean=[0.5, 0.5], std=[0.5, 0.5]))

        if not is_eval:
            img_transforms.extend([
                RandomHorizontalFlipFLow(),
                RandomOffsetFlow(),
                RandomGaussianNoise()
            ])
    else:
        raise NotImplementedError(modality)

    img_transform = torch.jit.script(nn.Sequential(*img_transforms))
    return crop_transform, img_transform

def _get_deferred_rgb_transform():
    img_transforms = [
        # Jittering separately is faster (low variance)
        transforms.RandomApply(
            nn.ModuleList([transforms.ColorJitter(hue=0.2)]), p=0.25),
        transforms.RandomApply(
            nn.ModuleList([
                transforms.ColorJitter(saturation=(0.7, 1.2))
            ]), p=0.25),
        transforms.RandomApply(
            nn.ModuleList([
                transforms.ColorJitter(brightness=(0.7, 1.2))
            ]), p=0.25),
        transforms.RandomApply(
            nn.ModuleList([
                transforms.ColorJitter(contrast=(0.7, 1.2))
            ]), p=0.25),

        transforms.RandomApply(
            nn.ModuleList([transforms.GaussianBlur(5)]), p=0.25),
        # add in 
        transforms.RandomApply(
            nn.ModuleList([RandomTubeGridPatchMask(grid_size=(16,16), min_patches=30, max_patches=100, fill_value=0.0)]), p=0.25),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]
    return torch.jit.script(nn.Sequential(*img_transforms))

def _get_deferred_bw_transform():
    img_transforms = [
        transforms.RandomApply(
            nn.ModuleList([transforms.ColorJitter(brightness=0.3)]), p=0.25),
        transforms.RandomApply(
            nn.ModuleList([transforms.ColorJitter(contrast=0.3)]), p=0.25),
        transforms.RandomApply(
            nn.ModuleList([transforms.GaussianBlur(5)]), p=0.25),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        RandomGaussianNoise()
    ]
    return torch.jit.script(nn.Sequential(*img_transforms))


class RandomHorizontalFlipFLow(nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        if torch.rand(1)[0] < self.p:
            shape = img.shape
            img.view((-1,) + shape[-3:])[:, 1, :, :] *= -1
            return img.flip(-1)
        return img


class RandomOffsetFlow(nn.Module):

    def __init__(self, p=0.5, x=0.1, y=0.05):
        super().__init__()
        self.p = p
        self.x = x
        self.y = y

    def forward(self, img):
        if torch.rand(1)[0] < self.p:
            shape = img.shape
            view = img.view((-1,) + shape[-3:])
            view[:, 1, :, :] += (
                torch.rand(1, device=img.device)[0] * 2 - 1) * self.x
            view[:, 0, :, :] += (
                torch.rand(1, device=img.device)[0] * 2 - 1) * self.y
        return img


class RandomGaussianNoise(nn.Module):

    def __init__(self, p=0.5, s=0.1):
        super().__init__()
        self.p = p
        self.std = s ** 0.5

    def forward(self, img):
        v = torch.rand(1)[0]
        if v < self.p:
            img += torch.randn(img.shape, device=img.device) * self.std
        return img


class SeedableRandomSquareCrop:

    def __init__(self, dim):
        self._dim = dim

    def __call__(self, img):
        c, h, w = img.shape[-3:]
        x, y = 0, 0
        if h > self._dim:
            y = random.randint(0, h - self._dim)
        if w > self._dim:
            x = random.randint(0, w - self._dim)
        return F.crop(img, y, x, self._dim, self._dim)


class ThreeCrop:

    def __init__(self, dim):
        self._dim = dim

    def __call__(self, img):
        c, h, w = img.shape[-3:]
        y = (h - self._dim) // 2
        ret = []
        dw = w - self._dim
        for x in (0, dw // 2, dw):
            ret.append(F.crop(img, y, x, self._dim, self._dim))
        return torch.stack(ret)


class RandomTubeGridPatchMask(nn.Module):
    def __init__(self, grid_size=(3, 3), min_patches=1, max_patches=None, fill_value=0.0):
        super().__init__()
        self.grid_rows, self.grid_cols = grid_size
        self.min_patches = min_patches
        self.max_patches = max_patches if max_patches is not None else (self.grid_rows * self.grid_cols)
        self.fill_value = fill_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor. 
               Supported shapes:
               - [B, T, C, H, W] (Batch of videos)
               - [T, C, H, W]    (Single video)
        Returns:
            Masked tensor with same shape.
        """
        if not self.training:
            return x

        # Handle Batch of Videos [B, T, C, H, W]
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            out = x.clone()
            for b in range(B):
                # Pass the single video [T, C, H, W] to the mask logic
                self._apply_tube_mask_single(out[b], H, W)
            return out
        
        # Handle Single Video [T, C, H, W]
        elif x.dim() == 4:
            T, C, H, W = x.shape
            out = x.clone()
            self._apply_tube_mask_single(out, H, W)
            return out
            
        else:
            raise ValueError(f"Expected 4D [T,C,H,W] or 5D [B,T,C,H,W] tensor, got {x.shape}")

    def _apply_tube_mask_single(self, out_video: torch.Tensor, H: int, W: int):
        """
        Applies spatial masks that extend through the entire temporal dimension (Tube).
        out_video shape: [T, C, H, W]  <-- Note the order
        """
        
        # 1. Compute Grid Edges
        # We compute these based on H and W (Spatial dims are always the last two)
        y_edges: List[int] = torch.linspace(0, H, steps=self.grid_rows + 1).round().long().tolist()
        x_edges: List[int] = torch.linspace(0, W, steps=self.grid_cols + 1).round().long().tolist()

        # 2. Randomly select number of patches 'k'
        total = self.grid_rows * self.grid_cols
        limit_max = min(self.max_patches, total)
        
        if self.min_patches >= limit_max:
            k = limit_max
        else:
            k = int(torch.randint(self.min_patches, limit_max + 1, (1,)).item())

        # 3. Select unique random patches
        # We select ONE set of indices for the entire video clip
        perm = torch.randperm(total, device=out_video.device)
        indices = perm[:k]

        # 4. Mask the pixels (Tube Masking)
        for i in range(k):
            idx = int(indices[i].item())
            r = idx // self.grid_cols
            c = idx % self.grid_cols
            
            y1, y2 = y_edges[r], y_edges[r + 1]
            x1, x2 = x_edges[c], x_edges[c + 1]
            
            if y2 > y1 and x2 > x1:
                # [:, :, y1:y2, x1:x2] selects:
                # - All Time Frames (1st dim)
                # - All Channels (2nd dim)
                # - Specific Height slice
                # - Specific Width slice
                # This effectively creates the "Tube" masking effect
                out_video[:, :, y1:y2, x1:x2] = self.fill_value


class RandomConsecutiveFrameMask(nn.Module):
    """
    Randomly mask out consecutive frames in a clip.

    Input:  clip [T, C, H, W]
    Output: clip [T, C, H, W]

    Parameters:
        mask_len:   number of consecutive frames per masked segment
        num_masks:  number of masked segments
        mask_value: "zero" | "noise" | "repeat"
        p:          probability to apply transform
    """

    def __init__(
        self,
        mask_len: int = 3,
        num_masks: int = 1,
        mask_value: str = "zero",
        p: float = 1.0,
    ):
        super().__init__()
        if mask_len <= 0:
            raise ValueError("mask_len must be > 0")
        if num_masks <= 0:
            raise ValueError("num_masks must be > 0")
        if mask_value not in ("zero", "noise", "repeat"):
            raise ValueError("mask_value must be 'zero', 'noise', or 'repeat'")
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1]")

        self.mask_len = mask_len
        self.num_masks = num_masks
        self.mask_value = mask_value
        self.p = p

    def _sample_starts(self, T: int) -> list[int]:
        """
        Sample non-overlapping segment start indices.
        Divides T into num_masks equal bins and samples once per bin,
        guaranteeing non-overlap without retry loops.
        """
        max_start = T - self.mask_len
        if max_start <= 0:
            return []

        # bin-based sampling: split valid range into num_masks bins
        bin_size = max(1, max_start // self.num_masks)
        starts = []
        for i in range(self.num_masks):
            lo = i * bin_size
            hi = min(lo + bin_size, max_start + 1)
            if lo >= hi:
                break
            starts.append(random.randint(lo, hi - 1))

        return starts

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return clip

        T, C, H, W = clip.shape
        starts = self._sample_starts(T)
        if not starts:
            return clip

        frame_mask = torch.zeros(T, dtype=torch.bool, device=clip.device)
        for s in starts:
            frame_mask[s: s + self.mask_len] = True

        # ── No clone — build output directly ─────────────────────────────────
        if self.mask_value == "zero":
            # zero out in one masked op, no full copy needed
            out = clip * (~frame_mask).view(T, 1, 1, 1).float()

        elif self.mask_value == "noise":
            n_masked = int(frame_mask.sum().item())
            noise = torch.randn(n_masked, C, H, W,
                            device=clip.device, dtype=clip.dtype)
            out = clip * (~frame_mask).view(T, 1, 1, 1).float()
            out[frame_mask] = noise

        elif self.mask_value == "repeat":
            out = clip.clone()  # unavoidable for repeat mode only
            for s in starts:
                ref = max(s - 1, 0) if s > 0 else min(s + self.mask_len, T - 1)
                out[s: s + self.mask_len] = clip[ref]

        return out

class MotionGridMosaicRGB(nn.Module):
    """
    Vectorized motion grid selection (ball-safe).
    All frame and patch operations are batched — no Python loops over T or patches.

    Input:  clip [T, C, 224, 224]
    Output: clip [T, C, 224, 224]
    """

    def __init__(
        self,
        grid: int = 3,
        topk: int = 4,
        out_hw: int = 224,
        p_frame: float = 1.0,
        threshold: float = 0.0,
        mask_height: float = 0.0,
        score_reduce: str = "topk_mean",
        top_pixels_frac: float = 0.02,
        dilate: int = 1,
        fill_mode: str = "soft",
        background_scale: float = 0.25,
        seed: int | None = None,
    ):
        super().__init__()
        self.grid = int(grid)
        self.topk = int(topk)
        self.out_hw = int(out_hw)
        self.p_frame = float(p_frame)
        self.threshold = float(threshold)
        self.mask_height = float(mask_height)
        self.score_reduce = score_reduce
        self.top_pixels_frac = float(top_pixels_frac)
        self.dilate = int(dilate)
        self.fill_mode = fill_mode
        self.background_scale = float(background_scale)

        # validation (unchanged)
        if self.grid <= 0:
            raise ValueError("grid must be > 0")
        if self.topk <= 0 or self.topk > self.grid * self.grid:
            raise ValueError(f"topk must be in [1, {self.grid*self.grid}]")
        if self.score_reduce not in ("mean", "sum", "topk_mean"):
            raise ValueError("score_reduce must be 'mean', 'sum', or 'topk_mean'")
        if self.fill_mode not in ("black", "soft", "orig"):
            raise ValueError("fill_mode must be 'black', 'soft', or 'orig'")

        self._rng = torch.Generator() if seed is not None else None
        if seed is not None:
            self._rng.manual_seed(seed)

    def _compute_scores(self, motion_patches: torch.Tensor) -> torch.Tensor:
        """
        motion_patches: [T, g*g, ph*pw]
        returns scores:  [T, g*g]
        """
        if self.score_reduce == "mean":
            return motion_patches.mean(dim=-1)
        if self.score_reduce == "sum":
            return motion_patches.sum(dim=-1)
        # topk_mean — best for small fast objects like balls
        k = max(1, int(self.top_pixels_frac * motion_patches.shape[-1]))
        return torch.topk(motion_patches, k=k, dim=-1).values.mean(dim=-1)

    def _dilate_mask(self, topk_mask: torch.Tensor) -> torch.Tensor:
        """
        topk_mask: [T, g, g] bool
        Dilate selected patches to include neighbors.
        Uses 2D max-pool as a fast dilation kernel — no Python loops.
        """
        if self.dilate == 0:
            return topk_mask

        # treat as [T, 1, g, g] float for max_pool2d
        x = topk_mask.unsqueeze(1).float()
        kernel = 2 * self.dilate + 1
        dilated = torch.nn.functional.max_pool2d(
            x,
            kernel_size=kernel,
            stride=1,
            padding=self.dilate,
        )
        return dilated.squeeze(1).bool()  # [T, g, g]

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        if clip.dim() != 4:
            raise ValueError(f"Expected clip [T,C,H,W], got {tuple(clip.shape)}")

        T, C, H, W = clip.shape
        g = self.grid
        ph, pw = H // g, W // g

        if ph * g != H or pw * g != W:
            raise ValueError(f"Frame {H}x{W} must be divisible by grid={g}")

        # ── 1. Frame selection mask [T] ───────────────────────────────────────
        if self.p_frame < 1.0:
            frame_mask = (torch.rand(T, device=clip.device) < self.p_frame)
            if not frame_mask.any():
                return clip  # early exit, nothing to do

        # ── 2. Motion maps [T, H, W] — no clone, use roll instead of cat ─────
        prev = torch.roll(clip, shifts=1, dims=0)  # avoids cat allocation
        motion = torch.abs(clip - prev).amax(dim=1)  # [T, H, W]
        motion[0] = 0.0

        if self.threshold > 0:
            motion.mul_(motion >= self.threshold)  # in-place

        if self.mask_height > 0:
            cut = int(H * self.mask_height)
            if cut > 0:
                motion[:, :cut, :] = 0.0

        # ── 3. Patch scores [T, g*g] ──────────────────────────────────────────
        motion_patches = (
            motion
            .reshape(T, g, ph, g, pw)
            .permute(0, 1, 3, 2, 4)
            .reshape(T, g * g, ph * pw)
        )
        scores = self._compute_scores(motion_patches)

        # ── 4. Top-k + dilation ───────────────────────────────────────────────
        topk_idx = torch.topk(scores, k=self.topk, dim=-1).indices  # [T, k]
        topk_mask = torch.zeros(T, g * g, dtype=torch.bool, device=clip.device)
        topk_mask.scatter_(1, topk_idx, True)
        topk_mask = topk_mask.reshape(T, g, g)
        sel_mask = self._dilate_mask(topk_mask)  # [T, g, g]

        # ── 5. Pixel mask via nearest upsample — faster than repeat_interleave ─
        pixel_mask = torch.nn.functional.interpolate(
            sel_mask.unsqueeze(1).float(),
            size=(H, W),
            mode="nearest",
        )  # [T, 1, H, W] — single GPU kernel, no intermediate tensors

        # ── 6. Blend — no clone, fused where op ──────────────────────────────
        if self.fill_mode == "black":
            out = clip * pixel_mask
        elif self.fill_mode == "orig":
            return clip  # keep-in-place with orig is a no-op
        else:  # soft
            out = clip * (self.background_scale + (1 - self.background_scale) * pixel_mask)

        # ── 7. Frame mask — skip if p_frame == 1.0 ───────────────────────────
        if self.p_frame < 1.0:
            out = torch.where(frame_mask.view(T, 1, 1, 1), out, clip)

        return out
