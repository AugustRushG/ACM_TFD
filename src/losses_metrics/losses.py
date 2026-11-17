import torch
import torch.nn as nn
import torch.nn.functional as F

class CELoss(nn.Module):
    """
    Cross-Entropy Loss wrapper for classification tasks.

    This class provides a convenient interface around 
    `torch.nn.CrossEntropyLoss` for computing the cross-entropy 
    between predicted class logits and target class indices.

    It supports optional class weighting and can be easily 
    extended to handle sequence predictions (e.g., [B, T, C]).

    Args:
        weight (Tensor, optional): A manual rescaling weight given to each class.
            It must be a 1D Tensor assigning weight to each class.
            Default is ``None``.

    Shape:
        - **Inputs:** Tensor of shape `(B, C)` or `(B, T, C)` containing raw, unnormalized class logits.
        - **Targets:** Tensor of shape `(B,)` or `(B, T)` containing class indices in `[0, C-1]`.

    Returns:
        torch.Tensor: A scalar tensor representing the computed cross-entropy loss.

    Example::
        >>> criterion = CELoss()
        >>> inputs = torch.randn(4, 10)       # (batch=4, classes=10)
        >>> targets = torch.tensor([1, 0, 3, 2])
        >>> loss = criterion(inputs, targets)
        >>> print(loss.item())
    """
    def __init__(self, num_classes, fg_weight=None):
        super(CELoss, self).__init__()
        if fg_weight is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1] + [fg_weight] * (num_classes - 1)))


    def forward(self, inputs, targets):
        """
        Compute the cross-entropy loss between `inputs` and `targets`.

        Args:
            inputs (Tensor): Model predictions of shape `(B, C)` or `(B, T, C)`.
            targets (Tensor): Ground-truth class indices of shape `(B,)` or `(B, T)`.

        Returns:
            Tensor: Scalar tensor containing the mean cross-entropy loss.
        """
        return self.criterion(inputs, targets)


class TemporalAlignPushLoss(nn.Module):
    """
    Align backbone features between two views (e.g., teacher vs student) on the
    overlapping temporal region, and push away other videos in the batch.

    Designed for backbone features (not projection logits).
    - Alignment: cosine distance on matched frames (optionally on overlap region)
    - Push: off-diagonal similarity penalty between per-video pooled embeddings

    Inputs:
        view1_seq: [B, T, D]
        view2_seq: [B, T, D]

    overlap_number:
        - overlap_number == 0: identical time windows; align all frames
        - overlap_number > 0: view2 is shifted forward by `overlap_number` frames
          (same convention as your previous code)
          Example: view1 is global [0..T-1], view2 is global [shift..shift+T-1]
          Overlap length = T - shift, align:
              view1[:, shift+buf : T-buf] with view2[:, buf : (T-shift)-buf]

    Notes:
        - This loss aligns *backbone* features directly.
        - Use teacher_first only if you want to stop gradients on view1 (teacher).
          Typically: teacher_first=True (view1 teacher, view2 student).
    """

    def __init__(
        self,
        overlap_number: int = 0,
        buffer: int = 0,
        align_weight: float = 1.0,
        push_weight: float = 0.5,
        teacher_first: bool = True,
        pool: str = "mean",           # "mean" | "max"
        push_mode: str = "mse",       # "mse" | "hinge"
        hinge_margin: float = 0.2,    # only used when push_mode="hinge"
        eps: float = 1e-6,
    ):
        super().__init__()
        if overlap_number is None:
            raise ValueError("overlap_number must be an int (0 or positive shift).")
        if overlap_number < 0:
            raise ValueError("overlap_number must be >= 0.")
        if buffer < 0:
            raise ValueError("buffer must be >= 0.")
        if pool not in ("mean", "max"):
            raise ValueError("pool must be 'mean' or 'max'.")
        if push_mode not in ("mse", "hinge"):
            raise ValueError("push_mode must be 'mse' or 'hinge'.")

        self.overlap_number = int(overlap_number)
        self.buffer = int(buffer)
        self.align_weight = float(align_weight)
        self.push_weight = float(push_weight)
        self.teacher_first = bool(teacher_first)
        self.pool = pool
        self.push_mode = push_mode
        self.hinge_margin = float(hinge_margin)
        self.eps = float(eps)

    def __str__(self):
        return (
            "---- TemporalAlignPushLoss Config ----\n"
            f"overlap_number: {self.overlap_number}\n"
            f"buffer:         {self.buffer}\n"
            f"align_weight:   {self.align_weight}\n"
            f"push_weight:    {self.push_weight}\n"
            f"teacher_first:  {self.teacher_first}\n"
            f"pool:           {self.pool}\n"
            f"push_mode:      {self.push_mode}\n"
            f"hinge_margin:   {self.hinge_margin}\n"
            "--------------------------------------"
        )

    def _get_overlap(self, v1: torch.Tensor, v2: torch.Tensor):
        """
        Return aligned sequences for the overlap region.
        v1, v2: [B,T,D]
        """
        B, T, D = v1.shape
        shift = self.overlap_number
        buf = self.buffer

        if shift == 0:
            # align everything
            v1_valid = v1
            v2_valid = v2
            return v1_valid, v2_valid

        overlap_len = T - shift
        if overlap_len <= 0:
            # no overlap
            return None, None

        # If buffer too large, drop it
        if 2 * buf >= overlap_len:
            buf = 0

        v1_valid = v1[:, (shift + buf):(T - buf), :]
        v2_valid = v2[:, buf:(overlap_len - buf), :]

        if v1_valid.shape[1] != v2_valid.shape[1]:
            min_len = min(v1_valid.shape[1], v2_valid.shape[1])
            v1_valid = v1_valid[:, :min_len, :]
            v2_valid = v2_valid[:, :min_len, :]

        if v1_valid.shape[1] == 0:
            return None, None

        return v1_valid, v2_valid

    def _pool_video(self, v: torch.Tensor) -> torch.Tensor:
        """
        v: [B,T,D] -> [B,D]
        """
        if self.pool == "mean":
            return v.mean(dim=1)
        # max
        return v.max(dim=1).values

    def forward(self, view1_seq: torch.Tensor, view2_seq: torch.Tensor):
        """
        view1_seq: [B,T,D]
        view2_seq: [B,T,D]
        """
        if view1_seq.dim() != 3 or view2_seq.dim() != 3:
            raise ValueError("Expected view1_seq/view2_seq shape [B,T,D].")
        if view1_seq.shape != view2_seq.shape:
            raise ValueError(f"Shapes must match. Got {view1_seq.shape} vs {view2_seq.shape}.")

        # Align overlap region
        v1_valid, v2_valid = self._get_overlap(view1_seq, view2_seq)

        if v1_valid is None:
            # no overlap => only push term
            loss_align = torch.zeros((), device=view1_seq.device, dtype=view1_seq.dtype)
        else:
            # Normalize
            v1n = F.normalize(v1_valid.float(), dim=-1)
            v2n = F.normalize(v2_valid.float(), dim=-1)

            # Stop-grad on teacher side if desired
            if self.teacher_first:
                v1n = v1n.detach()

            # Cosine alignment loss: 1 - cos
            cos = F.cosine_similarity(v2n, v1n, dim=-1)  # [B, T_valid]
            loss_align = (1.0 - cos).mean()

        # Push term: discourage similarity across different videos in batch
        # Pool each video to a single embedding, normalize, compute BxB similarity
        v1_pool = self._pool_video(view1_seq).float()
        v2_pool = self._pool_video(view2_seq).float()

        v1_pool = F.normalize(v1_pool, dim=-1)
        v2_pool = F.normalize(v2_pool, dim=-1)

        if self.teacher_first:
            v1_pool = v1_pool.detach()

        sim = v2_pool @ v1_pool.t()  # [B,B]
        B = sim.size(0)
        eye = torch.eye(B, device=sim.device, dtype=torch.bool)

        neg = sim[~eye]  # off-diagonal

        if neg.numel() == 0:
            loss_push = torch.zeros((), device=view1_seq.device, dtype=view1_seq.dtype)
        else:
            if self.push_mode == "mse":
                # Penalize any correlation between different videos
                loss_push = (neg ** 2).mean()
            else:
                # Hinge: only penalize if similarity exceeds margin
                loss_push = F.relu(neg - self.hinge_margin).mean()

        total = self.align_weight * loss_align + self.push_weight * loss_push
        return total, loss_align, loss_push
