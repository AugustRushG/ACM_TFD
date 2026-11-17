import torch
import os 
import torch.distributed as dist
import random
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Helper to check if we are on the main GPU (Rank 0)
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

# Helper to average loss across all GPUs


def to_cpu(state_dict):
    """Recursively moves a state_dict to CPU to save VRAM."""
    cpu_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            cpu_dict[k] = v.cpu()
        elif isinstance(v, dict): # Handle nested dicts (like optimizer param_groups)
            cpu_dict[k] = to_cpu(v)
        else:
            cpu_dict[k] = v
    return cpu_dict

@torch.no_grad()
def update_ema(student, teacher, ema_decay=0.999):
    """EMA update of teacher weights and buffers."""
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        if not torch.isfinite(s_param).all():
            continue  # skip NaN/Inf parameters safely
        t_param.data.mul_(ema_decay).add_(s_param.data, alpha=1 - ema_decay)
    for t_buf, s_buf in zip(teacher.buffers(), student.buffers()):
        t_buf.copy_(s_buf)


def setup_ddp():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # 1. Get the rank/device info from environment variables
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        # 2. CRITICAL STEP: Set the device BEFORE initializing the process group
        # This silences the warning and ensures NCCL binds correctly immediately.
        torch.cuda.set_device(local_rank)

        # 3. Initialize Process Group
        dist.init_process_group(backend="nccl", init_method="env://")
        
        print(f"Initialized DDP on Rank: {rank}, Local Rank: {local_rank}")
        return rank, local_rank, world_size
    else:
        # Not running with torchrun/DDP
        print("Not using DDP (Single GPU/CPU mode)")
        return 0, 0, 1


def cleanup_ddp():
    """Clean up the distributed environment"""
    if dist.is_initialized():
        dist.destroy_process_group()


def resume_from_checkpoint(resume_path, student_model, teacher_model, optimizer, device, map_location="cpu"):
    # Force CPU load, no matter what
    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)

    student_module = student_model.module if hasattr(student_model, "module") else student_model
    teacher_module = teacher_model.module if hasattr(teacher_model, "module") else teacher_model

    student_module.load_state_dict(checkpoint["student"], strict=True)
    teacher_module.load_state_dict(checkpoint["teacher"], strict=True)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

        # Move optimizer state tensors to the correct GPU for this rank
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device, non_blocking=True)

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_mAP = checkpoint.get("best_map", 0.0)

    return start_epoch, best_mAP
