import torch
import os
import json
import random
import copy
import math
from torch.utils.data import DataLoader
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from collections import Counter

from configs import get_args
from data_process.sl_dataset import ActionSpotVideoDataset
from data_process.ssl_dataset import SSLLabeledActionSpotDataset, SSLUnlabeledOverlappedActionSpotDataset
from data_process.utils import load_classes, load_json, split_by_video
from models import get_model
from training_utils.optimizer import build_optimizer, get_lr_scheduler
from losses_metrics.losses import CELoss, TemporalAlignPushLoss
from utils.io import save_to_json
from utils.logger import Logger

from sl_train import evaluate_map_one_epoch
from training_utils.utils import setup_ddp, cleanup_ddp, set_seed, is_main_process, to_cpu, update_ema, resume_from_checkpoint

# Number of frames sampled per training "epoch" (clips are sampled randomly).
EPOCH_NUM_FRAMES = 500000


def get_datasets(args, logger=None):
    class_file_path = os.path.join(args.labels_dir, 'class.txt')
    classes = load_classes(class_file_path)
    train_labels_path = os.path.join(args.labels_dir, 'train.json')
    train_labels = load_json(train_labels_path)
    val_labels_path = os.path.join(args.labels_dir, 'val.json')
    val_labels = load_json(val_labels_path)

    # Split datasets based on videos to avoid data leakage
    train_labels, train_unlabeled = split_by_video(train_labels, labeled_ratio=args.percent_use, seed=args.seed)
    if logger is not None:
        logger.info(f'After video-based split with labeled_ratio={args.percent_use}:')
        logger.info(f'Number of labeled training videos: {len(train_labels)}')
        logger.info(f'Number of unlabeled training videos: {len(train_unlabeled)}')
        counter = Counter()
        for e in train_labels:
            for ev in e['events']:
                counter[ev['label']] += 1
        logger.info(f'Class distribution in labeled dataset: {dict(counter)}')

    if len(train_unlabeled) == 0:
        if logger is not None:
            logger.info("No unlabeled data found, using labeled data for unlabeled as well.")
        train_unlabeled = copy.deepcopy(train_labels)

    # The student augmentation (MotionGridMosaicRGB) needs a square crop divisible by 16.
    if not args.crop_dim or args.crop_dim % 16 != 0:
        raise ValueError('ssl_train.py requires --crop_dim > 0 and divisible by 16 (e.g. 224).')
    crop_dim = args.crop_dim
    train_dataset_len = int(EPOCH_NUM_FRAMES // args.clip_len * args.percent_use)
    unlabeled_dataset_len = int(EPOCH_NUM_FRAMES) // args.clip_len
    val_dataset_len = EPOCH_NUM_FRAMES // args.clip_len // 4
    dataset_kwargs = {
        'crop_dim': crop_dim, 'dilate_len': args.dilate_len,
        'mixup': args.mixup
    }

    if args.fg_upsample is not None:
        assert args.fg_upsample > 0
        dataset_kwargs['fg_upsample'] = args.fg_upsample

    train_labeled_data = SSLLabeledActionSpotDataset(
        classes, train_labels,
        args.frames_dir, args.modality, args.clip_len, train_dataset_len,
        is_eval=False, **dataset_kwargs)
    train_unlabeled_data = SSLUnlabeledOverlappedActionSpotDataset(
        train_unlabeled, args.frames_dir, args.modality, args.clip_len, unlabeled_dataset_len, overlapped_number=args.overlapped_number,
        crop_dim=crop_dim, wacv_style_augmentation=args.tma_augment)
    val_data = SSLLabeledActionSpotDataset(
        classes, val_labels,
        args.frames_dir, args.modality, args.clip_len, val_dataset_len,
        **dataset_kwargs)

    # validation dataset for mAP evaluation
    val_data_frames = ActionSpotVideoDataset(
        classes, val_labels_path,
        args.frames_dir, args.modality, args.clip_len,
        crop_dim=crop_dim, overlap_len=0)

    return classes, train_labeled_data, train_unlabeled_data, val_data, val_data_frames


def main():
    args = get_args()

    # Setup DDP (returns 0, 0, 1 for single GPU)
    rank, local_rank, world_size = setup_ddp()

    if args.device == 'cuda':
        if world_size > 1:
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Logger only on rank 0
    if rank == 0:
        logger = Logger(args.logger_name, training=True, log_dir=args.output_dir)
        save_folder = logger.get_log_folder_path()
        logger.info(f'Logs and model checkpoints will be saved to: {save_folder}')
        logger.info(f'Running on {world_size} GPU(s)')
        logger.info(f'Seed: {args.seed}')
        logger.info(f'Num workers for data loading: {args.num_workers}')
    else:
        logger = None
        save_folder = None
    
    if world_size > 1:
        save_folder_list = [save_folder]
        dist.broadcast_object_list(save_folder_list, src=0)
        if rank != 0:
            save_folder = save_folder_list[0]
    
    set_seed(args.seed + rank)
    classes, train_labeled_dataset, train_unlabeled_dataset, val_dataset, val_map_dataset = get_datasets(args, logger)

    if rank == 0:
        logger.info(f'Random seed set to: {args.seed}')
        logger.info(f'Number of classes: {len(classes)}')
        logger.info(f'Number of labeled training samples: {len(train_labeled_dataset)}')
        logger.info(f'Number of unlabeled training samples: {len(train_unlabeled_dataset)}')
        logger.info(f'Number of validation samples: {len(val_dataset)}')
        logger.info(f'Using TMA augmentation: {"Yes" if args.tma_augment else "No"}')

    def worker_init_fn(worker_id):
        random.seed(args.seed + rank * 1000 + worker_id)

    if world_size > 1:
        train_labeled_sampler = DistributedSampler(
            train_labeled_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed
        )
        train_unlabeled_sampler = DistributedSampler(
            train_unlabeled_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        
        train_labeled_loader = DataLoader(
            train_labeled_dataset,
            batch_size=args.batch_size,
            sampler=train_labeled_sampler,
            persistent_workers=True,
            pin_memory=True,
            num_workers=args.num_workers,
            prefetch_factor=2,
            drop_last=True,
            worker_init_fn=worker_init_fn,
        )
        train_unlabeled_loader = DataLoader(
            train_unlabeled_dataset,
            batch_size=args.batch_size,
            sampler=train_unlabeled_sampler,
            persistent_workers=True,
            pin_memory=True,
            num_workers=args.num_workers,
            prefetch_factor=2,
            drop_last=True,
            worker_init_fn=worker_init_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            persistent_workers=True,
            pin_memory=False,
            num_workers=args.num_workers,
            prefetch_factor=2
        )
    else:
        train_labeled_loader = DataLoader(
            train_labeled_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=True,
            num_workers=args.num_workers,
            prefetch_factor=2,
            worker_init_fn=worker_init_fn
        )
        train_unlabeled_loader = DataLoader(
            train_unlabeled_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=True,
            num_workers=args.num_workers,
            prefetch_factor=2,
            worker_init_fn=worker_init_fn
        )
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=True,
            num_workers=args.num_workers,
            worker_init_fn=worker_init_fn,
            prefetch_factor=2
        )
        
        train_labeled_sampler = None
        train_unlabeled_sampler = None
        val_sampler = None
    
    if rank == 0:
        logger.info(f'Labeled batches per epoch: {len(train_labeled_loader)}')
        logger.info(f'Unlabeled batches per epoch: {len(train_unlabeled_loader)}')
        logger.info(f'Validation batches: {len(val_loader)}')

    # Initialize models
    with open(args.model_config_file, 'r') as f:
        model_config = json.load(f)

    model_config['num_classes'] = len(classes) + 1
    model_config['num_frames_clip'] = args.clip_len

    if is_main_process():
        save_to_json(model_config, os.path.join(save_folder, 'model_config.json'))
        save_to_json(vars(args), os.path.join(save_folder, 'train_config.json'))

    student_model = get_model(args.model_choice, model_config, semi_supervised=True).to(device)
    teacher_model = get_model(args.model_choice, model_config, semi_supervised=True).to(device)
    teacher_model.load_state_dict(student_model.state_dict())

    # Wrap with DDP only if multi-GPU
    if world_size > 1:
        student_model = DDP(student_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # Teacher never gets grads
    for p in teacher_model.parameters():
        p.requires_grad_(False)
    teacher_model.eval()

    # sqrt LR scaling with the number of GPUs
    learning_rate = args.learning_rate * math.sqrt(world_size)
    optimizer = build_optimizer(student_model, args.optimizer, learning_rate, args.weight_decay)

    use_amp = args.usefp16 and device.type == 'cuda'
    scaler = torch.amp.GradScaler() if use_amp else None

    num_steps_per_epoch = len(train_labeled_loader)
    lr_scheduler = get_lr_scheduler(args, optimizer, num_steps_per_epoch)

    labeled_criterion = CELoss(len(classes) + 1, args.fg_weight).to(device)
    unlabeled_criterion = TemporalAlignPushLoss(overlap_number=args.overlapped_number, 
                                                teacher_first=True, push_weight=args.push_loss_weight).to(device)

    if rank == 0:
        logger.info(f'Model initialized with configuration: {model_config}')
        logger.info(f'Number of steps per epoch: {num_steps_per_epoch}')
        logger.info(f'Scaler for mixed precision training: {"Enabled" if scaler is not None else "Disabled"}')
        logger.info(f'Using project head alignment for unsupervised loss: {"Yes" if args.proj_head_align else "No"}')
        logger.info(f'Optimizer: {args.optimizer}, Learning Rate: {learning_rate}, Weight Decay: {args.weight_decay}')
        logger.info(f'{unlabeled_criterion}')
        logger.info('>>>>>>>>>>>>>>>Setup complete. Starting training...<<<<<<<<<<<<<')

    best_mAP = 0.0
    avg_mAP = 0.0
    start_epoch = 0
    # Resume from checkpoint if provided
    if args.resume_path != '':
        start_epoch, best_mAP = resume_from_checkpoint(
            args.resume_path,
            student_model,
            teacher_model,
            optimizer=optimizer,
            device=device,
        )
        if rank == 0:
            logger.info(f'Resumed training from checkpoint: {args.resume_path}')

    for epoch in range(start_epoch, args.num_epochs):
        if world_size > 1:
            train_labeled_sampler.set_epoch(epoch)
            train_unlabeled_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        # Unsupervised loss weight: 0 during supervised warm-up, then cosine ramp to 1 over 10 epochs
        unsup_start = args.supervise_warmup_epochs + 1
        lambda_u = ramp_lambda(epoch+1, warm=unsup_start, ramp=10, max_lambda=1)
        ema_decay_value = ema_decay(epoch+1, args.num_epochs, base=args.ema_decay, fast=0.99)

        if rank == 0:
            logger.info(f'--- Epoch {epoch}/{args.num_epochs} ---')
            logger.info(f'Unsupervised Loss Weight (lambda_u): {lambda_u:.4f}')
            logger.info(f'EMA Decay for Teacher Update: {ema_decay_value:.6f}')
            logger.info(f'Unsupervised loss starting at epoch {unsup_start}, current epoch: {epoch}')
            logger.info(f'Start Validation after epoch {args.start_val_epoch}, current epoch: {epoch}')
        
        unsup_params = {
            'overlapped_number': args.overlapped_number,
            'lambda_u': lambda_u,
            'ema_decay': ema_decay_value
        }

        train_loss = train_one_epoch(
            student_model, teacher_model, 
            train_labeled_loader, train_unlabeled_loader,
            labeled_criterion, unlabeled_criterion, optimizer, lr_scheduler, unsup_params,
            device, project_head_alignment=args.proj_head_align,
            scaler=scaler, rank=rank, world_size=world_size, use_amp=use_amp
        )

        total_train_loss = train_loss['total_loss']
        sup_train_loss = train_loss['supervised_loss']
        unsup_train_loss = train_loss['unsupervised_loss']

        if world_size > 1:
            dist.barrier(device_ids=[local_rank])

        # Before start_val_epoch only the (cheap) validation loss is computed; afterwards the
        # teacher is evaluated with validation mAP, which drives model selection.
        val_loss = None
        if epoch < args.start_val_epoch:
            val_loss = val_one_epoch_loss(teacher_model, val_loader, labeled_criterion, device, use_amp=use_amp)

        if world_size > 1:
            dist.barrier(device_ids=[local_rank])

        if epoch >= args.start_val_epoch:
            model_to_eval = teacher_model.module if hasattr(teacher_model, 'module') else teacher_model
            avg_mAP = evaluate_map_one_epoch(model_to_eval, val_map_dataset, device, args.batch_size, logger,
                                             args.num_workers, tolerances=args.tolerances)

        if is_main_process():
            logger.info(f'Epoch {epoch} Training Loss: {total_train_loss:.4f} (Sup: {sup_train_loss:.4f}, Unsup: {unsup_train_loss:.4f})')
            if val_loss is not None:
                logger.info(f'Epoch {epoch} Validation Loss: {val_loss:.4f}')
            if lr_scheduler is not None:
                current_lr = optimizer.param_groups[0]["lr"]
                logger.info(f'Learning Rate after epoch {epoch}: {current_lr}')

            student_sd = student_model.module.state_dict() if hasattr(student_model, "module") else student_model.state_dict()
            teacher_sd = teacher_model.module.state_dict() if hasattr(teacher_model, "module") else teacher_model.state_dict()
            if avg_mAP > best_mAP:
                best_mAP = avg_mAP
                torch.save(to_cpu(teacher_sd), os.path.join(save_folder, 'best_model.pth'))
                logger.info(f'Best model saved with mAP: {best_mAP:.4f}')

            # Resumable checkpoint every 5 epochs and at the end
            if epoch % 5 == 0 or epoch == args.num_epochs - 1:
                checkpoint = {
                    'epoch': epoch,
                    'student': to_cpu(student_sd),
                    'teacher': to_cpu(teacher_sd),
                    'optimizer': to_cpu(optimizer.state_dict()),
                    'best_map': best_mAP,
                }
                torch.save(checkpoint, os.path.join(save_folder, 'checkpoint_latest.pth'))

            logger.info('-----------------------------------')

    cleanup_ddp()


def cycle_loader(dataloader):
    """Iterate over a DataLoader forever."""
    while True:
        for batch in dataloader:
            yield batch


def ramp_lambda(epoch, warm=20, ramp=10, max_lambda=1.0):
    """0 before `warm`, then cosine ramp-up to `max_lambda` over `ramp` epochs."""
    if epoch < warm: return 0.0
    t = min(1.0, (epoch - warm) / max(1, ramp))
    return max_lambda * (0.5 - 0.5 * math.cos(math.pi * t))  # cosine ramp


def ema_decay(epoch, total, base=0.9995, fast=0.99):
    """Teacher EMA decay, cosine-scheduled from `fast` (early) to `base` (late)."""
    t = epoch / max(1, total)
    return base - (base - fast) * (0.5 * (1 + math.cos(math.pi * t)))


def train_one_epoch(student_model, teacher_model, labeled_trainloader, unlabeled_trainloader,
                    labeled_criterion, unlabeled_criterion, 
                    optimizer, lr_scheduler, unsup_train_param, device, 
                    project_head_alignment=False, use_amp=True, scaler=None,
                    rank=0, world_size=1):
    """One epoch of teacher-student training.

    Each step:
      1. supervised cross-entropy on a labeled clip (student);
      2. temporal feature alignment on an unlabeled clip: the EMA teacher sees a weakly
         augmented view, the student a strongly augmented (optionally time-shifted) view;
      3. optimizer step on the student, then EMA update of the teacher.
    """
    device_type = device.type if isinstance(device, torch.device) else str(device)
    amp_enabled = use_amp and (device_type in ("cuda", "cpu"))

    student_model.train()
    
    # Unwrap modules for EMA and Clipping
    student_module = student_model.module if hasattr(student_model, 'module') else student_model
    teacher_module = teacher_model.module if hasattr(teacher_model, 'module') else teacher_model

    total_metrics = torch.zeros(3, device=device) # [total, sup, unsup]
    num_batches = 0
    
    autocast_ctx = (torch.autocast(device_type=device_type) if amp_enabled else nullcontext())
    sup_iter = cycle_loader(labeled_trainloader)
    unsup_iter = cycle_loader(unlabeled_trainloader)

    steps_per_epoch = len(unlabeled_trainloader)
    lambda_u = unsup_train_param['lambda_u']
    ema_decay_value = unsup_train_param['ema_decay']

    if rank == 0:
        progress_bar = tqdm(range(steps_per_epoch), total=steps_per_epoch, desc="Train (SSL)", ncols=120)
    else:
        progress_bar = range(steps_per_epoch)

    for step in progress_bar:
        batch_l = next(sup_iter)
        batch_u = next(unsup_iter)

        optimizer.zero_grad(set_to_none=True)

        # --- 1. Supervised Pass ---
        x_l = labeled_trainloader.dataset.load_frame_gpu(batch_l, device)
        y_l = batch_l['label'].to(device)
        y_l = y_l.flatten() if len(y_l.shape) == 2 else y_l.view(-1, y_l.shape[-1])
    
        with autocast_ctx:
            logits_l = student_model(x_l)
            B, T, C = logits_l.shape
            logits_l = logits_l.reshape(B*T, C)
            sup_loss = labeled_criterion(logits_l, y_l)
        
        # Skip the DDP gradient all-reduce here; it happens on the unsupervised backward below.
        sync_ctx = student_model.no_sync() if hasattr(student_model, 'no_sync') else nullcontext()
        with sync_ctx:
            if scaler is not None and amp_enabled:
                scaler.scale(sup_loss).backward()
            else:
                sup_loss.backward()

        sup_loss_val = sup_loss.detach()
        del x_l, y_l, logits_l, sup_loss
        
        # --- 2. Unsupervised Pass ---
        xa_w, xa_s = unlabeled_trainloader.dataset.load_frame_gpu(batch_u, device)

        with torch.no_grad(), autocast_ctx:
            if project_head_alignment:
                features_a_t = teacher_model(xa_w, return_pred_features=True, return_pred=False)
            else:
                features_a_t = teacher_model(xa_w, return_backbone_features=True, return_pred=False)
            features_a_t = F.normalize(features_a_t.float(), dim=-1)
        del xa_w

        with autocast_ctx:
            if project_head_alignment:
                features_a_s = student_model(xa_s, return_pred_features=True, return_pred=False)
            else:
                features_a_s = student_model(xa_s, return_backbone_features=True, return_pred=False) 
        del xa_s

        features_a_s = F.normalize(features_a_s.float(), dim=-1)

        unsup_loss, _, _ = unlabeled_criterion(features_a_t, features_a_s)
        scaled_unsup_loss = unsup_loss * lambda_u

        if scaler is not None and amp_enabled:
            scaler.scale(scaled_unsup_loss).backward()
        else:
            scaled_unsup_loss.backward()
        
        scaled_unsup_loss_val = scaled_unsup_loss.detach()
        del features_a_t, features_a_s, unsup_loss, scaled_unsup_loss

        # --- 3. Optimizer step + teacher EMA ---
        if scaler is not None and amp_enabled:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student_module.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(student_module.parameters(), max_norm=1.0)
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        update_ema(student_module, teacher_module, ema_decay=ema_decay_value)

        # --- 4. Logging ---
        total_metrics[0] += sup_loss_val + scaled_unsup_loss_val
        total_metrics[1] += sup_loss_val
        total_metrics[2] += scaled_unsup_loss_val
        num_batches += 1

        if rank == 0 and step % 10 == 0 and hasattr(progress_bar, 'set_postfix'):
            progress_bar.set_postfix({
                "Sup":   f"{sup_loss_val.item():.4f}",
                "Unsup": f"{scaled_unsup_loss_val.item():.4f}",
                "Avg":   f"{total_metrics[0].item() / num_batches:.4f}",
            })

        del sup_loss_val, scaled_unsup_loss_val

    if rank == 0 and hasattr(progress_bar, 'close'):
        progress_bar.close()

    if world_size > 1:
        dist.barrier()
        dist.all_reduce(total_metrics, op=dist.ReduceOp.SUM)
        
        num_batches_tensor = torch.tensor(num_batches, device=device)
        dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
        total_batches = num_batches_tensor.item()
    else:
        total_batches = num_batches

    avg_total = total_metrics[0].item() / max(1, total_batches)
    avg_sup = total_metrics[1].item() / max(1, total_batches)
    avg_unsup = total_metrics[2].item() / max(1, total_batches)
    
    return {'total_loss': avg_total, 'supervised_loss': avg_sup,'unsupervised_loss': avg_unsup}


def val_one_epoch_loss(model, val_dataloader, criterion, device, use_amp=True):
    """Average supervised loss on the validation clips (aggregated across ranks)."""
    device_type = device.type if isinstance(device, torch.device) else str(device)
    amp_enabled = use_amp and (device_type in ("cuda", "cpu"))
    model.eval()
    val_running_loss = torch.tensor(0.0, device=device)
    total_batches = torch.tensor(0, device=device)
    autocast_ctx = (torch.autocast(device_type=device_type) if amp_enabled else nullcontext())
    vpbar = tqdm(val_dataloader, desc="Validation", unit="batch", disable=(not is_main_process()))

    with torch.no_grad():
        for batch in vpbar:
            inputs = val_dataloader.dataset.load_frame_gpu(batch, device)
            labels = batch["label"].to(device)
            labels = labels.flatten() if len(labels.shape) == 2 else labels.view(-1, labels.shape[-1])

            with autocast_ctx:
                outputs = model(inputs)
                B, T, C = outputs.shape
                logits = outputs.reshape(B*T, C)
                loss = criterion(logits, labels)

            val_running_loss += loss.detach()
            total_batches += 1

            if is_main_process() and hasattr(vpbar, 'set_postfix'):
                vpbar.set_postfix(batch_loss=f"{loss.item():.4f}")

            del inputs, labels, outputs, logits, loss

    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()
        dist.all_reduce(val_running_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_batches, op=dist.ReduceOp.SUM)

    val_epoch_loss = val_running_loss.item() / max(1, total_batches.item())
    return val_epoch_loss


if __name__ == '__main__':
    try:
        main()
    finally:
        cleanup_ddp()
