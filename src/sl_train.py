import torch
import os
import json
import numpy as np
import random
from tabulate import tabulate
from torch.utils.data import DataLoader
from tqdm import tqdm
from contextlib import nullcontext
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


from configs import get_args
from data_process.sl_dataset import ActionSpotVideoDataset, ActionSpotDataset
from data_process.utils import load_classes, split_by_video, load_json
from models import get_model
from training_utils.optimizer import build_optimizer, get_lr_scheduler
from training_utils.utils import setup_ddp, cleanup_ddp, set_seed, is_main_process
from losses_metrics.losses import CELoss
from losses_metrics.metrics import compute_mAPs
from losses_metrics.eval import process_frame_predictions
from utils.io import save_to_json
from utils.logger import Logger

# Number of frames sampled per training "epoch" (clips are sampled randomly).
EPOCH_NUM_FRAMES = 500000


def get_datasets(args):
    class_file_path = os.path.join(args.labels_dir, 'class.txt')
    classes = load_classes(class_file_path)
    train_labels_path = os.path.join(args.labels_dir, 'train.json')
    train_labels = load_json(train_labels_path)
    val_labels_path = os.path.join(args.labels_dir, 'val.json')
    val_labels = load_json(val_labels_path)

    crop_dim = args.crop_dim or None  # 0 -> no crop (resize to 224x398)
    dataset_len = EPOCH_NUM_FRAMES // args.clip_len
    dataset_kwargs = {
        'crop_dim': crop_dim, 'dilate_len': args.dilate_len,
        'mixup': args.mixup
    }

    if args.fg_upsample is not None:
        assert args.fg_upsample > 0
        dataset_kwargs['fg_upsample'] = args.fg_upsample

    print('Dataset size:', dataset_len)
    train_dataset_len = dataset_len
    if args.percent_use < 1.0:
        # Split datasets based on videos to avoid data leakage
        train_labels, _ = split_by_video(train_labels, labeled_ratio=args.percent_use, seed=args.seed)
        train_dataset_len = int(dataset_len * args.percent_use)
        print(f'Number of training videos after percent_use={args.percent_use}: {len(train_labels)}')

    train_data = ActionSpotDataset(
        classes, train_labels,
        args.frames_dir, args.modality, args.clip_len, train_dataset_len,
        is_eval=False, **dataset_kwargs)
    val_data = ActionSpotDataset(
        classes, val_labels,
        args.frames_dir, args.modality, args.clip_len, dataset_len // 4,
        **dataset_kwargs)

    # Full-video dataset used for validation mAP
    val_data_frames = ActionSpotVideoDataset(
        classes, val_labels_path,
        args.frames_dir, args.modality, args.clip_len,
        crop_dim=crop_dim, overlap_len=args.clip_len//2)

    return classes, train_data, val_data, val_data_frames


def main():
    # parse command line arguments
    args = get_args()

    rank, local_rank, world_size = setup_ddp()

    if args.device == 'cuda' and torch.cuda.is_available():
        # Set the device specific to this process
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Logger only on rank 0
    if rank == 0:
        logger = Logger(args.logger_name, training=True, log_dir=args.output_dir)
        save_folder = logger.get_log_folder_path()
        logger.info(f'Logs and model checkpoints will be saved to: {save_folder}')
        logger.info(f'Running on {world_size} GPU(s)')
        logger.info(f'Seed: {args.seed}')
        logger.info(f'num workers for DataLoader: {args.num_workers}')
    else:
        logger = None
        save_folder = None

    # Broadcast save_folder to all ranks so they know where they are (if needed)
    if world_size > 1:
        save_folder_list = [save_folder]
        dist.broadcast_object_list(save_folder_list, src=0)
        if rank != 0:
            save_folder = save_folder_list[0]

    # Set random seed
    set_seed(args.seed + rank)
    
    # Load Datasets
    classes, train_dataset, val_dataset, val_map_dataset = get_datasets(args)

    if rank == 0:
        logger.info(f'Random seed set to: {args.seed}')
        logger.info(f'Number of classes: {len(classes)}')

    def worker_init_fn(worker_id):
        # Use seed + rank + worker_id to ensure every worker on every GPU is distinct
        random.seed(args.seed + rank + worker_id)

    if world_size > 1:
        # Multi-GPU: Use DistributedSampler
        train_labeled_sampler = DistributedSampler(
            train_dataset,
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
        
        train_loader = DataLoader(
            train_dataset, shuffle=False, batch_size=args.batch_size,
            pin_memory=True, persistent_workers=True,  num_workers=args.num_workers, sampler=train_labeled_sampler,
            prefetch_factor=2, worker_init_fn=worker_init_fn)
            
        val_loader = DataLoader(
            val_dataset, shuffle=False, batch_size=args.batch_size,
            pin_memory=True, persistent_workers=False, num_workers=args.num_workers, sampler=val_sampler,
            prefetch_factor=2, worker_init_fn=worker_init_fn)
    else:
        # Single GPU
        train_loader = DataLoader(
            train_dataset, shuffle=True, batch_size=args.batch_size,
            pin_memory=True, num_workers=args.num_workers,
            prefetch_factor=2, worker_init_fn=worker_init_fn)
            
        val_loader = DataLoader(
            val_dataset, shuffle=False, batch_size=args.batch_size,
            pin_memory=True, num_workers=args.num_workers,
            prefetch_factor=2, worker_init_fn=worker_init_fn)

    if rank == 0:
        logger.info(f'Training dataset size: {len(train_dataset)} clips')
        logger.info(f'Validation dataset size: {len(val_dataset)} clips')
        logger.info(f'Effective total batch size: {args.batch_size * world_size}')

    # Initialize the model
    with open(args.model_config_file, 'r') as f:
        model_config = json.load(f)

    model_config['num_classes'] = len(classes) + 1
    model_config['num_frames_clip'] = args.clip_len

    if rank == 0:
        save_to_json(model_config, os.path.join(save_folder, 'model_config.json'))
        # save args config as well 
        save_to_json(vars(args), os.path.join(save_folder, 'train_config.json'))

    model = get_model(args.model_choice, model_config)
    model = model.to(device)
    
    
    if world_size > 1:
        # Ensure we use the correct device_ids
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if rank == 0:
        logger.info(f'Model initialized with configuration: {model_config}')

    # Optimizer & Scaler
    use_amp = args.usefp16 and device.type == 'cuda'
    optimizer = build_optimizer(model, args.optimizer, args.learning_rate, args.weight_decay)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    num_steps_per_epoch = len(train_loader)
    lr_scheduler = get_lr_scheduler(args, optimizer, num_steps_per_epoch)

    # Loss
    criterion = CELoss(len(classes)+1, args.fg_weight).to(device)

    if rank == 0:
        logger.info(f'Tolerance values for mAP evaluation: {args.tolerances}')
        logger.info('>>>>>>>>>>>>>>>Setup complete. Starting training...<<<<<<<<<<<<<')

    best_mAP = 0.0
    
    for epoch in range(args.num_epochs):
        if world_size > 1:
            train_labeled_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        if rank == 0:
            logger.info(f'--- Epoch {epoch}/{args.num_epochs} ---')
        
        # Validation loss is skipped by default; model selection uses validation mAP.
        train_loss, val_loss = train_one_epoch(
            model, train_loader, val_loader, criterion,
            optimizer, lr_scheduler, device,
            scaler=scaler, val_loss=False, use_amp=use_amp
        )
                
        if rank == 0:
            logger.info(f'Training Loss: {train_loss:.4f}')
            if val_loss is not None:
                logger.info(f'Validation Loss: {val_loss:.4f}')
            if lr_scheduler is not None:
                # Access lr from the first param group
                current_lr = optimizer.param_groups[0]["lr"]
                logger.info(f'Learning Rate after epoch {epoch}: {current_lr}')
        
        if world_size > 1:
            dist.barrier(device_ids=[local_rank])

        # Evaluate mAP (all ranks participate; rank 0 returns the score)
        if epoch >= args.start_val_epoch:
            model_to_eval = model.module if hasattr(model, 'module') else model
            avg_mAP = evaluate_map_one_epoch(model_to_eval, val_map_dataset, device, args.batch_size, logger, 
                                             args.num_workers, tolerances=args.tolerances)

            if rank == 0:
                if avg_mAP > best_mAP:
                    best_mAP = avg_mAP
                    
                    torch.save(model_to_eval.state_dict(), os.path.join(save_folder, 'best_model.pth'))
                    logger.info(f'Best model saved with mAP: {best_mAP:.4f}')
            
            if world_size > 1:
                dist.barrier(device_ids=[local_rank])


def train_one_epoch(model, train_dataloader, val_dataloader, criterion, optimizer, lr_scheduler, device, *,
                    val_loss=True, use_amp=True, num_classes=None, scaler=None):
    
    device_type = device.type if isinstance(device, torch.device) else str(device)
    amp_enabled = use_amp and (device_type in ("cuda", "cpu"))
    autocast_ctx = (torch.amp.autocast(device_type=device_type) if amp_enabled else nullcontext())

    # -------- Train --------
    model.train()
    train_running_loss = 0.0
    
    pbar = tqdm(train_dataloader, desc="Training", unit="batch", disable=not is_main_process())

    for batch in pbar:
        inputs = train_dataloader.dataset.load_frame_gpu(batch, device)
        labels = batch["label"].to(device)

        labels = labels.flatten() if len(labels.shape) == 2 else labels.view(-1, labels.shape[-1])

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx:
            outputs = model(inputs)
            logits = outputs.reshape(-1, num_classes or outputs.size(-1))
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        if lr_scheduler is not None:
            lr_scheduler.step()

        train_running_loss += loss.item()
        if is_main_process():
            pbar.set_postfix(batch_loss=f"{loss.item():.4f}")
       
    train_epoch_loss = train_running_loss / len(train_dataloader)
    if dist.is_initialized():
        avg_tensor = torch.tensor(train_epoch_loss, device=device)
        dist.all_reduce(avg_tensor, op=dist.ReduceOp.SUM)
        train_epoch_loss = avg_tensor.item() / dist.get_world_size()

    # -------- Val --------
    val_epoch_loss = None

    if val_loss:
        model.eval()
        val_running_loss = 0.0
        
        vpbar = tqdm(val_dataloader, desc="Validation", unit="batch", disable=not is_main_process())

        with torch.no_grad():
            for batch in vpbar:
                inputs = val_dataloader.dataset.load_frame_gpu(batch, device)
                labels = batch["label"].to(device)

                labels = labels.flatten() if len(labels.shape) == 2 else labels.view(-1, labels.shape[-1])

                with autocast_ctx:
                    outputs = model(inputs)
                    logits = outputs.reshape(-1, num_classes or outputs.size(-1))
                    loss = criterion(logits, labels)

                val_running_loss += loss.item()
                
                if is_main_process():
                    vpbar.set_postfix(batch_loss=f"{loss.item():.4f}")

        val_epoch_loss = val_running_loss / len(val_dataloader)
        if dist.is_initialized():
            avg_tensor = torch.tensor(val_epoch_loss, device=device)
            dist.all_reduce(avg_tensor, op=dist.ReduceOp.SUM)
            val_epoch_loss = avg_tensor.item() / dist.get_world_size()

    return train_epoch_loss, val_epoch_loss


def evaluate_map_one_epoch(model, dataset, device, batch_size, logger, 
                           num_workers=8, high_recall_score_threshold=0.01, 
                           tolerances = [0, 1, 2, 4], plot_pr=False):
    
    """Run sliding-window inference on `dataset` and return the average mAP (%) on rank 0 (0.0 on other ranks)."""
    distributed = dist.is_initialized()
    rank = dist.get_rank() if distributed else 0

    classes = dataset._class_dict
    pred_dict = {}
    for video, video_len, _ in dataset.videos:
        pred_dict[video] = (
            np.zeros((video_len, model.num_classes), np.float32),
            np.zeros(video_len, np.int32)
        )
        
    sampler = DistributedSampler(dataset, shuffle=False) if distributed else None
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True, 
        batch_size=batch_size
    )
    
    model.eval()
    iterator = tqdm(dataloader, desc="Evaluating", unit="clip", disable=rank != 0)

    with torch.no_grad():
        for clip in iterator:
            frames = clip['frame'].to(device)
            _, batch_pred_scores = model.predict(frames, device)
            if isinstance(batch_pred_scores, torch.Tensor):
                batch_pred_scores = batch_pred_scores.cpu().numpy()

            for i in range(frames.shape[0]):
                video = clip['video'][i]
                scores, support = pred_dict[video]
                pred_scores = batch_pred_scores[i]
                start = clip['start'][i].item()
                
                if start < 0:
                    pred_scores = pred_scores[-start:, :]
                    start = 0
                end = start + pred_scores.shape[0]
                if end >= scores.shape[0]:
                    end = scores.shape[0]
                    pred_scores = pred_scores[:end - start, :]
                
                scores[start:end, :] += pred_scores
                support[start:end] += 1

    # Sum the per-rank partial predictions
    if distributed:
        for video in pred_dict.keys():
            scores_np, support_np = pred_dict[video]
            scores_tensor = torch.from_numpy(scores_np).to(device)
            support_tensor = torch.from_numpy(support_np).to(device)
            dist.all_reduce(scores_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(support_tensor, op=dist.ReduceOp.SUM)
            pred_dict[video] = (scores_tensor.cpu().numpy(), support_tensor.cpu().numpy())

    avg_mAP = 0.0
    if rank == 0:
        err, f1, pred_events, pred_events_high_recall, pred_scores = process_frame_predictions(dataset, classes, pred_dict, high_recall_score_threshold)

        logger.info('=== Results on validation (w/o NMS) ===')
        logger.info('Error (frame-level): {:0.2f}\n'.format(err.get() * 100))

        def get_f1_tab_row(str_k):
            k = classes[str_k] if str_k != 'any' else None
            return [str_k, f1.get(k) * 100, *f1.tp_fp_fn(k)]
        
        rows = [get_f1_tab_row('any')]
        for c in sorted(classes):
            rows.append(get_f1_tab_row(c))
            
        logger.info(tabulate(rows, headers=['Exact frame', 'F1', 'TP', 'FP', 'FN'], floatfmt='0.2f'))
        logger.info('')

        mAPs, _ = compute_mAPs(dataset.labels, pred_events_high_recall, logger=logger, plot_pr=plot_pr, tolerances=tolerances)
        avg_mAP = np.mean(mAPs[1:]) * 100

    if distributed:
        dist.barrier()

    return avg_mAP


if __name__ == '__main__':
    try:
        main()
    finally:
        cleanup_ddp()