"""Evaluate a trained model on the test split.

Loads `<output_dir>/<logger_name>/{model_config.json, train_config.json, best_model.pth}`,
runs sliding-window inference over `<labels_dir>/test.json` and reports frame-level F1 and
mAP at several frame tolerances (with and without NMS). Works with `python test.py ...`
(single GPU) or `torchrun --nproc_per_node=N test.py ...` (multi-GPU).
"""
import os
import time
import pickle
import json
import argparse
import torch
import numpy as np
import gzip
from ptflops import get_model_complexity_info

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from utils.logger import Logger
from models import get_model
from data_process.sl_dataset import ActionSpotVideoDataset
from data_process.utils import load_classes
from torch.utils.data import DataLoader
from tqdm import tqdm
from tabulate import tabulate
from losses_metrics.eval import process_frame_predictions
from losses_metrics.metrics import compute_mAPs_advanced
from losses_metrics.eval import non_maximum_supression
from training_utils.utils import setup_ddp, cleanup_ddp


def store_json(fpath, obj, pretty=False):
    kwargs = {}
    if pretty:
        kwargs['indent'] = 2
        kwargs['sort_keys'] = True
    with open(fpath, 'w') as fp:
        json.dump(obj, fp, **kwargs)


def store_gz_json(fpath, obj):
    with gzip.open(fpath, 'wt', encoding='ascii') as fp:
        json.dump(obj, fp)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logger_name', required=True, help='Experiment name used during training.')
    parser.add_argument('--output_dir', type=str, default='../logs', help='Root directory of experiment outputs.')
    parser.add_argument('--frames_dir', required=True, help='Directory containing extracted frames.')
    parser.add_argument('--labels_dir', required=True, help='Directory containing class.txt and test.json.')
    parser.add_argument('--model_choice', required=True, choices=['E2E', 'ASTRM', 'VIT'])
    parser.add_argument('--no_overlap', action='store_true', help='Use non-overlapping test windows.')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save', action='store_true', help='Save predictions to pred-test.json / pred-test.recall.json.gz.')
    parser.add_argument('--no_complexity', action='store_true', help='Skip FLOPs / latency measurement.')

    return parser.parse_args()


def load_model_compiled(model, checkpoint_path, device):
    raw_state = torch.load(checkpoint_path, map_location=device)
    if isinstance(raw_state, dict) and "state_dict" in raw_state:
        state_dict = raw_state["state_dict"]
    else:
        state_dict = raw_state

    fixed_state = {}
    for k, v in state_dict.items():
        new_k = k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k
        fixed_state[new_k] = v

    missing, unexpected = model.load_state_dict(fixed_state, strict=False)
    print("\n=== State Dict Load Report ===")
    print("Missing keys in model:", missing)
    print("Unexpected keys in checkpoint:", unexpected)
    model.to(device)
    return model


def gather_pred_dict(local_pred_dict, world_size, rank):
    """
    All-gather pred_dict entries from every rank onto rank 0.
    Returns the merged dict on rank 0, None on other ranks.
    """
    if world_size == 1:
        return local_pred_dict

    # Serialise local dict to bytes
    payload = pickle.dumps(local_pred_dict)
    payload_tensor = torch.ByteTensor(list(payload)).cuda()
    size_tensor = torch.tensor([payload_tensor.numel()], dtype=torch.long).cuda()

    # Exchange sizes so we can allocate buffers
    all_sizes = [torch.zeros(1, dtype=torch.long).cuda() for _ in range(world_size)]
    dist.all_gather(all_sizes, size_tensor)

    max_size = max(s.item() for s in all_sizes)
    padded = torch.zeros(max_size, dtype=torch.uint8).cuda()
    padded[:payload_tensor.numel()] = payload_tensor

    all_padded = [torch.zeros(max_size, dtype=torch.uint8).cuda() for _ in range(world_size)]
    dist.all_gather(all_padded, padded)

    if rank != 0:
        return None

    # Merge on rank 0
    merged = {}
    for buf, sz in zip(all_padded, all_sizes):
        chunk = pickle.loads(bytes(buf[:sz.item()].cpu().tolist()))
        for video, (scores, support) in chunk.items():
            if video not in merged:
                merged[video] = (scores.copy(), support.copy())
            else:
                merged[video][0][:] += scores
                merged[video][1][:] += support

    return merged


def main():
    args = get_args()

    rank, local_rank, world_size = setup_ddp()
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    is_main = rank == 0

    logger = Logger(args.logger_name, training=False, log_dir=args.output_dir) if is_main else None

    if is_main:
        logger.info('>>>>>>>>>>> Starting evaluation on the Test Dataset <<<<<<<<<<<<')
        logger.info(f'World size (GPUs): {world_size}')
        if device.type == 'cuda':
            logger.info(f'GPU name: {torch.cuda.get_device_name(local_rank)}')
            logger.info(f'GPU memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.2f} GB')

    saved_folder_path = os.path.join(args.output_dir, args.logger_name)

    # ── Load config & model ────────────────────────────────────────────────────
    model_config_path = os.path.join(saved_folder_path, 'model_config.json')
    with open(model_config_path, 'r') as f:
        config = json.load(f)

    model = get_model(args.model_choice, config)
    checkpoint_path = os.path.join(saved_folder_path, 'best_model.pth')
    model = load_model_compiled(model, checkpoint_path, device)
    model.eval()

    # sl_train/ssl_train write train_config.json (older ssl runs wrote training_args.json)
    train_config_path = os.path.join(saved_folder_path, 'train_config.json')
    if not os.path.exists(train_config_path):
        train_config_path = os.path.join(saved_folder_path, 'training_args.json')
    with open(train_config_path, 'r') as f:
        train_config = json.load(f)
    crop_dim = train_config.get('crop_dim', 224) or None

    train_tolerances = train_config.get('tolerances', [0, 1, 2, 4])
    batch_size = train_config.get('batch_size', 4)
    num_workers = args.num_workers or train_config.get('num_workers', 4)

    if is_main:
        logger.info(f'Tolerances: {train_tolerances}')
        logger.info(f'Batch size: {batch_size} | Workers: {num_workers}')

    # ── Dataset & distributed sampler ─────────────────────────────────────────
    frames_dir = args.frames_dir
    labels_file_path = os.path.join(args.labels_dir, 'test.json')
    classes = load_classes(os.path.join(args.labels_dir, 'class.txt'))

    dataset = ActionSpotVideoDataset(
        classes, labels_file_path, frames_dir, 'rgb',
        config['num_frames_clip'],
        overlap_len=0 if args.no_overlap else config['num_frames_clip'] // 2,
        crop_dim=crop_dim
    )

    # DistributedSampler splits clips across GPUs; shuffle=False keeps ordering
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    if is_main:
        logger.info(f'Test dataset size: {len(dataset)}')

    pred_file = None
    if args.save and is_main:
        pred_file = os.path.join(saved_folder_path, 'pred-test')
        logger.info(f'Saving predictions: {pred_file}')

    evaluate(model, dataset, dataloader, rank, world_size, logger, device,
             tolerances=train_tolerances, save_pred=pred_file,
             batch_size=batch_size, num_workers=num_workers, plot_pr=True,
             is_main=is_main)

    # Complexity stats only need one GPU
    if is_main and not args.no_complexity:
        frame_hw = (crop_dim, crop_dim) if crop_dim else (224, 398)
        evaluate_model_complexity(model, logger, device, num_frames=config['num_frames_clip'], frame_hw=frame_hw)


def evaluate(model, dataset, dataloader, rank, world_size, logger, device,
             save_pred, tolerances, num_workers=8, batch_size=8,
             high_recall_score_threshold=0.01, nms_window=1,
             plot_pr=False, is_main=False):

    classes = dataset._class_dict

    # Each rank accumulates predictions only for the clips it processes
    local_pred_dict = {}
    for video, video_len, _ in dataset.videos:
        local_pred_dict[video] = (
            np.zeros((video_len, model.num_classes), np.float32),
            np.zeros(video_len, np.int32)
        )

    with torch.no_grad():
        for clip in tqdm(dataloader, desc=f"[GPU {rank}] Evaluating", unit="clip", disable=not is_main):
            _, batch_pred_scores = model.predict(clip['frame'], device)

            for i in range(clip['frame'].shape[0]):
                video = clip['video'][i]
                scores, support = local_pred_dict[video]
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

    # ── Gather all partial predictions onto rank 0 ────────────────────────────
    if world_size > 1:
        dist.barrier()
    pred_dict = gather_pred_dict(local_pred_dict, world_size, rank)

    if not is_main:
        return

    # ── All metric computation happens only on rank 0 ─────────────────────────
    err, f1, pred_events, pred_events_high_recall, pred_scores = process_frame_predictions(
        dataset, classes, pred_dict, high_recall_score_threshold
    )

    logger.info('=== Results on test (w/o NMS) ===')
    logger.info('Error (frame-level): {:0.2f}\n'.format(err.get() * 100))

    def get_f1_tab_row(str_k):
        k = classes[str_k] if str_k != 'any' else None
        return [str_k, f1.get(k) * 100, *f1.tp_fp_fn(k)]

    rows = [get_f1_tab_row('any')]
    for c in sorted(classes):
        rows.append(get_f1_tab_row(c))
    logger.info(tabulate(rows, headers=['Exact frame', 'F1', 'TP', 'FP', 'FN'], floatfmt='0.2f'))

    logger.info('\n=== Results (w/o NMS) ===')
    mAPs, _ = compute_mAPs_advanced(dataset.labels, pred_events_high_recall,
                                    tolerances=tolerances, logger=logger, plot_pr=plot_pr)

    pred = non_maximum_supression(pred_events_high_recall, nms_window)
    logger.info('\n=== Results (w/ NMS) ===')
    compute_mAPs_advanced(dataset.labels, pred, tolerances=tolerances,
                          logger=logger, plot_pr=plot_pr)

    if save_pred is not None:
        store_json(save_pred + '.json', pred_events)
        store_gz_json(save_pred + '.recall.json.gz', pred_events_high_recall)


def evaluate_model_complexity(model, logger, device, num_frames=100, frame_hw=(224, 224),
                              num_warmup=20, num_iters=100):
    device = torch.device(device) if isinstance(device, str) else device
    model.eval()
    h, w = frame_hw

    dummy_input = torch.randn(1, num_frames, 3, h, w, device=device)

    macs, params = get_model_complexity_info(
        model,
        (num_frames, 3, h, w),
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False
    )

    gflops = 2 * macs / 1e9
    params_million = params / 1e6
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = (total_params * 4) / (1024 ** 2)

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        start_time = time.perf_counter()
        for _ in range(num_iters):
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        total_time = time.perf_counter() - start_time

    avg_latency_ms = (total_time / num_iters) * 1000
    clips_per_sec = 1.0 / (total_time / num_iters)
    frames_per_sec = clips_per_sec * num_frames

    logger.info("Model Complexity Summary:")
    logger.info(f"  MACs: {macs:.3e}")
    logger.info(f"  GFLOPs: {gflops:.2f}")
    logger.info(f"  Params: {params_million:.2f} M")
    logger.info(f"  Model Size: {model_size_mb:.2f} MB")
    logger.info(f"  Avg Latency: {avg_latency_ms:.2f} ms/clip")
    logger.info(f"  Throughput: {clips_per_sec:.2f} clips/s")
    logger.info(f"  Throughput: {frames_per_sec:.2f} frames/s")

    return {
        "macs": macs,
        "gflops": gflops,
        "params": params_million,
        "model_size_mb": model_size_mb,
        "latency_ms_per_clip": avg_latency_ms,
        "clips_per_sec": clips_per_sec,
        "frames_per_sec": frames_per_sec,
    }


if __name__ == '__main__':
    try:
        main()
    finally:
        cleanup_ddp()