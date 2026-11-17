import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description='Training arguments shared by sl_train.py (supervised) and ssl_train.py (semi-supervised).')

    # ---- Paths / experiment ----
    g = parser.add_argument_group('experiment')
    g.add_argument('--frames_dir', type=str, required=True,
                   help='Directory containing extracted frames, one sub-folder per video.')
    g.add_argument('--labels_dir', type=str, required=True,
                   help='Directory containing class.txt, train.json, val.json and test.json.')
    g.add_argument('--model_config_file', type=str, required=True,
                   help='Path to a model config JSON (see models/configs/).')
    g.add_argument('--model_choice', type=str, required=True, choices=['E2E', 'ASTRM', 'VIT'],
                   help='Model architecture.')
    g.add_argument('--logger_name', type=str, required=True,
                   help='Experiment name. Outputs are written to <output_dir>/<logger_name>/.')
    g.add_argument('--output_dir', type=str, default='../logs',
                   help='Root directory for experiment outputs.')
    g.add_argument('--resume_path', type=str, default='',
                   help='[ssl_train only] Path to checkpoint_latest.pth to resume from.')
    g.add_argument('--device', type=str, default='cuda', help='cuda or cpu.')
    g.add_argument('--seed', type=int, default=1, help='Random seed (also controls the labeled/unlabeled split).')

    # ---- Optimisation ----
    g = parser.add_argument_group('optimisation')
    g.add_argument('--num_epochs', type=int, default=50)
    g.add_argument('--batch_size', type=int, default=8, help='Per-GPU batch size.')
    g.add_argument('--num_workers', type=int, default=4, help='DataLoader workers per GPU.')
    g.add_argument('--learning_rate', type=float, default=1e-3)
    g.add_argument('--weight_decay', type=float, default=1e-4)
    g.add_argument('--optimizer', type=str, default='adamw',
                   choices=['adam', 'adamw', 'sgd', 'rmsprop', 'adagrad'])
    g.add_argument('--warm_up_epochs', type=int, default=3,
                   help='Linear LR warm-up epochs before cosine annealing.')
    g.add_argument('--usefp16', action='store_true', help='Enable automatic mixed precision.')
    g.add_argument('--start_val_epoch', type=int, default=10,
                   help='First epoch at which validation mAP is computed and the best model is saved.')
    g.add_argument('--tolerances', type=int, nargs='+', default=[0, 1, 2, 4],
                   help='Frame tolerances used for mAP.')

    # ---- Data ----
    g = parser.add_argument_group('data')
    g.add_argument('--clip_len', type=int, default=100, help='Frames per clip.')
    g.add_argument('--crop_dim', type=int, default=224,
                   help='Square crop size. 0 disables cropping and resizes frames to 224x398.')
    g.add_argument('--percent_use', type=float, default=1.0,
                   help='Fraction of training videos that are labeled (split by video). '
                        'In ssl_train.py the remaining videos are used as unlabeled data.')
    g.add_argument('--modality', type=str, default='rgb', choices=['rgb', 'bw', 'flow'])
    g.add_argument('--dilate_len', type=int, default=0, help='Label dilation (frames) during training.')
    g.add_argument('--fg_upsample', type=float, default=None,
                   help='Probability of sampling clips centred on foreground events.')
    g.add_argument('--no_mixup', dest='mixup', action='store_false', help='Disable mixup augmentation.')
    g.add_argument('--fg_weight', type=float, default=5.0,
                   help='Cross-entropy weight of foreground (event) classes.')

    # ---- Semi-supervised (ssl_train.py only) ----
    g = parser.add_argument_group('semi-supervised (ssl_train.py)')
    g.add_argument('--overlapped_number', type=int, default=0,
                   help='Temporal shift (frames) between the teacher and student clips. '
                        '0 = both views cover the same frames.')
    g.add_argument('--supervise_warmup_epochs', type=int, default=5,
                   help='Supervised-only epochs before the unsupervised loss is ramped in.')
    g.add_argument('--ema_decay', type=float, default=0.9995, help='Final EMA decay of the teacher.')
    g.add_argument('--push_loss_weight', type=float, default=0.0,
                   help='Weight of the inter-video push term in TemporalAlignPushLoss.')
    g.add_argument('--proj_head_align', action='store_true',
                   help='Align projection-head features instead of backbone features.')
    g.add_argument('--tma_augment', action='store_true',
                   help='Use the lighter TMA-style strong augmentation for the student view.')

    return parser.parse_args()
