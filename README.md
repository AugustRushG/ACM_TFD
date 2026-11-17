# SSL-PES: Semi-Supervised Learning for Precise Event Spotting

Code for semi-supervised **precise event spotting (PES)**: finding the exact frame at which
short events happen in long videos (for example a ball *hit* or *bounce* in tennis, or a *jump
take-off* in figure skating), when only a fraction of the training videos are labeled.

Official implementation of
**[Temporal Feature Distillation for Label-Efficient Precise Event Spotting in Sports Videos](https://doi.org/10.1145/3767308.3835371)**,
Hao Xu, Xinyu Wei, Samuel Wells and Sunil Aryal. ACM Multimedia 2026 (MM '26).

## Method overview

`ssl_train.py` trains a **teacher-student** model in which the teacher is an exponential moving
average (EMA) of the student:

1. **Supervised branch.** Labeled clips go through the student, trained with per-frame,
   foreground-weighted cross-entropy.
2. **Unsupervised branch (temporal feature distillation).** Two views are drawn from the same
   unlabeled video:
   - The **teacher** sees a weakly augmented clip.
   - The **student** sees a strongly augmented clip. It can be shifted in time
     (`--overlapped_number`) and receives temporally aware augmentations: high-motion patch
     mosaics and masks over short runs of consecutive frames.

   The student's per-frame features are aligned to the teacher's on the overlapping frames
   with a cosine loss (`TemporalAlignPushLoss`).
3. **Teacher update.** The teacher is updated by EMA, with the decay following a cosine schedule
   from 0.99 up to `--ema_decay`. The weight of the unsupervised loss starts after a
   supervised-only warm-up and ramps in over 10 epochs.

The teacher is evaluated with validation mAP every epoch, and the best teacher is kept.

Supported backbones (`--model_choice`):

| Model   | Description | Configs |
|---------|-------------|---------|
| `E2E`   | 2D CNN (RegNetY / ResNet / ConvNeXt) with temporal shift modules (GSM, TSM, GSF, ...) and a GRU head | `models/configs/e2e_configs/` |
| `ASTRM` | E2E-style model with an additional spatio-temporal refinement module | `models/configs/astrm_configs/` |
| `VIT`   | timm ViT-Tiny/Small/Base with **Transformer Gate Shift** (multi-scale gated temporal shift, e.g. `vit_small_msgsm_[1,3,5].json`) and a GRU head | `models/configs/vit_configs/` |

## Repository layout

```text
src/
├── sl_train.py            # supervised baseline training
├── ssl_train.py           # semi-supervised teacher-student training (main method)
├── test.py                # test-set evaluation (F1, mAP @ tolerances, with/without NMS)
├── eval.py                # re-score a saved prediction file against ground truth
├── frames_as_jpg.py       # extract frames from raw videos
├── configs.py             # command-line arguments for training
├── data_process/          # datasets, frame reader, augmentations
├── losses_metrics/        # losses, mAP and F1 metrics, NMS
├── models/                # E2E, ASTRM, VIT, temporal shift modules, and JSON configs
├── training_utils/        # optimizer/scheduler, DDP and EMA helpers
└── utils/                 # logger, json IO
scripts/
├── train_sl.sh            # example: supervised training + test
└── train_ssl.sh           # example: semi-supervised training + test
```

## Installation

Tested with Python 3.11, PyTorch 2.x and CUDA 12.

```bash
conda create -n ssl_pes python=3.11 -y
conda activate ssl_pes
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # choose the build for your CUDA version
pip install -r requirements.txt
```

`environment.yml` contains the full conda environment used for the experiments
(`conda env create -f environment.yml`).

ImageNet-pretrained backbone weights are downloaded automatically by `timm` / `torchvision` the
first time a model is built.

## Data preparation

The code expects a directory of extracted frames and a directory of label files:

```text
<frames_dir>/
└── <video_name>/
    ├── 000000.jpg
    ├── 000001.jpg
    └── ...
<labels_dir>/
├── class.txt      # one event class per line
├── train.json
├── val.json
└── test.json
```

Each `*.json` file is a list with one entry per video (or per video segment):

```json
[
  {
    "video": "usopen_2021_womens_final_33813_34331",
    "num_frames": 518,
    "fps": 25,
    "events": [
      {"frame": 41, "label": "near_court_swing"},
      {"frame": 67, "label": "far_court_bounce"}
    ]
  }
]
```

`video` must match the folder name under `<frames_dir>`, and `label` must be one of the classes in
`class.txt`. Frame indices are 0-based.

The label format matches [E2E-Spot](https://github.com/jhong93/spot), so its label files for
Tennis, Figure Skating (FS-Comp / FS-Perf) and FineDiving can be used directly. To extract frames
for Tennis or Figure Skating from the raw videos:

```bash
cd src
python frames_as_jpg.py tennis /path/to/tennis/videos --labels_dir /path/to/data/tennis \
    -o /path/to/data/tennis/frames --max_height 224
```

Frames are resized to a height of 224. By default training uses random 224×224 crops
(`--crop_dim 224`). With `--crop_dim 0`, frames are instead resized to 224×398 with no cropping.
This option works only with `sl_train.py`: `ssl_train.py` needs a square crop whose size is
divisible by 16.

## Training

All commands are run from `src/`. Training uses `torchrun` for multi-GPU DDP. A single GPU also
works with `python sl_train.py ...` or with `torchrun --nproc_per_node=1`. Outputs are written to
`<output_dir>/<logger_name>/` (default `output_dir` is `../logs`).

### Supervised baseline

```bash
cd src
torchrun --nproc_per_node=2 sl_train.py \
    --frames_dir /path/to/data/tennis/frames \
    --labels_dir /path/to/data/tennis \
    --model_choice E2E \
    --model_config_file models/configs/e2e_configs/e2e_gsm.json \
    --logger_name E2E_GSM_Tennis_SL_p0.1_seed42 \
    --percent_use 0.1 --seed 42 \
    --clip_len 100 --crop_dim 224 \
    --batch_size 4 --learning_rate 1e-3 --num_epochs 50 --start_val_epoch 10 \
    --usefp16
```

`--percent_use` sets the fraction of training **videos** that are used, so labeled and
unlabeled data never share a video. The split is fixed by `--seed`.

### Semi-supervised (teacher-student)

```bash
cd src
torchrun --nproc_per_node=8 ssl_train.py \
    --frames_dir /path/to/data/tennis/frames \
    --labels_dir /path/to/data/tennis \
    --model_choice VIT \
    --model_config_file "models/configs/vit_configs/vit_small_msgsm_[1,3,5].json" \
    --logger_name VITS_MSGSM_Tennis_SSL_p0.1_seed42 \
    --percent_use 0.1 --seed 42 \
    --clip_len 100 --crop_dim 224 \
    --batch_size 4 --learning_rate 1e-4 --num_epochs 50 --start_val_epoch 10 \
    --supervise_warmup_epochs 0 \
    --usefp16
```

With `--percent_use p`, a fraction `p` of the training videos is labeled and the rest serve as
unlabeled data. When `p = 1.0`, the labeled videos are reused as unlabeled data. In multi-GPU runs
the learning rate is scaled by `sqrt(num_gpus)`.

Options specific to `ssl_train.py`:

| Argument | Default | Description |
|----------|---------|-------------|
| `--supervise_warmup_epochs` | 5 | Supervised-only epochs before the unsupervised loss starts (it then ramps in over 10 epochs) |
| `--ema_decay` | 0.9995 | Final EMA decay of the teacher (cosine schedule starting from 0.99) |
| `--overlapped_number` | 0 | Temporal shift (frames) of the student clip relative to the teacher clip; 0 = same frames |
| `--proj_head_align` | off | Align GRU-head features instead of backbone features |
| `--push_loss_weight` | 0.0 | Weight of an optional term that pushes apart embeddings of different videos |
| `--tma_augment` | off | Use the lighter student augmentation (flip, colour, blur) |
| `--resume_path` | none | Resume from `<run>/checkpoint_latest.pth` |

Run `python sl_train.py --help` to see every option.

See `scripts/train_sl.sh` and `scripts/train_ssl.sh` for complete examples that loop over label
ratios and seeds.

## Evaluation

```bash
cd src
torchrun --nproc_per_node=1 test.py \
    --logger_name VITS_MSGSM_Tennis_SSL_p0.1_seed42 \
    --frames_dir /path/to/data/tennis/frames \
    --labels_dir /path/to/data/tennis \
    --model_choice VIT \
    --save
```

`test.py` loads `best_model.pth`, `model_config.json` and `train_config.json` from the run folder,
runs sliding-window inference on `test.json`, and logs the following to
`<run>/<logger_name>_eval.txt`:

- frame-level F1;
- per-class AP and mAP at the training tolerances (default 0, 1, 2 and 4 frames), with and without
  NMS;
- model FLOPs, parameter count and throughput (skip these with `--no_complexity`).

`--save` also writes `pred-test.json` and `pred-test.recall.json.gz`. A saved prediction file can
be re-scored with:

```bash
python eval.py ../logs/<run>/pred-test.recall.json.gz /path/to/data/tennis/test.json -t 0 1 2 4
```

## Pretrained models

Checkpoints trained with 10% labeled videos are available on the Hugging Face Hub at
[AugustRushG123/TFD](https://huggingface.co/AugustRushG123/TFD). All four use the ViT-Small
backbone with Transformer Gate Shift (`--model_choice VIT`).

| Dataset | Folder |
|---------|--------|
| Tennis | `tennis_10pct/` |
| FS-Comp | `fscomp_10pct/` |
| FS-Perf | `fsperf_10pct/` |
| FineDiving | `finediving_10pct/` |

Each folder contains `best_model.pth` (teacher weights), `model_config.json` and
`training_args.json`. That is the layout `test.py` expects, so a downloaded checkpoint can be
evaluated directly:

```bash
pip install -U huggingface_hub
hf download AugustRushG123/TFD --local-dir checkpoints

cd src
python test.py \
    --output_dir ../checkpoints \
    --logger_name tennis_10pct \
    --frames_dir /path/to/data/tennis/frames \
    --labels_dir /path/to/data/tennis \
    --model_choice VIT
```

To download a single model, add `--include "tennis_10pct/*"` to `hf download`. The test log is
written to `checkpoints/tennis_10pct/tennis_10pct_eval.txt`.

## Output folder

```text
logs/<logger_name>/
├── <logger_name>_train.txt   # training log
├── <logger_name>_eval.txt    # test log
├── model_config.json
├── train_config.json         # all command-line arguments
├── best_model.pth            # best checkpoint by validation mAP (teacher weights for SSL)
├── checkpoint_latest.pth     # [ssl_train] student + teacher + optimizer, for resuming
├── pr_curves_tol_*.png       # [test.py] precision-recall curves per tolerance
└── pred-test.*               # [test.py --save] predictions
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{10.1145/3767308.3835371,
  author    = {Xu, Hao and Wei, Xinyu and Wells, Samuel and Aryal, Sunil},
  title     = {Temporal Feature Distillation for Label-Efficient Precise Event Spotting in Sports Videos},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  series    = {MM '26},
  year      = {2026},
  pages     = {12470--12478},
  numpages  = {9},
  location  = {Rio de Janeiro, Brazil},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  isbn      = {9798400722134},
  doi       = {10.1145/3767308.3835371},
  url       = {https://doi.org/10.1145/3767308.3835371},
  keywords  = {precise event spotting, semi-supervised learning, sports video understanding}
}
```

## Acknowledgements

The data pipeline, evaluation code and E2E baseline build on
[E2E-Spot](https://github.com/jhong93/spot) (Hong et al., ECCV 2022). Backbones come from
[timm](https://github.com/huggingface/pytorch-image-models) and torchvision. The temporal shift
modules follow GSM (Sudhakaran et al., CVPR 2020), GSF and TSM.

## License

This project is released under the [MIT License](LICENSE). It includes code adapted from
E2E-Spot (BSD 3-Clause), GSM (BSD 2-Clause), TSM (MIT) and GSF. Their copyright notices and
license terms are listed in [NOTICE](NOTICE).
