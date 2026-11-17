from torch.utils.data import Dataset
import random
import numpy as np
import torch
import copy
from .transform import  _get_img_transforms, _get_deferred_rgb_transform, _get_deferred_bw_transform
from .utils import load_json
from .frame_reader import FrameReader
from .variables import DEFAULT_PAD_LEN

class ActionSpotDataset(Dataset):

    def __init__(
            self,
            classes,                    # dict of class names to idx
            labels,                     # labels already loaded from json
            frame_dir,                  # path to frames
            modality,                   # [rgb, bw, flow]
            clip_len,
            dataset_len,                # Number of clips
            is_eval=True,               # Disable random augmentation
            crop_dim=None,
            stride=1,                   # Downsample frame rate
            same_transform=True,        # Apply the same random augmentation to
                                        # each frame in a clip
            dilate_len=0,               # Dilate ground truth labels
            mixup=False,
            pad_len=DEFAULT_PAD_LEN,    # Number of frames to pad the start
                                        # and end of videos
            fg_upsample=-1,             # Sample foreground explicitly
    ):
        self._labels = labels
        self._class_dict = classes
        self._video_idxs = {x['video']: i for i, x in enumerate(self._labels)}

        # Sample videos weighted by their length
        num_frames = [v['num_frames'] for v in self._labels]
        self._weights_by_length = np.array(num_frames) / np.sum(num_frames)

        self._clip_len = clip_len
        assert clip_len > 0
        self._stride = stride
        assert stride > 0
        self._dataset_len = dataset_len
        assert dataset_len > 0
        self._pad_len = pad_len
        assert pad_len >= 0
        self._is_eval = is_eval

        # Label modifications
        self._dilate_len = dilate_len
        self._fg_upsample = fg_upsample

        # Sample based on foreground labels
        if self._fg_upsample > 0:
            self._flat_labels = []
            for i, x in enumerate(self._labels):
                for event in x['events']:
                    if event['frame'] < x['num_frames']:
                        self._flat_labels.append((i, event['frame']))

        self._mixup = mixup

        # Try to do defer the latter half of the transforms to the GPU
        self._gpu_transform = None
        if not is_eval and same_transform:
            if modality == 'rgb':
                print('=> Deferring some RGB transforms to the GPU!')
                self._gpu_transform = _get_deferred_rgb_transform()
            elif modality == 'bw':
                print('=> Deferring some BW transforms to the GPU!')
                self._gpu_transform = _get_deferred_bw_transform()

        crop_transform, img_transform = _get_img_transforms(
            is_eval, crop_dim, modality, same_transform,
            defer_transform=self._gpu_transform is not None)

        self._frame_reader = FrameReader(
            frame_dir, modality, crop_transform, img_transform, same_transform)
        
    def _load_frame_deferred(self, gpu_transform, batch, device):
        frame = batch['frame'].to(device)
        with torch.no_grad():
            for i in range(frame.shape[0]):
                frame[i] = gpu_transform(frame[i])

            if 'mix_weight' in batch:
                weight = batch['mix_weight'].to(device)
                frame *= weight[:, None, None, None, None]

                frame_mix = batch['mix_frame']
                for i in range(frame.shape[0]):
                    frame[i] += (1. - weight[i]) * gpu_transform(
                        frame_mix[i].to(device))
        return frame

    def load_frame_gpu(self, batch, device):
        if self._gpu_transform is None:
            frame = batch['frame'].to(device)
        else:
            frame = self._load_frame_deferred(self._gpu_transform, batch, device)
        return frame

    def _sample_uniform(self):
        video_meta = random.choices(
            self._labels, weights=self._weights_by_length)[0]

        video_len = video_meta['num_frames']
        base_idx = -self._pad_len * self._stride + random.randint(
            0, max(0, video_len - 1
                       + (2 * self._pad_len - self._clip_len) * self._stride))
        return video_meta, base_idx

    def _sample_foreground(self):
        video_idx, frame_idx = random.choices(self._flat_labels)[0]
        video_meta = self._labels[video_idx]
        video_len = video_meta['num_frames']

        lower_bound = max(
            -self._pad_len * self._stride,
            frame_idx - self._clip_len * self._stride + 1)
        upper_bound = min(
            video_len - 1 + (self._pad_len - self._clip_len) * self._stride,
            frame_idx)

        base_idx = random.randint(lower_bound, upper_bound) \
            if upper_bound > lower_bound else lower_bound

        assert base_idx <= frame_idx
        assert base_idx + self._clip_len > frame_idx
        return video_meta, base_idx

    def _get_one(self):
            if self._fg_upsample > 0 and random.random() >= self._fg_upsample:
                video_meta, base_idx = self._sample_foreground()
            else:
                video_meta, base_idx = self._sample_uniform()

            # Initialize labels as 2D float array: (Time, Num_Classes + Background)
            num_classes = len(self._class_dict) + 1
            labels = np.zeros((self._clip_len, num_classes), dtype=np.float32)
            
            # Set background to 1.0 initially
            labels[:, 0] = 1.0

            for event in video_meta['events']:
                event_frame = event['frame']
                label_idx = (event_frame - base_idx) // self._stride
                
                # Check if event is within the clip window (accounting for dilation)
                if (label_idx >= -self._dilate_len and label_idx < self._clip_len + self._dilate_len):
                    
                    label_class = self._class_dict[event['label']]
                    
                    if self._dilate_len > 0:
                        # --- OPTION A: Gaussian Distribution ---
                        sigma = self._dilate_len / 2.0 
                        start = max(0, int(label_idx - self._dilate_len))
                        end = min(self._clip_len, int(label_idx + self._dilate_len + 1))
                        
                        if end > start:
                            x = np.arange(start, end)
                            gaussian = np.exp(-0.5 * ((x - label_idx) / sigma) ** 2)
                            labels[x, label_class] = np.maximum(labels[x, label_class], gaussian)
                    
                    else:
                        # --- OPTION B: Hard Label (No Dilation) ---
                        # Since dilate_len is 0, the outer `if` guarantees 0 <= label_idx < clip_len
                        # We cast to int just to be safe for indexing
                        idx = int(label_idx)
                        labels[idx, label_class] = 1.0

            # Re-normalize Background
            # Works for both Gaussian (continuous) and Hard (binary) labels
            labels[:, 0] = np.maximum(0.0, 1.0 - np.sum(labels[:, 1:], axis=1))

            frames = self._frame_reader.load_frames(
                video_meta['video'], base_idx,
                base_idx + self._clip_len * self._stride, pad=True,
                stride=self._stride, randomize=not self._is_eval)

            return {'frame': frames, 
                    'contains_event': int(np.sum(labels[:, 1:]) > 0.01),
                    'label': labels}

    def __getitem__(self, unused):
        ret = self._get_one()
        
        if self._mixup:
            mix = self._get_one()    # Sample another clip
            l = random.betavariate(0.2, 0.2)
            
            # Mixup Logic Updated:
            # Since 'label' is already a distribution, we just do a weighted average
            ret['label'] = l * ret['label'] + (1. - l) * mix['label']

            if self._gpu_transform is None:
                ret['frame'] = l * ret['frame'] + (1. - l) * mix['frame']
            else:
                ret['mix_frame'] = mix['frame']
                ret['mix_weight'] = l

            ret['contains_event'] = max(
                ret['contains_event'], mix['contains_event'])
        
        return ret
    
    def __len__(self):
        return self._dataset_len


class ActionSpotVideoDataset(Dataset):

    def __init__(
            self,
            classes,
            label_file,
            frame_dir,
            modality,
            clip_len,
            overlap_len=0,
            crop_dim=None,
            stride=1,
            pad_len=DEFAULT_PAD_LEN,
            flip=False,
            multi_crop=False,
            skip_partial_end=True
    ):
        self._src_file = label_file
        self._labels = load_json(label_file)
        self._class_dict = classes
        self._video_idxs = {x['video']: i for i, x in enumerate(self._labels)}
        self._clip_len = clip_len
        self._stride = stride

        crop_transform, img_transform = _get_img_transforms(
            is_eval=True, crop_dim=crop_dim, modality=modality, same_transform=True, multi_crop=multi_crop)

        # No need to enforce same_transform since the transforms are
        # deterministic
        self._frame_reader = FrameReader(
            frame_dir, modality, crop_transform, img_transform, False)

        self._flip = flip
        self._multi_crop = multi_crop

        self._clips = []
        for l in self._labels:
            has_clip = False
            for i in range(
                -pad_len * self._stride,
                max(0, l['num_frames'] - (overlap_len * stride)
                        * int(skip_partial_end)), \
                # Need to ensure that all clips have at least one frame
                (clip_len - overlap_len) * self._stride
            ):
                has_clip = True
                self._clips.append((l['video'], i))
            assert has_clip, l

    def __len__(self):
        return len(self._clips)

    def __getitem__(self, idx):
        video_name, start = self._clips[idx]
        
        frames = self._frame_reader.load_frames(
            video_name, start, start + self._clip_len * self._stride, pad=True,
            stride=self._stride)

        if self._flip:
            frames = torch.stack((frames, frames.flip(-1)), dim=0)

        return {'video': video_name, 'start': start // self._stride,
                'frame': frames}

    def get_labels(self, video):
        meta = self._labels[self._video_idxs[video]]
        num_frames = meta['num_frames']
        num_labels = num_frames // self._stride
        if num_frames % self._stride != 0:
            num_labels += 1
        labels = np.zeros(num_labels, int)
        for event in meta['events']:
            frame = event['frame']
            if frame < num_frames:
                labels[frame // self._stride] = self._class_dict[event['label']]
            else:
                print('Warning: {} >= {} is past the end {}'.format(
                    frame, num_frames, meta['video']))
        return labels

    @property
    def augment(self):
        return self._flip or self._multi_crop

    @property
    def videos(self):
        return sorted([
            (v['video'], v['num_frames'] // self._stride,
             v['fps'] / self._stride) for v in self._labels])

    @property
    def labels(self):
        assert self._stride > 0
        if self._stride == 1:
            return self._labels
        else:
            labels = []
            for x in self._labels:
                x_copy = copy.deepcopy(x)
                x_copy['fps'] /= self._stride
                x_copy['num_frames'] //= self._stride
                for e in x_copy['events']:
                    e['frame'] //= self._stride
                labels.append(x_copy)
            return labels

    def print_info(self):
        num_frames = sum([x['num_frames'] for x in self._labels])
        num_events = sum([len(x['events']) for x in self._labels])
        print('{} : {} videos, {} frames ({} stride), {:0.5f}% non-bg'.format(
            self._src_file, len(self._labels), num_frames, self._stride,
            num_events / num_frames * 100))
