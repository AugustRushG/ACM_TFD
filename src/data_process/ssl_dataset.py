
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from .frame_reader import FrameReader
from .transform import get_ssl_transform, _get_deferred_rgb_transform, _get_deferred_bw_transform, _get_img_transforms, get_wacv_transform
from .variables import DEFAULT_PAD_LEN


class SSLLabeledActionSpotDataset(Dataset):

    def __init__(
            self,
            classes,                    # dict of class names to idx
            labels,                     # list of label dicts
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

        labels = np.zeros(self._clip_len, np.int64)
        for event in video_meta['events']:
            event_frame = event['frame']

            # Index of event in label array
            label_idx = (event_frame - base_idx) // self._stride
            if (label_idx >= -self._dilate_len
                and label_idx < self._clip_len + self._dilate_len
            ):
                label = self._class_dict[event['label']]
                
                for i in range(
                    max(0, label_idx - self._dilate_len),
                    min(self._clip_len, label_idx + self._dilate_len + 1)
                ):
                    labels[i] = label

        frames = self._frame_reader.load_frames(
            video_meta['video'], base_idx,
            base_idx + self._clip_len * self._stride, pad=True,
            stride=self._stride, randomize=not self._is_eval)

        return {'frame': frames, 'contains_event': int((labels > 0).sum()),
                'label': labels}

    def __getitem__(self, unused):
        ret = self._get_one()
        
        if self._mixup:
            mix = self._get_one()    # Sample another clip
            l = random.betavariate(0.2, 0.2)
            label_dist = np.zeros((self._clip_len, len(self._class_dict) + 1))
            label_dist[range(self._clip_len), ret['label']] = l
            label_dist[range(self._clip_len), mix['label']] += 1. - l

            if self._gpu_transform is None:
                ret['frame'] = l * ret['frame'] + (1. - l) * mix['frame']
            else:
                ret['mix_frame'] = mix['frame']
                ret['mix_weight'] = l

            ret['contains_event'] = max(
                ret['contains_event'], mix['contains_event'])
            ret['label'] = label_dist
         

        return ret

    def __len__(self):
        return self._dataset_len


class SSLUnlabeledOverlappedActionSpotDataset(Dataset):

    def __init__(
            self,
            labels,                 # path to label json
            frame_dir,                  # path to frames
            modality,                   # [rgb, bw, flow]
            clip_len,
            dataset_len,                # Number of clips
            overlapped_number=0,
            crop_dim=None,
            stride=1,                   # Downsample frame rate
            same_transform=True,        # Apply the same random augmentation to
                                        # each frame in a clip
            use_random_aug=False,
            pad_len=DEFAULT_PAD_LEN,    # Number of frames to pad the start and end of videos
            wacv_style_augmentation=False, # Whether to use the WACV-style augmentation 
    ):
        self._labels = labels
        self.overlapped_number = overlapped_number

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


        if wacv_style_augmentation:
            crop_transform, weak_img_transform, strong_img_transform = get_wacv_transform(crop_dim=crop_dim, use_randaugment=use_random_aug)
        else:
            crop_transform, weak_img_transform, strong_img_transform = get_ssl_transform(crop_dim=crop_dim, use_randaugment=use_random_aug)

        self.crop_transform = crop_transform
        self.weak_img_transform = weak_img_transform
        self.strong_img_transform = strong_img_transform

        self._frame_reader = FrameReader(frame_dir, modality, crop_transform, None, same_transform)

    def load_frame_cpu(self, batch, device):
        weak_framea = self._load_frame_cpu(self.weak_img_transform, batch['frames_a'])
        strong_frameb = self._load_frame_cpu(self.strong_img_transform, batch['frames_b'])

        # Now move once to GPU
        weak_framea = weak_framea.to(device, non_blocking=True)
        strong_frameb = strong_frameb.to(device, non_blocking=True)

        return weak_framea, strong_frameb
    
    def _load_frame_cpu(self, transform, frames):
        """
        frames: CPU tensor [T, C, H, W]
        """
        with torch.no_grad():
            # Apply transform per frame on CPU
            out = []
            for i in range(frames.shape[0]):
                out.append(transform(frames[i]))  # CPU op
            frames = torch.stack(out, dim=0)

        return frames

    def load_frame_gpu(self, batch, device):
        weak_framea = self._load_frame_deferred(self.weak_img_transform, batch['frames_a'], device)
        strong_frameb = self._load_frame_deferred(self.strong_img_transform, batch['frames_b'], device)
        # strong_framea = self._load_frame_deferred(self.strong_img_transform, batch['frames_a'], device)
        # weak_frameb = self._load_frame_deferred(self.weak_img_transform, batch['frames_b'], device)
        return weak_framea, strong_frameb

        
    def _load_frame_deferred(self, transform, frames, device):
        frames = frames.to(device)
        with torch.no_grad():
            for i in range(frames.shape[0]):
                frames[i] = transform(frames[i])

        return frames


    def _sample_uniform(self):
        video_meta = random.choices(self._labels, weights=self._weights_by_length)[0]

        video_len = video_meta['num_frames']
        base_idx = -self._pad_len * self._stride + random.randint(
            0, max(0, video_len - 1 + (2 * self._pad_len - self._clip_len) * self._stride))
        return video_meta, base_idx


    def _get_one(self):
        video_meta, base_idx = self._sample_uniform()
        # overlapped_number = random.randint(self._clip_len//4,self._clip_len//2) # 25% to 50% overlap
        if self.overlapped_number == 0:  # means complete overlap 
            second_video_base_idx = base_idx
        else: # partial overlap
            second_video_base_idx = base_idx + self._clip_len * self._stride - self.overlapped_number * self._stride # ensure overlapping
        second_video_end_idx = second_video_base_idx + self._clip_len * self._stride

        frames_a = self._frame_reader.load_frames(
            video_meta['video'], base_idx,
            base_idx + self._clip_len * self._stride, pad=True,
            stride=self._stride, randomize=False)
        
        frames_b = self._frame_reader.load_frames(
            video_meta['video'], second_video_base_idx,
            second_video_end_idx, pad=True,
            stride=self._stride, randomize=False)


        return {'frames_a': frames_a, 'frames_b': frames_b}

    def __getitem__(self, unused):
        ret = self._get_one()
         
        return ret

    def __len__(self):
        return self._dataset_len


# --- Save first 5 frames individually ---
