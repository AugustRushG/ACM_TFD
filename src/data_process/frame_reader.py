import os
import random
import torch
import torch.nn as nn
import torchvision

class FrameReader:

    IMG_NAME = '{:06d}.jpg'

    def __init__(self, frame_dir, modality, crop_transform, img_transform,
                 same_transform):
        self._frame_dir = frame_dir
        self._is_flow = modality == 'flow'
        self._crop_transform = crop_transform
        self._img_transform = img_transform
        self._same_transform = same_transform

    def read_frame(self, frame_path):
        # check if frame path exist 
        img = torchvision.io.read_image(frame_path).float() / 255.0
        if self._is_flow:
            # Use GB channels only (flow data)
            img = img[1:, :, :]
        return img

    def load_frames(self, video_name, start, end, pad=False, stride=1,
                    randomize=False):
        rand_crop_state = None
        rand_state_backup = None
        ret = []
        n_pad_start = 0
        n_pad_end = 0
        for frame_num in range(start, end, stride):
            if randomize and stride > 1:
                frame_num += random.randint(0, stride - 1)

            if frame_num < 0:
                n_pad_start += 1
                continue
            
            frame_path = os.path.join(
                self._frame_dir, video_name,
                FrameReader.IMG_NAME.format(frame_num))
   
            try:
                img = self.read_frame(frame_path)
                if self._crop_transform:
                    if self._same_transform:
                        if rand_crop_state is None:
                            rand_crop_state = random.getstate()
                        else:
                            rand_state_backup = random.getstate()
                            random.setstate(rand_crop_state)

                    img = self._crop_transform(img)

                    if rand_state_backup is not None:
                        # Make sure that rand state still advances
                        random.setstate(rand_state_backup)
                        rand_state_backup = None

                if not self._same_transform:
                    img = self._img_transform(img)
                ret.append(img)
            except RuntimeError:
                # print('Missing file!', frame_path)
                n_pad_end += 1
                continue

        if len(ret) == 0:
            channels = 2 if self._is_flow else 3
            dummy_frame = torch.zeros((channels, 224, 224), dtype=torch.float32)
            if self._crop_transform:
                dummy_frame = self._crop_transform(dummy_frame)
            if not self._same_transform:
                dummy_frame = self._img_transform(dummy_frame)
            ret.append(dummy_frame)
            # print(f'Warning: All frames missing for video {video_name}, using dummy data')
            if n_pad_end >0:
                n_pad_end -= 1

        
        # In the multicrop case, the shape is (B, T, C, H, W) if not the shape is (T, C, H, W)
        ret = torch.stack(ret, dim=int(len(ret[0].shape) == 4))
        if self._same_transform and self._img_transform is not None:
            ret = self._img_transform(ret)

        # Always pad start, but only pad end if requested
        if n_pad_start > 0 or (pad and n_pad_end > 0):
            ret = nn.functional.pad(
                ret, (0, 0, 0, 0, 0, 0, n_pad_start, n_pad_end if pad else 0))
        return ret
