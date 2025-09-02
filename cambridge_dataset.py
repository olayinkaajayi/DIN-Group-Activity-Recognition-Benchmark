import os
import sys
import json
import random

import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
import cv2


ACTIVITIES = ['B1', 'F1', 'B2', 'F2', 'B3',
              'F3', 'B4', 'F4', 'B5', 'F5']

NUM_ACTIVITIES = 10


def camD_annotate(path, split_ratio=0.7, seed=21, inference=False):
    """
    Read annotations and prepare training/test split
    Returns only file paths and bbox json for lazy loading
    """
    random.seed(seed)
    annotations = []
    gact_to_id = {name: i for i, name in enumerate(ACTIVITIES)}

    for d_ent in os.listdir(path):
        if d_ent[:-5] not in ['RED', 'YELLOW', 'BLACK', 'GREEN', 'BLUE', 'WHITE']:
            continue

        video_path = os.path.join(path, d_ent, f'{d_ent}_left.mp4')
        json_path = os.path.join(path, d_ent, 'bodies.json')
        if not os.path.exists(video_path) or not os.path.exists(json_path):
            continue

        with open(json_path, 'r') as f:
            bbox_data = json.load(f)

        annotations.append({
            'file_name': d_ent,
            'video_path': video_path,
            'group_activity': gact_to_id[d_ent[-5:-3]],
            'bbox_dict': bbox_data
        })

    if inference:
        return [], annotations

    random.shuffle(annotations)
    split_idx = int(len(annotations) * split_ratio)
    return annotations[:split_idx], annotations[split_idx:]


class CambridgeDataset(data.Dataset):
    def __init__(self, ann, image_size=None, num_boxes=6, is_training=True, is_finetune=False, resize=False,
                 down_sample=False, min_frame_id=10, ignore_last_n_frames=10, max_video_len=100, use_random_sampling=False):
        """
        Args:
            ann: List of annotation dicts
            image_size: Desired (H, W) to resize frames to
            num_boxes: Number of boxes per frame (e.g., 6 people)
            is_training: Bool flag
            resize: Whether to resize frames
            down_sample: reduce the frame rate of the videos
        """
        self.anns = ann
        self.image_size = image_size
        self.num_boxes = num_boxes
        self.is_training = is_training
        self.is_finetune = is_finetune
        self.resize = resize
        self.down_sample = down_sample
        self.min_frame_id = min_frame_id
        self.ignore_last_n_frames = ignore_last_n_frames
        self.max_video_len = max_video_len

        # Random frame sampling
        self.use_random_sampling = use_random_sampling

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, index):
        sample = self.anns[index]
        video_path = sample['video_path']
        bboxes_json = sample['bbox_dict']
        group_activity = sample['group_activity']

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        images = []
        bboxes = []
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR (OpenCV) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = frame_rgb.shape[:2] # Needed to rescale bbox

            # Reduce frame rate
            if self.down_sample:
                frame_id, skip = self.skip_frames(frame_id)

                if skip: # we pass over reading the data of the current frame
                    continue

            if self.resize and self.image_size is not None:
                new_h, new_w = self.image_size  # Needed to rescale bbox
                frame_rgb = cv2.resize(frame_rgb, (new_w, new_h)) # (H, W, 3)
            else:
                new_h, new_w = orig_h, orig_w # Needed to rescale bbox

            # Convert to tensor without normalization: [H, W, 3] -> [3, H, W]
            img_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float()
            images.append(img_tensor)

            # Get bounding boxes for this frame
            frame_bboxes = np.zeros((self.num_boxes, 4))
            frame_key = str(frame_id)

            # Scale Bboxes
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
            if frame_key in bboxes_json:
                for entry in bboxes_json[frame_key]['body_list']:

                    rescaled = self.rescaled_bbox(entry["bounding_box_2d"][0], entry["bounding_box_2d"][2], scale_x, scale_y)
                    actor_id = entry["id"] #### decide whether to use id or unique_object_id

                    if actor_id < self.num_boxes:
                        frame_bboxes[actor_id] = np.array(rescaled)
                    else:
                        pass # ignore warning for now
                        # print(f"Warning: actor_id {actor_id} exceeds num_boxes {self.num_boxes}")
                        # sys.stdout.flush()

            bboxes.append(torch.from_numpy(frame_bboxes).float())
            frame_id += 1

        cap.release()

        # Ignore last set of frames where they cluster
        images = self.ignore_ending_frames(images)
        bboxes = self.ignore_ending_frames(bboxes)
        # Downsampling
        indices = self.downsample_frames(images) if not self.use_random_sampling else self.random_temporal_sample(images)
        images = self.return_downsampled_items(images, indices)
        bboxes = self.return_downsampled_items(bboxes, indices)

        images = torch.stack(images)             # [T, 3, H, W]
        bboxes = torch.stack(bboxes)             # [T, num_boxes, 4]
        activities = torch.full((images.size(0),), group_activity, dtype=torch.long)
        
        return images, bboxes, activities
    
    def skip_frames(self, frame_id):
        "skip begining frames of the video."
        if frame_id < self.min_frame_id:
            return (frame_id + 1), True
        else:
            return frame_id, False


    def downsample_frames(self, frames):
        """
        Downsample a list of video frames to a specified target length.
        
        Args:
            frames (list): A list of video frames (e.g., list of numpy arrays or tensors).
            target_frame_count (int): The desired number of frames after downsampling.
            
        Returns:
            list: A uniformly downsampled list of frames.
        """
        if self.is_finetune:
            return self.downsample_frames_random(frames, order=True)
        
        T = len(frames)
        target_frame_count = self.max_video_len
        
        # If the video is already short enough, return as is or pad if needed
        if target_frame_count >= T:
            return frames
        
        # Compute indices for uniform sampling
        indices = [int(i * T / target_frame_count) for i in range(target_frame_count)]
        
        # Ensure last index doesn't exceed bounds
        indices[-1] = min(indices[-1], T - 1)

        return indices
    
    
    def downsample_frames_random(self, frames, order=False):
        """
        Returns sorted random indices for downsampling T frames to max_frames.

        Args:
            T (int): Total number of frames in the video.
            max_frames (int): Maximum number of frames to keep after downsampling.

        Returns:
            list[int]: Sorted list of frame indices to keep.
        """

        T = len(frames)
        max_frames = self.max_video_len

        if T <= max_frames:
            return list(range(T))  # No downsampling needed

        # Randomly sample without replacement, then sort to preserve original order
        indices = random.sample(range(T), max_frames)

        if order:
            indices.sort()

        return indices
    


    def random_temporal_sample(self, frames):
        """
        Randomly sample uniformly distributed frames with jitter.
        Ensures exactly self.max_video_len frames are selected.
        
        Args:
            frames (list): List of frames (numpy arrays or tensors).
        
        Returns:
            list: A list of sampled frames.
        """
        T = len(frames)
        target_frame_count = self.max_video_len

        # If video is short, return as-is or pad
        if target_frame_count >= T:
            return frames

        # Divide video into equal segments
        segment_length = T / target_frame_count
        indices = []

        for i in range(target_frame_count):
            start = int(np.floor(i * segment_length))
            end = int(np.floor((i + 1) * segment_length))

            if start < end:
                # pick a random frame inside this segment
                idx = np.random.randint(start, end)
            else:
                # degenerate case: just pick start
                idx = start
            
            indices.append(idx)

        return indices


    
    def return_downsampled_items(self, ent_list, indices):
        return [ent_list[i] for i in indices]


    def ignore_ending_frames(self, ent_list):
        return ent_list[:-self.ignore_last_n_frames]

    def rescaled_bbox(self, x1y1, x2y2, scale_x, scale_y):
        x1, y1 = x1y1
        x2, y2 = x2y2
        return [
                x1 * scale_x,
                y1 * scale_y,
                x2 * scale_x,
                y2 * scale_y
                ]