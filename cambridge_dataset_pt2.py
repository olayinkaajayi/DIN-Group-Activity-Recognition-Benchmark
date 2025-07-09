import os
import json
import random
from itertools import chain

import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
import cv2


ACTIVITIES = ['B1', 'F1', 'B2', 'F2', 'B3',
              'F3', 'B4', 'F4', 'B5', 'F5']

NUM_ACTIVITIES = 10


def camD_annotate(path, split_ratio=0.7, seed=21):
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

    random.shuffle(annotations)
    split_idx = int(len(annotations) * split_ratio)
    return annotations[:split_idx], annotations[split_idx:]


class CambridgeDataset(data.Dataset):
    def __init__(self, ann, image_size=None, num_boxes=6, is_training=True, is_finetune=False, resize=False):
        """
        Args:
            ann: List of annotation dicts
            image_size: Desired (H, W) to resize frames to
            num_boxes: Number of boxes per frame (e.g., 6 people)
            is_training: Bool flag
            resize: Whether to resize frames
        """
        self.anns = ann
        self.image_size = image_size
        self.num_boxes = num_boxes
        self.is_training = is_training
        self.is_finetune = is_finetune
        self.resize = resize

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

            if self.resize and self.image_size is not None:
                frame_rgb = cv2.resize(frame_rgb, self.image_size[::-1])  # (W, H)

            # Convert to tensor without normalization: [H, W, 3] -> [3, H, W]
            img_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float()
            images.append(img_tensor)

            # Get bounding boxes for this frame
            frame_bboxes = np.zeros((self.num_boxes, 4))
            frame_key = str(frame_id)
            if frame_key in bboxes_json:
                for entry in bboxes_json[frame_key]['body_list']:
                    x1y1 = entry["bounding_box_2d"][0]
                    x2y2 = entry["bounding_box_2d"][2]
                    frame_bboxes[entry["id"]] = np.array(list(chain(*[x1y1, x2y2])))

            bboxes.append(torch.from_numpy(frame_bboxes).float())
            frame_id += 1

        cap.release()

        images = torch.stack(images)             # [T, 3, H, W]
        bboxes = torch.stack(bboxes)             # [T, num_boxes, 4]
        activities = torch.full((images.size(0),), group_activity, dtype=torch.long)
        actions = torch.full((images.size(0),self.num_boxes), group_activity, dtype=torch.long) #this is redundant

        return images, bboxes, actions, activities
