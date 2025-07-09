import numpy as np
import json
from itertools import chain

import torch
import torchvision.transforms as transforms
from torch.utils import data

import cv2
from PIL import Image
import os
import random


ACTIVITIES = ['B1', 'F1', 'B2', 'F2', 'B3',
              'F3', 'B4', 'F4', 'B5', 'F5']

NUM_ACTIVITIES = 10

def get_video_as_frames(file):
    """Returns video as individual frames"""
    cap = cv2.VideoCapture(file)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)  # Each frame is a NumPy array (H x W x 3)

    cap.release()

    return frames


def get_bounding_box(filename, frame_id):
    # Open and read the JSON file
    with open(filename, 'r') as file:
        data = json.load(file)

    bbox = np.zeros((6,4))

    for entry in data[str(frame_id)]['body_list']:
        # entry["bounding_box_2d"]: (x1,y1),(x2,y1),(x2,y2),(x1,y2)
        x1y1=entry["bounding_box_2d"][0]
        x2y2=entry["bounding_box_2d"][2]
        bbox[entry["id"]] = np.array(list(chain(*[x1y1,x2y2])))

    return bbox


def camD_annotate(path,split_ratio=0.7,seed=21):
    """
    reading annotations for the given sequence
    """
    random.seed(seed)

    annotations = []

    gact_to_id = {name: i for i, name in enumerate(ACTIVITIES)} # group activity
    data_list = os.listdir(path)
    
    cnt=0
    for d_ent in data_list:
        
        if d_ent[:-5] not in ['RED','YELLOW','BLACK','GREEN','BLUE','WHITE']:
            continue
        
        video = get_video_as_frames(os.path.join(path,d_ent,f'{d_ent}_left.mp4'))
        
        annotations.append( {
                'file_name': d_ent,
                'video_frames': video,
                'group_activity': gact_to_id[d_ent[-5:-3]],
                'bboxes': [get_bounding_box(os.path.join(path,d_ent,'bodies.json'), frame) for frame in range(len(video))]
            } )
        cnt += 1
        
    # Shuffle and split
    random.shuffle(annotations)

    split_idx = int(len(annotations) * split_ratio)

    train_ann = annotations[:split_idx]
    test_ann = annotations[split_idx:]
    
    return train_ann, test_ann




class CambridgeDataset(data.Dataset):
    """
    Characterize Cambridge dataset for pytorch
    """
    def __init__(self,ann,image_size,num_boxes=6,is_training=True,is_finetune=False):
        
        self.image_size=image_size

        self.anns=ann
        
        self.num_boxes=num_boxes
        
        self.is_training=is_training
        self.is_finetune=is_finetune


    def __len__(self):
        """
        Return the total number of samples
        """
        return len(self.anns)
    
    
    def __getitem__(self,index):
        """
        Generate one sample of the dataset
        """
        sample = self.load_samples_sequence(self.anns[index])
        
        return sample
    
    

    def load_samples_sequence(self,sample):
        """
        load samples sequence

        Returns:
            pytorch tensors
        """
        
        images, boxes = [], []
        activities = []
        for i,frame in enumerate(sample['video_frames']):
            # Check to see if the resize if necessary
            img=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img=transforms.functional.resize(img,self.image_size)

            img=np.array(img)

            # H,W,3 -> 3,H,W
            img=img.transpose(2,0,1)
            images.append(img)

            boxes.append(sample['bboxes'][i])
            
            if len(boxes[-1]) != self.num_boxes:
                boxes[-1] = np.vstack([boxes[-1], boxes[-1][:self.num_boxes-len(boxes[-1])]])

            activities.append(sample['group_activity']) # group activity for set of frames

        images = np.stack(images)
        activities = np.array(activities, dtype=np.int32)
        bboxes = np.vstack(boxes).reshape([-1, self.num_boxes, 4])
        
        #convert to pytorch tensor
        images=torch.from_numpy(images).float()
        bboxes=torch.from_numpy(bboxes).float()
        activities=torch.from_numpy(activities).long()

        return images, bboxes,  activities
    

