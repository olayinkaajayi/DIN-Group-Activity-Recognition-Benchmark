import numpy as np
# import skimage.io
# import skimage.transform

import torch
import torchvision.transforms as transforms
from torch.utils import data
import torchvision.models as models

from PIL import Image
import random

import sys
"""
Reference:
https://github.com/cvlab-epfl/social-scene-understanding/blob/master/volleyball.py
"""

ACTIVITIES = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',
              'l_set', 'l-spike', 'l-pass', 'l_winpoint']

NUM_ACTIVITIES = 8

ACTIONS = ['blocking', 'digging', 'falling', 'jumping',
           'moving', 'setting', 'spiking', 'standing',
           'waiting']
NUM_ACTIONS = 9


def volley_read_annotations(path):
    """
    reading annotations for the given sequence
    """
    annotations = {} # This is a dictionary for each unique file ID (data point). Each key is tied to a dictionary containing filename, group activity, actions of each person and bboxes(np.array)

    gact_to_id = {name: i for i, name in enumerate(ACTIVITIES)} # group activity
    act_to_id = {name: i for i, name in enumerate(ACTIONS)}

    with open(path) as f:
        for l in f.readlines():
            values = l[:-1].split(' ')
            file_name = values[0] # file name
            activity = gact_to_id[values[1]] # give integer associated with group activity

            values = values[2:] # get all the remaining entries in the current line of the file
            num_people = len(values) // 5 # Each person has 5 values assigned to them (y,x,w,h,action)

            action_names = values[4::5] # for the same above reason (5 values for each person), we get the action of each person [would not need this for the Cambridge dataset (CamD)]
            actions = [act_to_id[name]
                       for name in action_names] # we give a unique number for each person's action [not needed for CamD]

            def _read_bbox(xywh):
                x, y, w, h = map(int, xywh)
                return y, x, y+h, x+w # I assume x,y is the coordinate of the botom left corner of the bounding box.
            bboxes = np.array([_read_bbox(values[i:i+4]) # we pass the section of the read line that contains bounding box info
                               for i in range(0, 5*num_people, 5)]) # shape of bbox: num_people x 4 (for the 4 corners of the box)

            fid = int(file_name.split('.')[0]) # we get the file name without the .m4v extension (I am assuming the file format)
            annotations[fid] = {
                'file_name': file_name,
                'group_activity': activity,
                'actions': actions,
                'bboxes': bboxes,
            }
    return annotations


def volley_read_dataset(path, seqs):
    data = {}
    for sid in seqs:
        # Each sid is a full video that has been divided already into subsequences with the relevant annotations provided.
        # Note that the arg 'seqs' is the IDs that would be used for training and
        # For CamD we have each video to be self-contained: it has just 1 group action.
        data[sid] = volley_read_annotations(path + '/%d/annotations.txt' % sid)
    return data


def volley_all_frames(data):
    """
        This function gets the sequence ID and 'frame' ID of each of the data points. And this is placed in an list.
        The frame ID is a subsequence in the main video sequence. This subsequence is given a unique ID.
    """
    frames = []
    for sid, anns in data.items():
        for fid, ann in anns.items():
            frames.append((sid, fid))
    return frames


def volley_random_frames(data, num_frames):
    frames = []
    for sid in np.random.choice(list(data.keys()), num_frames):
        fid = int(np.random.choice(list(data[sid]), []))
        frames.append((sid, fid))
    return frames


def volley_frames_around(frame, num_before=5, num_after=4):
    sid, src_fid = frame
    return [(sid, src_fid, fid)
            for fid in range(src_fid-num_before, src_fid+num_after+1)]


def load_samples_sequence(anns,tracks,images_path,frames,image_size,num_boxes=12,):
    """
    load samples of a bath
    
    Returns:
        pytorch tensors
    """
    images, boxes, boxes_idx = [], [], []
    activities, actions = [], []
    for i, (sid, src_fid, fid) in enumerate(frames):
        #img=skimage.io.imread(images_path + '/%d/%d/%d.jpg' % (sid, src_fid, fid))
        #img=skimage.transform.resize(img,(720, 1280),anti_aliasing=True)
        
        img = Image.open(images_path + '/%d/%d/%d.jpg' % (sid, src_fid, fid))
        
        img=transforms.functional.resize(img,image_size)
        img=np.array(img)
        
        # H,W,3 -> 3,H,W
        img=img.transpose(2,0,1)
        images.append(img)

        boxes.append(tracks[(sid, src_fid)][fid])
        actions.append(anns[sid][src_fid]['actions'])
        if len(boxes[-1]) != num_boxes:
          boxes[-1] = np.vstack([boxes[-1], boxes[-1][:num_boxes-len(boxes[-1])]])
          actions[-1] = actions[-1] + actions[-1][:num_boxes-len(actions[-1])]
        boxes_idx.append(i * np.ones(num_boxes, dtype=np.int32))
        activities.append(anns[sid][src_fid]['group_activity'])


    images = np.stack(images)
    activities = np.array(activities, dtype=np.int32)
    bboxes = np.vstack(boxes).reshape([-1, num_boxes, 4])
    bboxes_idx = np.hstack(boxes_idx).reshape([-1, num_boxes])
    actions = np.hstack(actions).reshape([-1, num_boxes])
    
    #convert to pytorch tensor
    images=torch.from_numpy(images).float()
    bboxes=torch.from_numpy(bboxes).float()
    bboxes_idx=torch.from_numpy(bboxes_idx).int()
    actions=torch.from_numpy(actions).long()
    activities=torch.from_numpy(activities).long()

    return images, bboxes, bboxes_idx, actions, activities


class VolleyballDataset(data.Dataset):
    """
    Characterize volleyball dataset for pytorch
    """
    def __init__(self,anns,tracks,frames,images_path,image_size,feature_size,inference_module_name,num_boxes=12,num_before=4,num_after=4,is_training=True,is_finetune=False):
        self.anns=anns
        self.tracks=tracks
        self.frames=frames
        self.images_path=images_path
        self.image_size=image_size
        self.feature_size=feature_size
        self.inference_module_name = inference_module_name
        
        self.num_boxes=num_boxes
        self.num_before=num_before
        self.num_after=num_after
        
        self.is_training=is_training
        self.is_finetune=is_finetune

        # self.frames_seq = np.empty((1337, 2), dtype = np.int)
        # self.flag = 0

    def __len__(self):
        """
        Return the total number of samples
        """
        return len(self.frames)
    
    def __getitem__(self,index):
        """
        Generate one sample of the dataset
        """
        # Save frame sequences
        # self.frames_seq[self.flag] = self.frames[index]# [0], self.frames[index][1]
        # if self.flag == 1336:
        #     save_seq = self.frames_seq
        #     np.savetxt('vis/frames_seq.txt', save_seq)
        # self.flag += 1

        select_frames = self.volley_frames_sample(self.frames[index])
        sample = self.load_samples_sequence(select_frames)
        
        return sample
    
    def volley_frames_sample(self,frame):
        sid, src_fid = frame
        
        if self.is_finetune:
            # This part is relevant when finetunning the backbone
            if self.is_training:
                fid=random.randint(src_fid-self.num_before, src_fid+self.num_after)
                return [(sid, src_fid, fid)]
            else:
                return [(sid, src_fid, fid)
                        for fid in range(src_fid-self.num_before, src_fid+self.num_after+1)]
        else:
            # if self.is_training:
            #     sample_frames=random.sample(range(src_fid-self.num_before, src_fid+self.num_after+1), 3)
            #     return [(sid, src_fid, fid)
            #             for fid in sample_frames]
            # else:
            #     return [(sid, src_fid, fid)
            #             for fid in  [src_fid-3,src_fid,src_fid+3, src_fid-4,src_fid-1,src_fid+2, src_fid-2,src_fid+1,src_fid+4 ]]
            if self.inference_module_name == 'arg_volleyball':
                if self.is_training:
                    sample_frames=random.sample(range(src_fid-self.num_before, src_fid+self.num_after+1), 3)
                    return [(sid, src_fid, fid)
                            for fid in sample_frames]
                else:
                    return [(sid, src_fid, fid)
                            for fid in  [src_fid-3,src_fid,src_fid+3, src_fid-4,src_fid-1,src_fid+2, src_fid-2,src_fid+1,src_fid+4 ]]
            else:
                # This is the part we are interested in for the DIN model
                # Why do we need 4 frames before and 5 frames after???
                # Ans: these are the frames where the specific group action takes place.
                if self.is_training:
                    return [(sid, src_fid, fid)  for fid in range(src_fid-self.num_before, src_fid+self.num_after+1)]
                else:
                    return [(sid, src_fid, fid) for fid in range(src_fid - self.num_before, src_fid + self.num_after + 1)]



    def load_samples_sequence(self,select_frames):
        """
        load samples sequence

        Returns:
            pytorch tensors
        """
        
        OH, OW=self.feature_size
        
        images, boxes = [], []
        activities, actions = [], []
        for i, (sid, src_fid, fid) in enumerate(select_frames):

            img = Image.open(self.images_path + '/%d/%d/%d.jpg' % (sid, src_fid, fid))

            img=transforms.functional.resize(img,self.image_size)
            img=np.array(img)

            # H,W,3 -> 3,H,W
            img=img.transpose(2,0,1)
            images.append(img)

            temp_boxes=np.ones_like(self.tracks[(sid, src_fid)][fid])
            for i,track in enumerate(self.tracks[(sid, src_fid)][fid]):
                # Run through the bboxes of all 12 players
                y1,x1,y2,x2 = track
                w1,h1,w2,h2 = x1*OW, y1*OH, x2*OW, y2*OH  # scale the bboxes to match image resolution
                temp_boxes[i]=np.array([w1,h1,w2,h2])
            
            boxes.append(temp_boxes)
            
            
            actions.append(self.anns[sid][src_fid]['actions'])
            
            if len(boxes[-1]) != self.num_boxes:
                boxes[-1] = np.vstack([boxes[-1], boxes[-1][:self.num_boxes-len(boxes[-1])]])
                actions[-1] = actions[-1] + actions[-1][:self.num_boxes-len(actions[-1])]
            activities.append(self.anns[sid][src_fid]['group_activity']) # group activity for set of frames

        images = np.stack(images)
        activities = np.array(activities, dtype=np.int32)
        bboxes = np.vstack(boxes).reshape([-1, self.num_boxes, 4])
        actions = np.hstack(actions).reshape([-1, self.num_boxes])
        

        #convert to pytorch tensor
        images=torch.from_numpy(images).float()
        bboxes=torch.from_numpy(bboxes).float()
        actions=torch.from_numpy(actions).long()
        activities=torch.from_numpy(activities).long()

        return images, bboxes,  actions, activities
    
