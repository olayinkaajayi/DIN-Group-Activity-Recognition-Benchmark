import sys
import traceback # Added to save and properly read error

sys.path.append(".")
from train_net import *

try:
    cfg=Config('collective', use_root=True)

    cfg.device_list="0,1" # we have just 2 GPU
    cfg.use_multi_gpu=True # just 2 GPU   
    cfg.training_stage=1
    cfg.train_backbone=True
    cfg.backbone='inv3'# we already defaulted to using InceptionV3 in stage 1. Can change to vgg16 later

    cfg.image_size=480, 720
    cfg.out_size=57,87
    cfg.num_boxes=13
    cfg.num_actions=6
    cfg.num_activities=5
    cfg.num_frames=10

    # Added to make suitable for pytorch RoIAlign
    cfg.crop_size={'output_size':cfg.crop_size,
            'spatial_scale':1.0/16, # Adjust based on your backbone stride
            'sampling_ratio':2
            }

    cfg.batch_size=16
    cfg.test_batch_size=8 
    cfg.train_learning_rate=1e-5
    cfg.train_dropout_prob=0.5
    cfg.weight_decay=1e-2
    cfg.lr_plan={}
    cfg.max_epoch=100

    cfg.exp_note='Collective_stage1'
    train_net(cfg)
    # Added an exception block to catch and save error
except:
    with open("output_exceptions.log", "w") as logfile:
            traceback.print_exc(file=logfile)
    raise