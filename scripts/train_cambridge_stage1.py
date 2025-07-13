import sys
import traceback # Added to save and properly read error

sys.path.append(".")
from train_net import *

cfg=Config('cambridge',use_root=True)

cfg.use_multi_gpu = True
cfg.device_list="0,1,2"
cfg.training_stage=1
cfg.stage1_model_path=''
cfg.train_backbone=True
cfg.test_before_train = True

# VGG16
cfg.backbone = 'res18'
cfg.image_size = 720, 1280
cfg.out_size = 22, 40
cfg.emb_features = 512

cfg.batch_size=8
cfg.test_batch_size=1
cfg.num_frames=1
cfg.num_boxes=6

cfg.num_actions=10  #number of action categories (we actually have no actions)
cfg.num_activities=10  #number of activity categories

# Added to make suitable for pytorch RoIAlign
cfg.crop_size={'output_size':cfg.crop_size,
        'spatial_scale':1.0/16, # Adjust based on your backbone stride
        'sampling_ratio':2
        }

# cfg.train_learning_rate=1e-5
# cfg.lr_plan={}
# cfg.max_epoch=200
cfg.train_learning_rate=1e-4
cfg.lr_plan={30:5e-5, 60:2e-5, 90:1e-5}
cfg.max_epoch=100
cfg.set_bn_eval = False

cfg.exp_note='Cambridge_stage1'
train_net(cfg)
