import sys
import traceback # Added to save and properly read error

sys.path.append(".")
from train_net import *

try:
        cfg=Config('cambridge')

        cfg.use_multi_gpu = False
        cfg.device_list="0"
        cfg.training_stage=1
        cfg.stage1_model_path=''
        cfg.train_backbone=True
        cfg.test_before_train = True

        # VGG16
        cfg.backbone = 'vgg16'
        cfg.image_size = 720, 1280
        cfg.out_size = 22, 40
        cfg.emb_features = 512

        cfg.batch_size=8
        cfg.test_batch_size=1
        cfg.num_frames=1

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
        cfg.max_epoch=120
        cfg.set_bn_eval = False
        cfg.actions_weights=[[1., 1., 2., 3., 1., 2., 2., 0.2, 1., 1.]]  # Added 1 extra to make len=10

        cfg.exp_note='Cambridge_stage1'
        train_net(cfg)
except:
    with open("zz_output_exceptions.log", "w") as logfile:
            traceback.print_exc(file=logfile)
    raise