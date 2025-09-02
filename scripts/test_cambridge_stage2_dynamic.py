import sys
sys.path.append(".")
from train_net_dynamic import *

cfg=Config('cambridge', data_folder='CamD-1-Subset-2',use_root=True)
cfg.inference_module_name = 'dynamic_cambridge'

cfg.device_list = "0,1"
cfg.use_gpu = True
cfg.use_multi_gpu = True
cfg.training_stage = 2
cfg.train_backbone = False
cfg.test_before_train = True
cfg.test_interval_epoch = 1

# Run Test of Model
cfg.run_test_only = True
cfg.load_stage2model = True
cfg.load_backbone_stage2 = False
cfg.stage2model = 'result/CamD-1-Subset_stage2_T=20_epoch7_84.48%.pth'
cfg.max_video_len = 20
cfg.test_batch_size = 12
cfg.start_epoch = 0
cfg.max_epoch = 0
cfg.exp_note = 'CamD-1-Subset_stage2_res18_Testing'

# vgg16 setup
# cfg.backbone = 'vgg16'
# cfg.stage1_model_path = 'result/basemodel_CamD_res18.pth'
# cfg.out_size = 22, 40
# cfg.emb_features = 512

# res18 setup
cfg.backbone = 'res18'
cfg.stage1_model_path = 'result/basemodel_CamD_res18.pth'
cfg.out_size = 23, 40
cfg.emb_features = 512

cfg.num_actions=10  #number of action categories (we actually have no actions)
cfg.num_activities=10  #number of activity categories

# Added to make suitable for pytorch RoIAlign
cfg.crop_size={'output_size':cfg.crop_size,
        'spatial_scale':1.0/16, # Adjust based on your backbone stride
        'sampling_ratio':2
        }


# Dynamic Inference setup
cfg.group = 1
cfg.stride = 1
cfg.dynamic_sampling = True
cfg.sampling_ratio = [1]
cfg.lite_dim = None #Set at some value less than 1024 to compress embedding dimension
cfg.scale_factor = True
cfg.beta_factor = False
cfg.hierarchical_inference = False
cfg.parallel_inference = False
cfg.num_graph=0  #number of graphs (set to 0 to use multi-layer)
cfg.num_DIM = 1 #number of layers
cfg.ST_kernel_size = [(3, 3)]*cfg.num_DIM
cfg.train_dropout_prob = 0.3

cfg.batch_size = 12
cfg.num_boxes = 6
cfg.num_frames = cfg.max_video_len
cfg.train_learning_rate = 1e-4
# cfg.lr_plan = {11: 3e-5, 21: 1e-5}
# cfg.max_epoch = 60
# cfg.lr_plan = {11: 3e-5, 21: 1e-5}
cfg.lr_plan = {11: 1e-5}

train_net(cfg)
