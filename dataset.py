from volleyball import *
from collective import *
from cambridge_dataset import camD_annotate, CambridgeDataset
from config import Config

import pickle


def return_dataset(cfg):
    # My inclusion
    """
        This function loads the datasets into a Data.dataset module (pytorch suitable).
    """
    if cfg.dataset_name=='volleyball':
        train_anns = volley_read_dataset(cfg.data_path, cfg.train_seqs)
        train_frames = volley_all_frames(train_anns)

        test_anns = volley_read_dataset(cfg.data_path, cfg.test_seqs)
        test_frames = volley_all_frames(test_anns)

        all_anns = {**train_anns, **test_anns}
        all_tracks = pickle.load(open(cfg.data_path + '/tracks_normalized.pkl', 'rb'))


        training_set=VolleyballDataset(all_anns,all_tracks,train_frames,
                                      cfg.data_path,cfg.image_size,cfg.out_size,cfg.inference_module_name,num_before=cfg.num_before,
                                       num_after=cfg.num_after,is_training=True,is_finetune=(cfg.training_stage==1))

        validation_set=VolleyballDataset(all_anns,all_tracks,test_frames,
                                      cfg.data_path,cfg.image_size,cfg.out_size,cfg.inference_module_name,num_before=cfg.num_before,
                                         num_after=cfg.num_after,is_training=False,is_finetune=(cfg.training_stage==1))
    
    elif cfg.dataset_name=='collective':
        train_anns=collective_read_dataset(cfg.data_path, cfg.train_seqs)
        train_frames=collective_all_frames(train_anns)

        test_anns=collective_read_dataset(cfg.data_path, cfg.test_seqs)
        test_frames=collective_all_frames(test_anns)

        training_set=CollectiveDataset(train_anns,train_frames,
                                      cfg.data_path,cfg.image_size,cfg.out_size,
                                      num_frames = cfg.num_frames, is_training=True,is_finetune=(cfg.training_stage==1))

        validation_set=CollectiveDataset(test_anns,test_frames,
                                      cfg.data_path,cfg.image_size,cfg.out_size,
                                      num_frames = cfg.num_frames, is_training=False,is_finetune=(cfg.training_stage==1))
        
    elif cfg.dataset_name=='cambridge':
        train_anns, test_anns = camD_annotate(cfg.data_path, cfg.split_ratio, inference=cfg.run_test_only)
        
        training_set=CambridgeDataset(train_anns,cfg.image_size,is_training=True,is_finetune=(cfg.training_stage==1),
                                       num_boxes=cfg.num_boxes,down_sample=cfg.down_sample,resize=cfg.resize,min_frame_id=cfg.min_frame_id,
                                       ignore_last_n_frames=cfg.ignore_last_n_frames,max_video_len=cfg.max_video_len, use_random_sampling=cfg.use_random_sampling) if not cfg.run_test_only else []

        validation_set=CambridgeDataset(test_anns,cfg.image_size,is_training=False,is_finetune=(cfg.training_stage==1),
                                        num_boxes=cfg.num_boxes,down_sample=cfg.down_sample,resize=cfg.resize,min_frame_id=cfg.min_frame_id,
                                       ignore_last_n_frames=cfg.ignore_last_n_frames,max_video_len=cfg.max_video_len)

        print('Reading dataset finished...')
        print('%d train samples'%len(training_set))
        print('%d test samples'%len(validation_set))
        
        return training_set, validation_set

    else:
        assert False
                                         
    
    print('Reading dataset finished...')
    print('%d train samples'%len(train_frames))
    print('%d test samples'%len(test_frames))
    
    return training_set, validation_set
    

if __name__=="__main__":
    cfg = Config('cambridge')
    training_set, validation_set = return_dataset(cfg)