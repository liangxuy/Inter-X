import os

from os.path import join as pjoin

from options.train_options import TrainLenEstOptions

from networks.modules import *
from networks.trainers import LengthEstTrainer, collate_fn
from data.dataset import Text2MotionDatasetHHI
from scripts.motion_process import *
from torch.utils.data import DataLoader
from utils.word_vectorizer import WordVectorizer, POS_enumerator


if __name__ == '__main__':
    parser = TrainLenEstOptions()
    opt = parser.parse()

    opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.log_dir = pjoin('./log', opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    if opt.dataset_name == 't2m':
        opt.data_root = './dataset/HumanML3D'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        dim_pose = 263
    elif opt.dataset_name == 'kit':
        opt.data_root = './dataset/KIT-ML'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        dim_pose = 251
    elif opt.dataset_name == 'hhi':
        opt.data_root = './dataset/inter-x'
        opt.motion_dir = pjoin(opt.data_root, 'motions')
        opt.text_dir = pjoin(opt.data_root, 'texts_processed')
        opt.joints_num = 56
        dim_pose = opt.joints_num * 12
    else:
        raise KeyError('Dataset Does Not Exist')

    dim_word = 300
    dim_pos_ohot = len(POS_enumerator)
    opt.max_motion_length = 150
    opt.max_text_len = 35
    num_classes = 152 // opt.unit_length

    w_vectorizer = WordVectorizer(pjoin(opt.data_root, 'glove'), 'hhi_vab')
    train_split_file = pjoin(opt.data_root, 'splits/train.txt')
    val_split_file = pjoin(opt.data_root, 'splits/val.txt')
    train_motion_file = pjoin(opt.motion_dir, 'train.h5')
    val_motion_file = pjoin(opt.motion_dir, 'val.h5')

    if opt.estimator_mod == 'bigru':
        estimator = MotionLenEstimatorBiGRU(dim_word, dim_pos_ohot, 512, num_classes)
    else:
        raise Exception('Estimator Mode is not Recognized!!!')

    pc_est = sum(param.numel() for param in estimator.parameters())
    print(estimator)
    print("Total parameters of posterior net: {}".format(pc_est))

    trainer = LengthEstTrainer(opt, estimator)

    train_dataset = Text2MotionDatasetHHI(opt, train_split_file, w_vectorizer, train_motion_file)
    val_dataset = Text2MotionDatasetHHI(opt, val_split_file, w_vectorizer, val_motion_file)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                              shuffle=True, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                            shuffle=True, collate_fn=collate_fn, pin_memory=True)

    trainer.train(train_loader, val_loader)