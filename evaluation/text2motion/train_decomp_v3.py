import os

from os.path import join as pjoin

import utils.paramUtil as paramUtil
import utils.paramUtil_hhi as paramUtil_hhi
from options.train_options import TrainDecompOptions
from utils.plot_script import *

from networks.modules import *
from networks.trainers import DecompTrainerV3
from data.dataset import MotionDatasetV2HHI
from scripts.motion_process import *
from torch.utils.data import DataLoader
from utils.word_vectorizer import WordVectorizer, POS_enumerator

def plot_t2m(data, save_dir):
    data = train_dataset.inv_transform(data)
    for i in range(len(data)):
        joint_data = data[i]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        save_path = pjoin(save_dir, '%02d.gif'%(i))
        plot_3d_motion(save_path, kinematic_chain, joint, title="None", fps=fps, radius=radius)


if __name__ == '__main__':
    parser = TrainDecompOptions()
    opt = parser.parse()

    opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)
    if opt.gpu_id != -1:
        # self.opt.gpu_id = int(self.opt.gpu_id)
        torch.cuda.set_device(opt.gpu_id)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log', opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    if opt.dataset_name == 't2m':
        opt.data_root = './dataset/HumanML3D'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.max_motion_length = 196
        dim_pose = 263
        radius = 4
        fps = 20
        kinematic_chain = paramUtil.t2m_kinematic_chain
    elif opt.dataset_name == 'kit':
        opt.data_root = './dataset/KIT-ML'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        radius = 240 * 8
        fps = 12.5
        dim_pose = 251
        opt.max_motion_length = 196
        kinematic_chain = paramUtil.kit_kinematic_chain
    elif opt.dataset_name == 'hhi':
        opt.data_root = './dataset/inter-x'
        opt.motion_dir = pjoin(opt.data_root, 'motions')
        opt.text_dir = pjoin(opt.data_root, 'texts_processed')
        opt.joints_num = 56
        opt.max_motion_length = 150
        dim_pose = opt.joints_num * 12 # two persons, rot6d
        radius = 4
        fps = 30
        kinematic_chain = paramUtil_hhi.hhi_kinematic_chain
    else:
        raise KeyError('Dataset Does Not Exist')

    w_vectorizer = WordVectorizer(pjoin(opt.data_root, 'glove'), 'hhi_vab')
    train_split_file = pjoin(opt.data_root, 'splits/train.txt')
    val_split_file = pjoin(opt.data_root, 'splits/val.txt')
    train_motion_file = pjoin(opt.motion_dir, 'train.h5')
    val_motion_file = pjoin(opt.motion_dir, 'val.h5')

    movement_enc = MovementConvEncoder(dim_pose, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    movement_dec = MovementConvDecoder(opt.dim_movement_latent, opt.dim_movement_dec_hidden, dim_pose)

    all_params = 0
    pc_mov_enc = sum(param.numel() for param in movement_enc.parameters())
    print(movement_enc)
    print("Total parameters of prior net: {}".format(pc_mov_enc))
    all_params += pc_mov_enc

    pc_mov_dec = sum(param.numel() for param in movement_dec.parameters())
    print(movement_dec)
    print("Total parameters of posterior net: {}".format(pc_mov_dec))
    all_params += pc_mov_dec

    trainer = DecompTrainerV3(opt, movement_enc, movement_dec)

    train_dataset = MotionDatasetV2HHI(opt, train_split_file, train_motion_file)
    val_dataset = MotionDatasetV2HHI(opt, val_split_file, val_motion_file)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                            shuffle=True, pin_memory=True)

    trainer.train(train_loader, val_loader, plot_t2m)
