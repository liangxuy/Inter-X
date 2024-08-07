import os
import pickle
import h5py
import torch
import joblib
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
from tqdm import tqdm
import utils.rotation_conversions as geometry
from ipdb import set_trace

SRC = 'xxx/Inter-X_Open_Source/motions'
DEST_H5 = './inter-x.h5'
neutral_bm_path = '../body_models/smplx/SMPLX_NEUTRAL.npz'

comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from human_body_prior.body_model.body_model import BodyModel

num_betas = 10
downsample = 4

neutral_bm = BodyModel(bm_fname=neutral_bm_path, num_betas=num_betas).to(comp_device)


def parse_motion_file(motion_file):
    """
    body_pose (896, 21, 3)
    lhand_pose (896, 15, 3)
    rhand_pose (896, 15, 3)
    jaw_pose (896, 3)
    leye_pose (896, 3)
    reye_pose (896, 3)
    betas (1, 10)
    global_orient (896, 3)
    transl (896, 3)
    gender ()
    """
    data = np.load(motion_file, allow_pickle=True)
    body_pose = data['pose_body'][::downsample]
    left_hand_pose = data['pose_lhand'][::downsample]
    right_hand_pose = data['pose_rhand'][::downsample]
    root_transl = data['trans'][::downsample]
    global_orient = data['root_orient'][::downsample]
    frame_number = body_pose.shape[0]
    jaw_pose = np.zeros((frame_number, 3))
    leye_pose = np.zeros((frame_number, 3))
    reye_pose = np.zeros((frame_number, 3))
    bm = neutral_bm # gender is ignored for training/evaluation

    pose_seq = []
    with torch.no_grad():
        for fId in range(0, frame_number):
            root_orient = torch.Tensor(global_orient[fId:fId+1,:]).to(comp_device)
            pose_body = torch.Tensor(body_pose[fId:fId+1,:].reshape([1, -1])).to(comp_device)
            hand_pose = np.concatenate([left_hand_pose[fId:fId+1, :], right_hand_pose[fId:fId+1, :]], axis=1).reshape([1,-1])
            pose_hand = torch.Tensor(hand_pose).to(comp_device)
            trans = torch.Tensor(root_transl[fId:fId+1]).to(comp_device)
            body = bm(pose_body=pose_body, pose_hand=pose_hand, root_orient=root_orient) # betas is ignored for training/evaluation
            joint_loc = body.Jtr[0] + trans
            pose_seq.append(joint_loc.unsqueeze(0))
    pose_seq = torch.cat(pose_seq, dim=0)
    pose_seq_np = pose_seq.detach().cpu().numpy()
    root_transl[:,1] = root_transl[:,1] - np.min(pose_seq_np[:,:,1]) # min-value align to the ground
    final_pose = np.concatenate((global_orient[:,None], body_pose, jaw_pose[:,None], leye_pose[:,None], reye_pose[:,None], left_hand_pose, right_hand_pose, root_transl[:, None]), axis=1)

    return final_pose # [T, 56, 3]


def normalize_transl(motion_p1, motion_p2):
    base = motion_p1[0,-1]
    motion_p2[:, -1] -= base
    motion_p1[:, -1] -= base
    return motion_p1, motion_p2


if __name__ == '__main__':
    fw = h5py.File(DEST_H5, 'w')
    # Load all samples
    pbar = tqdm(sorted(os.listdir(SRC)))
    for split_id in pbar:
        idx = split_id.find('A')
        cur_action_id = int(split_id[idx+1:idx+4])
        motion_files = sorted(os.listdir(os.path.join(SRC, split_id)))
        P1_data = parse_motion_file(os.path.join(SRC, split_id, motion_files[0]))
        P2_data = parse_motion_file(os.path.join(SRC, split_id, motion_files[1]))
        P1_data, P2_data = normalize_transl(P1_data, P2_data)
        data = np.concatenate([P1_data, P2_data], axis=-1)
        fw.create_dataset(split_id, data=data, dtype='f4')
    fw.close()