# The implementation is based on https://github.com/eth-ait/aitviewer
import os
import platform
import time

import glfw
import imgui
import numpy as np
import pyperclip
import trimesh
from aitviewer.configuration import CONFIG as C
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.skeletons import Skeletons
from aitviewer.renderables.plane import Plane
from aitviewer.viewer import Viewer
from scipy.spatial.transform import Rotation as R

from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence


glfw.init()
primary_monitor = glfw.get_primary_monitor()
mode = glfw.get_video_mode(primary_monitor)
width = mode.size.width
height = mode.size.height

C.update_conf({'window_width': width*0.9, 'window_height': height*0.9})
C.update_conf({'smplx_models':'./body_models'})


class SMPLX_Viewer(Viewer):
    title='Inter-X Viewer' 

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.gui_controls.update(
            {
                'show_text':self.gui_show_text
            }
        )
        self._set_prev_record=self.wnd.keys.UP
        self._set_next_record=self.wnd.keys.DOWN

        # reset
        self.reset_for_interx()
        self.load_one_sequence()

    def reset_for_interx(self):
        
        self.text_val = ''

        self.clip_folder = './data/'
        self.text_folder = './texts'

        self.label_npy_list = []
        self.get_label_file_list()
        self.total_tasks = len(self.label_npy_list)

        self.label_pid = 0
        self.go_to_idx = 0

    def key_event(self, key, action, modifiers):
        if action==self.wnd.keys.ACTION_PRESS:
            if key==self._set_prev_record:
                self.set_prev_record()
            elif key==self._set_next_record:
                self.set_next_record()
            else:
                return super().key_event(key, action, modifiers)
        else:
            return super().key_event(key, action, modifiers)

    def gui_show_text(self):
        imgui.set_next_window_position(self.window_size[0] * 0.6, self.window_size[1]*0.25, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(self.window_size[0] * 0.35, self.window_size[1]*0.4, imgui.FIRST_USE_EVER)
        expanded, _ = imgui.begin("Inter-X Text Descriptions", None)

        if expanded:
            npy_folder = self.label_npy_list[self.label_pid].split('/')[-1]
            imgui.text(str(npy_folder))
            bef_button = imgui.button('<<Before')
            if bef_button:
                self.set_prev_record()
            imgui.same_line()
            next_button = imgui.button('Next>>')
            if next_button:
                self.set_next_record()
            imgui.same_line()
            tmp_idx = ''
            imgui.set_next_item_width(imgui.get_window_width() * 0.1)
            is_go_to, tmp_idx = imgui.input_text('', tmp_idx); imgui.same_line()
            if is_go_to:
                try:
                    self.go_to_idx = int(tmp_idx) - 1
                except:
                    pass
            go_to_button = imgui.button('>>Go<<'); imgui.same_line()
            if go_to_button:
                self.set_goto_record(self.go_to_idx)
            imgui.text(str(self.label_pid+1) + '/' + str(self.total_tasks))

            imgui.text_wrapped(self.text_val)
        imgui.end()

    def set_prev_record(self):
        self.label_pid = (self.label_pid - 1) % self.total_tasks
        self.clear_one_sequence()
        self.load_one_sequence()
        self.scene.current_frame_id=0

    def set_next_record(self):
        self.label_pid = (self.label_pid + 1) % self.total_tasks
        self.clear_one_sequence()
        self.load_one_sequence()
        self.scene.current_frame_id=0

    def set_goto_record(self, idx):
        self.label_pid = int(idx) % self.total_tasks
        self.clear_one_sequence()
        self.load_one_sequence()
        self.scene.current_frame_id=0

    def get_label_file_list(self):
        for clip in sorted(os.listdir(self.clip_folder)):
            if not clip.startswith('.'):
                self.label_npy_list.append(os.path.join(self.clip_folder, clip))
    
    def load_text_from_file(self):
        self.text_val = ''
        clip_name = self.label_npy_list[self.label_pid].split('/')[-1]
        if os.path.exists(os.path.join(self.text_folder, clip_name+'.txt')):
            with open(os.path.join(self.text_folder, clip_name+'.txt'), 'r') as f:
                for line in f.readlines():
                    self.text_val += line
                    self.text_val += '\n'


    def load_one_sequence(self):
        npy_folder = self.label_npy_list[self.label_pid]

        # load smplx
        smplx_path_p1 = os.path.join(npy_folder, 'P1.npz')
        smplx_path_p2 = os.path.join(npy_folder, 'P2.npz')
        params_p1 = np.load(smplx_path_p1, allow_pickle=True)
        params_p2 = np.load(smplx_path_p2, allow_pickle=True)
        nf_p1 = params_p1['pose_body'].shape[0]
        nf_p2 = params_p2['pose_body'].shape[0]

        betas_p1 = params_p1['betas']
        poses_root_p1 = params_p1['root_orient']
        poses_body_p1 = params_p1['pose_body'].reshape(nf_p1,-1)
        poses_lhand_p1 = params_p1['pose_lhand'].reshape(nf_p1,-1)
        poses_rhand_p1 = params_p1['pose_rhand'].reshape(nf_p1,-1)
        transl_p1 = params_p1['trans']
        gender_p1 = str(params_p1['gender'])

        betas_p2 = params_p2['betas']
        poses_root_p2 = params_p2['root_orient']
        poses_body_p2 = params_p2['pose_body'].reshape(nf_p2,-1)
        poses_lhand_p2 = params_p2['pose_lhand'].reshape(nf_p2,-1)
        poses_rhand_p2 = params_p2['pose_rhand'].reshape(nf_p2,-1)
        transl_p2 = params_p2['trans']
        gender_p2 = str(params_p2['gender'])

        # create body models
        smplx_layer_p1 = SMPLLayer(model_type='smplx',gender=gender_p1,num_betas=10,device=C.device)
        smplx_layer_p2 = SMPLLayer(model_type='smplx',gender=gender_p2,num_betas=10,device=C.device)

        # create smplx sequence for two persons
        smplx_seq_p1 = SMPLSequence(poses_body=poses_body_p1,
                            smpl_layer=smplx_layer_p1,
                            poses_root=poses_root_p1,
                            betas=betas_p1,
                            trans=transl_p1,
                            poses_left_hand=poses_lhand_p1,
                            poses_right_hand=poses_rhand_p1,
                            device=C.device,
                            color=(0.11, 0.53, 0.8, 1.0)
                            )
        smplx_seq_p2 = SMPLSequence(poses_body=poses_body_p2,
                            smpl_layer=smplx_layer_p2,
                            poses_root=poses_root_p2,
                            betas=betas_p2,
                            trans=transl_p2,
                            poses_left_hand=poses_lhand_p2,
                            poses_right_hand=poses_rhand_p2,
                            device=C.device,
                            color=(1.0, 0.27, 0, 1.0)
                            )
        self.scene.add(smplx_seq_p1)
        self.scene.add(smplx_seq_p2)
        self.load_text_from_file()


    def clear_one_sequence(self):
        for x in self.scene.nodes.copy():
            if type(x) is SMPLSequence or type(x) is SMPLLayer:
                self.scene.remove(x)


if __name__=='__main__':

    viewer=SMPLX_Viewer()
    viewer.scene.fps=120
    viewer.playback_fps=120
    viewer.run()