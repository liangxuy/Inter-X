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

OPTITRACK_LIMBS=[
[0,1],[1,2],[2,3],[3,4],
[0,5],[5,6],[6,7],[7,8],
[0,9],[9,10],
[10,11],[11,12],[12,13],[13,14],
    [14,15],[15,16],[16,17],[17,18],
    [14,19],[19,20],[20,21],[21,22],
    [14,23],[23,24],[24,25],[25,26],
    [14,27],[27,28],[28,29],[29,30],
    [14,31],[31,32],[32,33],[33,34],
[10,35],[35,36],[36,37],[37,38],
    [38,39],[39,40],[40,41],[41,42],
    [38,43],[43,44],[44,45],[45,46],
    [38,47],[47,48],[48,49],[49,50],
    [38,51],[51,52],[52,53],[53,54],
    [38,55],[55,56],[56,57],[57,58],
[10,59],[59,60]
]

SELECTED_JOINTS=np.concatenate(
[range(0,5),range(6,10),range(11,63)]
)


class Skeleton_Viewer(Viewer):
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

        # load joint
        skeleton_path_p1 = os.path.join(npy_folder, 'P1.npy')
        skeleton_path_p2 = os.path.join(npy_folder, 'P2.npy')

        joint_positions_p1 = np.load(skeleton_path_p1)
        joint_positions_p2 = np.load(skeleton_path_p2)
        joint_positions_p1 = joint_positions_p1[:, SELECTED_JOINTS]
        joint_positions_p2 = joint_positions_p2[:, SELECTED_JOINTS]

        if joint_positions_p1.shape[0] == 0:
            return
        bvh_skel_p1 = Skeletons(joint_positions=joint_positions_p1,
                                joint_connections=OPTITRACK_LIMBS, color=(0.11, 0.53, 0.8, 1.0))
        bvh_skel_p2 = Skeletons(joint_positions=joint_positions_p2,
                                joint_connections=OPTITRACK_LIMBS, color=(1.0, 0.27, 0, 1.0))
        self.scene.add(bvh_skel_p1)
        self.scene.add(bvh_skel_p2)

        self.load_text_from_file()


    def clear_one_sequence(self):
        for x in self.scene.nodes.copy():
            if type(x) == Skeletons:
                self.scene.nodes.remove(x)


if __name__=='__main__':

    viewer=Skeleton_Viewer()
    viewer.scene.fps=120
    viewer.playback_fps=120
    viewer.run()