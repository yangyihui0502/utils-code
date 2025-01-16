import os
import numpy as np
import joblib 

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch3d.transforms as transforms 
import smplx
import torch
import random
import pickle
import trimesh
import tqdm
import numpy as np
import glob
import subprocess
import cv2
import copy
from PIL import Image
import copy
import pdb
import gc
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import sys
from scipy.spatial.transform import Rotation
import pyquaternion as pyquat
from wis3d import Wis3D
from egomocap.utils.wis3d_utils import make_vis3d, vis3d_add_coords, vis3d_add_skeleton
from egomogen.scripts.boxing_render import adjust_scene2ground
from egomogen.utils.net_utils import load_other_network
from egomogen.utils.lowlevel_repr_utils import *
from egomogen.utils.geo_transform import apply_T_on_points
from pytorch3d.transforms import axis_angle_to_matrix
from egomogen.utils.smplx_utils import make_smplx
from egomogen.utils.data_utils import *
from egomogen.utils.motion_repr_transform import *
from egomogen.scripts.visualize_dataset import visualize_dataset
from egomogen.utils.matrix import *

# T_z2y: zup to yup
T_z2y = torch.FloatTensor([[
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1]]]) # (1, 4, 4)

class BoxingDataset(Dataset):
    def __init__(
        self,
        opt,
        folder,
        window=120,
        split_size=2, 
        for_eval=False,
        run_demo=False, 
        device='cpu',
        split='train',
        block_size=48,
        test_frame_size=30,
        image_type = 'optical flow'
    ):

        if opt != None:
            self.device = opt.device
            self.folder = opt.data_folder
            self.input_frames = opt.input_frames
        else:
            self.device = device
            self.folder = folder
            self.input_frames = 1
        self.split = split
        self.block_size = block_size
        self.test_frame_size = test_frame_size
        self.image_type = image_type

        self.smplx_body_params = []
        self.camera_params = []
        self.file_num = 0
        self.pose_pos = []
        self.oppo_params = []
        self.pose_rot = []
        self.T_z2y = T_z2y.double().to(self.device)
        self.image = []
        self.split_params = {}
        self.file_path = {}
        self.train_motions = []
        self.test_motions = []

        self.smplx_model = make_smplx(type='boxing').double().to(self.device)
        
        self.file_folder = os.path.join(self.folder,'ego_render')
        print("start to load the image path")
        self.load_image(self.file_folder)

        if not os.path.exists(os.path.join(folder,'preprocessed_data/preprocessed_data.pkl')): # preprocess data and save
            self.load_motion()
            self.preprocess_data()  

        if not os.path.exists(os.path.join(folder,'preprocessed_data/train_data.pkl')):
        # if True:
            with open(os.path.join(folder,'preprocessed_data/preprocessed_data.pkl'),'rb') as pickle_file:
                params = pickle.load(pickle_file)
            pose_pos = params['pose_pos']
            pose_rot = params['pose_rot']
            oppo_params = params['oppo_params']
            camera_params = params['camera_params']
            camera_params = [camera_params[i][downsample:] for i in range(len(camera_params))]
            self.file_path = params['origin_file_path']
            
            print('start to split data')
            for file_idx in tqdm.tqdm(range(len(pose_pos))):
                prev_params = self.get_motion_rel2prev(pose_pos[file_idx].reshape(1,-1,22,3),pose_rot[file_idx].reshape(1,-1,22,3,3), oppo_params[file_idx])
                train_motion, test_motion = self.split_motion(prev_params, camera_params, file_idx)

                self.train_motions.extend(train_motion)
                self.test_motions.extend(test_motion)
            save_train_path = os.path.join(folder,'preprocessed_data/train_data.pkl')
            save_test_path = os.path.join(folder,'preprocessed_data/test_data.pkl')
            with open(save_train_path,'wb') as pickle_file:
                pickle.dump(self.train_motions,pickle_file)
            with open(save_test_path,'wb') as pickle_file:
                pickle.dump(self.test_motions,pickle_file)

        # concat params and normalize
        squeeze = True
        if not os.path.exists(os.path.join(folder,'preprocessed_data/final_train_data.pkl')):
            # print("start to load image")

            print('start to concat and normalize train data')
            with open(os.path.join(folder,'preprocessed_data/train_data.pkl'),'rb') as file:
                train_data = pickle.load(file)
            data_len = len(train_data)
            pose_series = []
            camera_trans = []
            final_params = []
            all_image = []
            for idx in tqdm.tqdm(range(data_len)):
                meta = train_data[idx].meta
                motion = train_data[idx].motion
                camera_params = train_data[idx].camera_params

                image_path = self.image[meta.file_idx]

                # if self.image_type == 'rgb':
                #     image = self.image[meta.file_idx][motion['frames']]/255
                # else:
                #     image = self.image[meta.file_idx][motion['frames']]
                
                # image = [torch.tensor(item) for item in image]
                concat_data = self.concat_params(meta, motion, camera_params, image_path)
           
                pose_series.extend(concat_data.pose_series)
                camera_trans.extend(concat_data.camera_params[:, 6:])
                # all_image.extend(image.reshape(image.shape[0], -1))
                final_params.append(concat_data)
            
            # all_image = [torch.tensor(item) for item in all_image]
            norm_image = self.normalize_flow('EgoMoGen_data/normalize/image_flow_norm.npy')
            norm_pose = self.normalize(pose_series, 'EgoMoGen_data/normalize/pose_series_norm.npy')
            norm_camera_trans = self.normalize(camera_trans, 'EgoMoGen_data/normalize/camera_trans_norm.npy')
            
            for idx in range(len(final_params)):
                final_params[idx].pose_series = Normalize(final_params[idx].pose_series, norm_pose)
                final_params[idx].camera_params[:, 6:] = Normalize(final_params[idx].camera_params[:, 6:], norm_camera_trans)
                # if self.image_type != 'rgb':
                    # final_params[idx].image = Normalize(final_params[idx].image.reshape(-1, 224*224*2), np.array(norm_image)).reshape(-1, 224, 224, 2)
                
            with open(os.path.join(folder,'preprocessed_data/final_train_data.pkl'),'wb') as pickle_file:
                pickle.dump(final_params,pickle_file)
        else:
            print('Train data exists,skip loading and normalizing')
        
        if self.split == 'train':
            with open(os.path.join(folder,'preprocessed_data/final_train_data.pkl'),'rb') as pickle_file:
                self.data = pickle.load(pickle_file)
            print("Successfully load the training data!")
        else:
            with open(os.path.join(folder,'preprocessed_data/test_data.pkl'),'rb') as pickle_file:
                self.data = pickle.load(pickle_file)
            print("Successfully load the test data!")

    def load_motion(self):
        folder = self.file_folder
        print('start to load motion')
        for file_name in tqdm.tqdm(os.listdir(folder)):
        # for file_name in ['027_ma_comb1_liu_comb2_joints_smpl','028_ma_comb3_liu_comb4_joints_smpl']:
            file_path = os.path.join(folder,file_name,'smplx_params/smplx_params.npy')
            
            # params = np.load(file_path,allow_pickle=True)[:100]
            params = np.load(file_path, allow_pickle=True)
            self.file_path[self.file_num] = file_path

            smplx_params = self.load_body_params(params[:, 0])
            oppo_params = self.load_body_params(params[:, 1])

            # oppo_smplx_output = self.smplx_model(**oppo_params)
            # oppo_pos = oppo_smplx_output.joints[:, :22, :]
            # oppo_params['transl'] -= self.adjust_pos2ground(oppo_pos.unsqueeze(0))[1][0]
            self.oppo_params.append(oppo_params)

            smplx_output = self.smplx_model(**smplx_params)
            # visualize:
            # wis3d = Wis3D('output/visualize_boxing',f'{file_name}_')
            # print(f'Save visualize data to output/visualize_boxing/{file_name}_')
            # scene=trimesh.load('blender/assets/boxring_color_mesh.obj')
            # height=adjust_scene2ground(scene)
            # scene.vertices[:,1]-=height
            # wis3d.add_mesh(scene,name='scene')
            # motion_mesh = trimesh.Trimesh(vertices=smplx_output.vertices[0], faces=self.smplx_model.bm.faces)
            # wis3d.add_mesh(motion_mesh, name='motion')

            camera_params=self.load_camera(params[:,0])
            
            self.pose_pos.append(smplx_output.joints[:, :22, :])

            # scene=trimesh.load('blender/assets/boxring_color_mesh.obj')
            # height=adjust_scene2ground(scene)
            # scene.vertices[:,1]-=height
            # vis3d = make_vis3d(None, f'smplx_{file_name}', 'output/visualize_dataset',)
            # vis3d.add_mesh(scene, name='scene')
            # vis3d_add_skeleton(vis3d, 0, smplx_output.joints[0, :22], parents, f'frame_0')

            self.smplx_body_params.append(smplx_params)    # [file_num,...]
            self.camera_params.append(camera_params)
            self.file_num += 1
        
 
    def load_body_params(self,body_params):
        human_pose = torch.from_numpy(body_params[:,6:69].reshape(-1,21,3)).to(self.device)
        human_orient = torch.from_numpy(body_params[:,3:6].reshape(-1,3)).to(self.device)
        human_transl = torch.from_numpy(body_params[:,0:3].reshape(-1,3)).to(self.device)
        betas = torch.from_numpy(body_params[:,85:95].reshape(-1,10)).to(self.device)
        gender = torch.from_numpy(body_params[:,95].reshape(-1,1)).to(self.device)

        return {'body_pose':human_pose,'global_orient':human_orient,'transl':human_transl,'betas':betas,'gender':gender}

    def load_camera(self,params):
        full_trans_matrix = []
        for params_idx in range(len(params)):
            origin_matrix=params[params_idx,69:85].reshape(4,4)
            camera_params=torch.from_numpy(np.linalg.inv(origin_matrix))
            # camera_params = torch.from_numpy(origin_matrix)

            rot_6d = transforms.matrix_to_rotation_6d(camera_params[:3,:3])
            camera_params = torch.cat((rot_6d,camera_params[:3,3]))
            # trans_matrix=np.eye(4)
            # trans_matrix[:3,3]=camera_params[:3,3]
            # full_trans_matrix.append(trans_matrix)
            # full_trans_matrix.append(camera_params[:3])
            full_trans_matrix.append(camera_params)

        full_trans_matrix = torch.stack(full_trans_matrix)
        return full_trans_matrix.reshape(len(full_trans_matrix),9)
        
    def load_image(self, folder):
        for file_name in tqdm.tqdm(os.listdir(folder)):
        # for file_name in ['027_ma_comb1_liu_comb2_joints_smpl','028_ma_comb3_liu_comb4_joints_smpl']:  
            if file_name == 'preprocessed_data':
                continue
            # image_folder = os.path.join(self.folder,file_name,'rgb')
            if self.image_type == 'rgb':
                image_path = os.path.join(folder, file_name, 'rgb/rgb.npy')

                # for image_name in os.listdir(image_folder):
                #     file_path = os.path.join(image_folder,image_name)
                #     image = cv2.imread(file_path)
                #     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #     self.image.append(image_rgb)

                # image_rgb = np.load(image_path,mmap_mode='r')[:100]
                # image_rgb = np.load(image_path, mmap_mode='r').astype(np.float32)
                # self.image.append(image_rgb[downsample:])
                self.image.append(image_path)
            else:
                subfolder = os.path.join(folder, file_name, 'optical_flow')
                os.makedirs(subfolder, exist_ok=True)
                if os.path.exists(os.path.join(subfolder, 'optical_flow.npy')):
                    # image_flow = np.load(os.path.join(subfolder, 'optical_flow.npy'), mmap_mode='r').astype(np.float32)
                    # self.image.append(image_flow)
                    image_path = os.path.join(subfolder, 'optical_flow.npy')
                    self.image.append(image_path)
                elif os.path.exists(os.path.join(subfolder, 'normalized_flow.npy')):
                    image_path = os.path.join(subfolder, 'optical_flow.npy')
                    self.image.append(image_path)
                else:
                    image_path = os.path.join(folder, file_name, 'rgb/rgb.npy')
                    image_rgb = np.load(image_path, mmap_mode='r')
                    flow = np.zeros((image_rgb.shape[0]-downsample, image_rgb.shape[1]//2, image_rgb.shape[2]//2, 2))
                    for idx in range(downsample, len(image_rgb), 1):
                        frame1 = cv2.resize(cv2.cvtColor(image_rgb[idx-downsample].astype(np.float32), cv2.COLOR_RGB2GRAY), None, fx=0.5, fy=0.5)
                        frame2 = cv2.resize(cv2.cvtColor(image_rgb[idx].astype(np.float32), cv2.COLOR_RGB2GRAY), None, fx=0.5, fy=0.5)
                        flow[idx-downsample] = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        gc.collect()
                        
                    flow = np.float32(flow)
                    np.save(os.path.join(subfolder, 'optical_flow.npy'), flow)
                    # self.image.append(flow)
                    self.image.append(os.path.join(subfolder, 'optical_flow.npy'))


    def process_test_data(self, test_data):
        pose_series = []
        camera_trans = []
        final_params = []
        all_image = []
        folder = self.file_folder
        meta = test_data.meta
        motion = test_data.motion
        camera_params = test_data.camera_params

        # image_path = self.image[meta.file_idx]
        image_path = os.path.join(meta.origin_file_path[:-30], 'optical_flow/normalized_flow.npy')
        # image = np.load(os.path.join(image_path[:-17], 'normalized_flow.npy'))[downsample:][motion['frames']]
        image = np.load(image_path)[motion['frames']]

        # if self.image_type == 'rgb':
        #     image = self.image[meta.file_idx][motion['frames']]/255
        # else:
        #     image = self.image[meta.file_idx][motion['frames']]
        
        # image = [torch.tensor(item) for item in image]
        concat_data = self.concat_params(meta, motion, camera_params, image_path)
   
        pose_series.extend(concat_data.pose_series)
        camera_trans.extend(concat_data.camera_params[:, 6:])
        # all_image.extend(image.reshape(image.shape[0], -1))
        final_params=concat_data
        
        # all_image = [torch.tensor(item) for item in all_image]
        norm_image = np.load('EgoMoGen_data/normalize/image_flow_norm.npy')
        norm_pose = np.load('EgoMoGen_data/normalize/pose_series_norm.npy')
        norm_camera_trans = np.load('EgoMoGen_data/normalize/camera_trans_norm.npy')
        
        final_params.pose_series = Normalize(final_params.pose_series, norm_pose)
        final_params.camera_params[:, 6:] = Normalize(final_params.camera_params[:, 6:], norm_camera_trans)

        shape = image.shape
        # image = Normalize(image.reshape(-1), norm_image).reshape(shape)
        image = torch.from_numpy(image)
        # if self.image_type != 'rgb':
            # final_params.image = Normalize(final_params.image.reshape(-1, 224*224*2), np.array(norm_image)).reshape(-1, 224, 224, 2)
        return final_params, image
    


    def shift_smplx2matrix(self,smplx_dict):
        root_pos = torch.eye(4)
        pose_rot = torch.zeros((smplx_dict['transl'].shape[0], njoints, 3, 3))
        # root_pos = root_pos.repeat(len(smplx_dict['transl']),1,1)

        body_pose = smplx_dict['body_pose']
        orient = smplx_dict['global_orient']
        pose_rot[:, 1:] = axis_angle_to_matrix(body_pose)
        pose_rot[:, 0] = axis_angle_to_matrix(orient)
        
        # root_trans = smplx_dict['transl']
        # root_rotation = axis_angle_to_matrix(smplx_dict['global_orient'])
        
        # root_pos[:,:3,:3] = root_rotation
        # root_pos[:,:3,3] = root_trans

        #full_root_pos.append(root_pos)

        #full_pose_rot = torch.stack(full_pose_rot)
        #full_root_pos = torch.stack(full_root_pos)
        return pose_rot.double(), root_pos


    def get_rel_position(self,pose_rot,root_pos):
        root_trans = root_pos[...,:3,3]
        root_rotation = root_pos[...,:3,:3]

        pose_pos = torch.einsum("...ij,...jk->...ik",root_rotation.transpose(-1,-2),pose_rot-root_trans)
        return pose_pos


    def adjust_pos2ground(self, pose_pos):
        y = pose_pos[:, :, :, 1]
        y_min, _ = torch.min(y, dim=-1)
        pose_pos[:, :, :, 1] -= y_min.unsqueeze(-1)
        return pose_pos, y_min


    def get_motion_rel2prev(self, pose_pos, pose_rot, oppo_params):   
        # pose_pos:(1,frames,joints,3)——每个joint在全局坐标系的位置
        # pose_rot:每个joint相对父节点的rotation
        
        # transform from zup to yup ——z轴向上转为y轴向上
        # pose_pos = apply_T_on_points(pose_pos, self.T_z2y)   # 坐标系间转换，T为[R t]   
        # 将人体pos转换到地面上
        pose_pos, _ = self.adjust_pos2ground(pose_pos)

        # pose_rot[:, :, 0] = self.T_z2y[..., :3, :3] @ pose_rot[:, :, 0] 
        # transform rel2parents to global rotations 计算每个点的全局rotation matrix
        pose_rot = matrix.forward_kinematics(pose_rot, parents)
        # get root pos
        root_pos = pose_pos[:, :, 0].clone()
        root_pos[..., 1] = 0 # y = 0,把root的投影点作为原点
        # get pose vel
        pose_vel = (pose_pos[:, downsample:] - pose_pos[:, :-downsample]) * train_fps
        # get root mat
        # root_mat:[R t],root_rot:R 局部坐标系：y轴向上，x轴按照joint的左右方向，z轴为叉乘
        root_mat, root_rot = MoReprTrans.get_root_transmat(pose_pos=pose_pos.float(), root_pos=root_pos)
        # get root ctrl
        # 将root_pos转化为downsample帧前的根坐标系下的相对坐标
        root_off = matrix.get_relative_position_to(root_pos[:, downsample:][:, :, None], root_mat[:, :-downsample].double()) # (T, None, 3), (T, 4, 4) -> (T, 1, 3)
        root_dir = matrix.get_mat_BtoA(root_rot[:, :-downsample], root_rot[:, downsample:]) # (T, 3, 3), (T, 3, 3) -> (T, 3, 3)
        root_off_2d = root_off[..., 0, [0, 2]]
        root_dir_2d = root_dir[..., [0, 2], 2]
        root_ctrl = torch.cat((root_off_2d, root_dir_2d), dim=-1)

        # for i in range(len(oppo_params)):
        for key in oppo_params.keys():
            oppo_params[key] = oppo_params[key][downsample:] 
        
        return dotdict(
            pose_pos = pose_pos[:,downsample:].double(),  # global,在后面concat的时候转换
            pose_rot = pose_rot[:,downsample:].double(),  # global
            pose_vel = pose_vel.double(),                 # global
            root_mat = root_mat[:,downsample:].double(),  # root坐标系的tran matrix
            root_rot = root_rot[:,downsample:].double(),
            root_pos = root_pos[:,downsample:].double(),
            root_off = root_off_2d.double(),
            root_dir = root_dir_2d.double(),
            root_ctrl = root_ctrl.double(),
            oppo_params = oppo_params,
        )

    def preprocess_data(self):  # shift smplx to relative and save in dict
        print('start to preprocess data')
        for file_idx in tqdm.tqdm(range(self.file_num)):
            pose_rot,root_pos = self.shift_smplx2matrix(self.smplx_body_params[file_idx])
            self.pose_rot.append(pose_rot)
            # pose_pos = self.get_rel_position(pose_rot,root_pos)
            # image = self.image_rgb[file_idx]
            # rel_params = self.get_motion_rel2prev(self.pose_pos[file_idx].reshape(1,-1,21,3),pose_rot.reshape(1,-1,21,3,3))
            # if file_idx == 0:
        #         for name in rel_params.keys():
        #             self.split_params[name] = []
            #self.split_motion(dict(rel_params))

        # for name in self.split_params.keys():
        #     self.split_params[name]=torch.stack(self.split_params[name]).flatten(0,1)
        
        # pose_rot,root_pos = self.shift_smplx2matrix(self.smplx_body_params)
        # rel_params = self.get_motion_rel2prev(self.pose_pos,pose_rot)
            # meta = self.get_meta(self.file_path[file_idx],downsample,split,file_idx)

            # concat_params = self.concat_params(meta,rel_params,image)
            # all_params = all_params.append(concat_params)
        # self.normalize(all_params,'pose_series')
        # self.normalize(all_params,'image')

        os.makedirs(os.path.join(self.folder,'preprocessed_data'), exist_ok=True)
    
        with open(os.path.join(self.folder,'preprocessed_data','preprocessed_data.pkl'),'wb') as pickle_file:
            pickle.dump({'pose_pos':self.pose_pos, 'pose_rot':self.pose_rot, 'camera_params':self.camera_params, 'origin_file_path':self.file_path, "oppo_params":self.oppo_params},pickle_file)
    
    def split_motion(self, motion_data, camera_params, file_idx):
        train_data, test_data = [], []
        train_oppo_params, test_oppo_params = {}, {}
        oppo_params = motion_data['oppo_params']

        data_length = motion_data.pose_vel.size(1)

        # downsample 隔downsample取一帧，把这样作为相邻的两帧，降帧
        for d in range(downsample):
            frames, FN, TRAIN_FN = self.get_train_test_frames(data_length, d)
            for key in oppo_params.keys():
                train_oppo_params[key] = oppo_params[key][frames[:TRAIN_FN]]
                test_oppo_params[key] = oppo_params[key][frames[TRAIN_FN:]]

            train_data.append(dotdict(
                meta = self.get_meta(self.file_path[file_idx], d, 'train', file_idx, motion_data['root_mat'][0][frames[:TRAIN_FN]], train_oppo_params),
                motion = self.get_actor_motion(motion_data, frames[:TRAIN_FN], squeeze=False),
                camera_params = camera_params[file_idx][frames[:TRAIN_FN]],
                # root_mat = motion_data['root_mat'][frames[:TRAIN_FN]]
            ))
            test_data.append(dotdict(
                meta = self.get_meta(self.file_path[file_idx], d, 'test', file_idx, motion_data['root_mat'][0][frames[TRAIN_FN:]], test_oppo_params),
                motion = self.get_actor_motion(motion_data, frames[TRAIN_FN:], squeeze=False),
                camera_params = camera_params[file_idx][frames[TRAIN_FN:]],
                # root_mat = motion_data['root_mat'][frames[:TRAIN_FN]]
            ))
        return train_data, test_data
    
    def concat_params(self, meta, params, camera_params, image_path, visualize_pose=False, squeeze=True):  
        # calculate local coodinate and concat params
        frames = list(range(params.pose_vel.size(1)))

        # if squeeze:
        #     pose_series = LowlevelMoRepr.cal_pose_series(params, frames)[0]
        # else:
        #     pose_series = LowlevelMoRepr.cal_pose_series(params, frames)
        pose_series, pose = LowlevelMoRepr.cal_pose_series(params, frames)
        if visualize_pose:
            wis3d = wis3d = make_wis3d(f'dataset_loc_pos', 'output/visualize_dataset')
            visualize_pos(wis3d, pose)

        return dotdict(
            meta = meta,
            pose_series = pose_series,
            camera_params = camera_params,
            image_path = image_path,
            frames = params['frames']
            # root_info = root_info,
        )

    def get_meta(self, origin_file_path, downsample, split, file_idx, root_mat, oppo_params):
        return dotdict(
            origin_file_path = origin_file_path,
            downsample = downsample,
            split = split,
            file_idx = file_idx,
            # origin_frames = frames,
            skeleton = skeleton,
            root_mat = root_mat,
            oppo_params = oppo_params
        )
    

    def cal_root_info(motion, oppo, frames, f0):
        root2oppo_ctrl = motion.root2oppo_ctrl[:, frames]
        # cal root_pos to the center of the first frame
        center = (motion.root_pos[:, [f0]] + oppo.root_pos[:, [f0]]) / 2
        root2center = motion.root_pos[:, frames] - center
        # 计算欧几里得距离
        root2center = (root2center[..., [0, 2]] ** 2).sum(-1, keepdim=True) ** 0.5
        root_info = torch.cat([root2center, root2oppo_ctrl], dim=-1)
        return root_info
    

    def get_train_test_frames(self, data_length, d):
        '''
            data_length: total frames
            d: downsample
        '''
        frames = list(range(d, data_length, downsample))    # 间隔取样
        FN = len(frames) // U * U
        frames = frames[:FN]    # 选取帧的列表
        TRAIN_FN = int(FN * 0.8) // U * U   # 训练帧数
        return frames, FN, TRAIN_FN
    

    def get_actor_motion(self,motion, frames, squeeze):
        actor_motion = dotdict()
        if squeeze:
            for key, item in motion.items():
                if isinstance(item, torch.Tensor):
                    actor_motion[key] = item[0, frames]
            return actor_motion
        
        for key, item in motion.items():
            if isinstance(item, torch.Tensor):
                actor_motion[key] = item[:, frames]
        actor_motion['frames'] = frames
        return actor_motion
    

    def normalize(self, data, norm_file):
        shape = data[0].shape
        all_data = [data[i].view(-1, shape[-1]) for i in range(len(data))]
        all_data = torch.cat(all_data, dim=0)
        norm_data = all_data.mean(dim=0), all_data.std(dim=0)
        norm_data = torch.stack(norm_data, dim=0)
        save_norm_data(norm_file, norm_data.detach().cpu().numpy())
        
        return norm_data


    def normalize_flow(self, norm_file):
        if not os.path.exists(norm_file):
            image_path = self.image
            n = 0
            mean = 0
            M2 = 0
            for path in tqdm.tqdm(image_path):
                image = np.load(path).astype(np.float32)
                n += len(image)
                batch_size = len(image)
                image = image.reshape(len(image), -1)

                # 计算当前批次的均值和方差
                batch_mean = np.mean(image, axis=0)

                # 增量式更新均值
                delta = batch_mean - mean
                mean += delta * batch_size / n

                # 增量式更新M2
                M2 += np.sum((image - batch_mean) * (image - mean), axis=0)
            
            var = (M2 / (n-1)).astype(np.float32)
            mean = mean.astype(np.float32)
            save_norm_data(norm_file, (mean, var))
        else:
            mean, var = np.load(norm_file).astype(np.float32)

        for path in tqdm.tqdm(self.image):
            image = np.load(path).astype(np.float32)
            shape = image.shape
            image = Normalize(image.reshape(len(image), -1), (mean,var)).reshape(shape)

            np.save(os.path.join(path[:-17], 'normalized_flow.npy'), image)


    def pad_motion(self, block_size, pose_series, pose_sid, data_len):
        padding_motion = []
        # padding_motion[:data_len-pose_sid] = pose_series[pose_sid:data_len]
        # padding_motion[data_len-pose_sid:] = pose_series[data_len].repeat(self.block_size-data_len+pose_sid)
        
        padding_motion.extend(pose_series[pose_sid:data_len])
        padding_motion.extend(pose_series[data_len-1].reshape(1,-1).repeat(block_size-data_len+pose_sid,1))
        return padding_motion

    def pad_oppo_motion(self, block_size, oppo_params, pose_sid, data_len):
        padding_motion = {}
    
        for key in oppo_params.keys():
            padding_motion[key] = []
            padding_motion[key].extend(oppo_params[key][pose_sid: data_len])
            if key != 'body_pose':
                padding_motion[key].extend(oppo_params[key][data_len-1].reshape(1,-1).repeat(block_size-data_len+pose_sid,1))
            else:
                padding_motion[key].extend(oppo_params[key][data_len-1].reshape(1,21,3).repeat(block_size-data_len+pose_sid,1,1))

        return padding_motion
    
    def pad_image(self, block_size, image, image_sid, data_len):
        padding_image = []
 
        shape = image.shape
        padding_image.extend(image[image_sid:data_len])
        padding_image.extend(image[data_len-1].reshape(1,shape[1],shape[2],-1).tile(block_size-data_len+image_sid,1,1,1))
        return padding_image


    def __len__(self):
        if self.split == 'train':
            return len(self.data)*100
        else:
            return len(self.data)

    def __getitem__(self, index):
        index = index % len(self.data)
        wholeseq = self.data[index]
        
        # frames = wholeseq.motion['frames']
        
        if self.split == 'test':
            wholeseq, image = self.process_test_data(self.data[index])
            pose_sid = random.randint(1, max(len(wholeseq.pose_series) - self.test_frame_size,2))   # start frame
            pose_eid = pose_sid + self.test_frame_size   # end frame

        else:
            # image = torch.from_numpy(np.load(os.path.join(wholeseq.image_path[:-17], 'normalized_flow.npy'))[frames])
            image = torch.from_numpy(np.load(os.path.join(wholeseq.meta.origin_file_path[:-30], 'optical_flow/normalized_flow.npy')))

            pose_sid = random.randint(1,len(wholeseq.pose_series)-self.input_frames)   # start frame
            pose_eid = pose_sid + self.block_size   # end frame
        
        # print('idx=', index, f',pose_sid={pose_sid}, pose_eid={pose_eid}')

        block_size = pose_eid - pose_sid
        meta = wholeseq.meta
        pose_series = wholeseq.pose_series
        camera_params = wholeseq.camera_params
        mask = torch.ones(block_size)
        # image = torch.from_numpy(wholeseq.image)

        if pose_eid > len(wholeseq.pose_series):
            pose_series = self.pad_motion(block_size, pose_series, pose_sid, len(wholeseq.pose_series))
            image = self.pad_image(block_size, image, pose_sid, len(wholeseq.pose_series))
            camera_params = self.pad_motion(block_size, camera_params, pose_sid, len(wholeseq.pose_series))
            oppo_params = self.pad_oppo_motion(block_size, meta.oppo_params, pose_sid, len(wholeseq.pose_series))
            pose_eid = len(wholeseq.pose_series)

            pose_series = torch.stack(pose_series)
            camera_params = torch.stack(camera_params)
            image = torch.stack(image)
            for key in oppo_params.keys():
                oppo_params[key] = torch.stack(oppo_params[key])
        else:
            pose_series = pose_series[pose_sid: pose_eid]
            image = image[pose_sid: pose_eid]
            camera_params = camera_params[pose_sid: pose_eid]

            oppo_params = {}
            for key in meta.oppo_params.keys():
                oppo_params[key] = meta.oppo_params[key][pose_sid: pose_eid]

        seq_meta = dotdict(
                original_file_path = meta.origin_file_path,
                downsample = meta.downsample,
                split = meta.split,
                skeleton = meta.skeleton,
                init_root_mat = meta.root_mat[pose_sid+self.input_frames-2],
                oppo_params = oppo_params,
            )
        
        mask[pose_eid - pose_sid:] = 0

        return {
            'meta' : seq_meta,
            'mask' : mask.to('cuda'),
            'pose_series' : pose_series.to('cuda'),
            'view' : camera_params.to('cuda'),
            'image' : image.to(dtype=torch.float32, device='cuda'),
        }

if __name__ == "__main__":
    #dataset=BoxingDataset(folder='/nas/share/ego_data/boxing/ego_render/')
    image_type = 'optical_flow'
    dataset = BoxingDataset(split='test', opt=None, folder='EgoMoGen_data', image_type=image_type, block_size=50)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    for idx,data in enumerate(dataloader):
        data = dotdict(
            meta = data['meta'],
            mask = data['mask'].to('cpu'),
            pose_series = data['pose_series'].to('cpu'),
            view = data['view'].to('cpu'),
            image = data['image']
        )

        visualize_dataset(data, image_type, vis_mesh=True)
