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
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import sys
from scipy.spatial.transform import Rotation
import pyquaternion as pyquat
from egomocap.utils.wis3d_utils import make_vis3d, vis3d_add_coords, vis3d_add_skeleton
from egomogen.scripts.boxing_render import adjust_scene2ground
from egomogen.utils.net_utils import load_other_network
from egomogen.utils.lowlevel_repr_utils import *
from egomogen.utils.geo_transform import apply_T_on_points
from pytorch3d.transforms import axis_angle_to_matrix
from egomogen.utils.smplx_utils import make_smplx
from egomogen.utils.data_utils import *
from egomogen.utils.motion_repr_transform import *
from egomogen.scripts.visualize_dataset import visualize

# T_z2y: zup to yup
T_z2y = torch.FloatTensor([[
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1]]]) # (1, 4, 4)

class BoxingDataset(Dataset):
    def __init__(
        self,
        folder,
        window=120,
        split_size=2, 
        for_eval=False,
        run_demo=False, 
        device='cuda',
        split='train',
        block_size=48
    ):


        self.device = device
        self.folder = folder
        self.split = split
        self.block_size = block_size

        self.smplx_body_params = []
        self.camera_params = []
        self.file_num = 0
        self.pose_pos = []
        self.pose_rot = []
        self.T_z2y = T_z2y.double().to(self.device)
        self.image = []
        self.split_params = {}
        self.file_path = {}
        self.train_motions = []
        self.test_motions = []

        self.smplx_model = make_smplx(type='boxing').double().to(self.device)
        
        file_folder = os.path.join(self.folder,'ego_render')
        print("start to load image")
        self.load_image(file_folder)
        
        if not os.path.exists(os.path.join(folder,'preprocessed_data/preprocessed_data.pkl')): # preprocess data and save
            self.load_motion(file_folder)
            self.preprocess_data()  

        if not os.path.exists(os.path.join(folder,'preprocessed_data/train_data.pkl')):
            with open(os.path.join(folder,'preprocessed_data/preprocessed_data.pkl'),'rb') as pickle_file:
                params = pickle.load(pickle_file)
            pose_pos = params['pose_pos']
            pose_rot = params['pose_rot']
            camera_params = params['camera_params']
            camera_params = [camera_params[i][downsample:] for i in range(len(camera_params))]
            self.file_path = params['origin_file_path']
            
            for file_idx in range(len(pose_pos)):
                prev_params = self.get_motion_rel2prev(pose_pos[file_idx].reshape(1,-1,22,3),pose_rot[file_idx].reshape(1,-1,22,3,3))
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
            print('start to concat and normalize train data')
            with open(os.path.join(folder,'preprocessed_data/train_data.pkl'),'rb') as file:
                train_data = pickle.load(file)
            data_len = len(train_data)
            pose_series = []
            camera_trans = []
            final_params = []
            for idx in tqdm.tqdm(range(data_len)):
                meta = train_data[idx].meta
                motion = train_data[idx].motion
                camera_params = train_data[idx].camera_params
                image = self.image[meta.file_idx][motion['frames']]/255

                concat_data = self.concat_params(meta, motion, camera_params, image, squeeze)
                '''
                pose_series: unnormalized, frames*256
                image: normalized, frames*224*224*3
                camera_params: unnormalized, frames*9
                '''
                pose_series.extend(concat_data.pose_series)
                camera_trans.extend(concat_data.camera_params[:, 6:])
                final_params.append(concat_data)
            norm_pose = self.normalize(pose_series, 'EgoMoGen_data/normalize/pose_series_norm.npy')
            norm_camera_trans = self.normalize(camera_trans, 'EgoMoGen_data/normalize/camera_trans_norm.npy')
            
            for idx in range(len(final_params)):
                final_params[idx].pose_series = Normalize(final_params[idx].pose_series,norm_pose)
                final_params[idx].camera_params[:, 6:] = Normalize(final_params[idx].camera_params[:, 6:],norm_camera_trans)
            with open(os.path.join(folder,'preprocessed_data/final_train_data.pkl'),'wb') as pickle_file:
                pickle.dump(final_params,pickle_file)
        
        if self.split == 'train':
            with open(os.path.join(folder,'preprocessed_data/final_train_data.pkl'),'rb') as pickle_file:
                self.data = pickle.load(pickle_file)
            print("Successfully load the training data!")
        else:
            with open(os.path.join(folder,'preprocessed_data/test_data.pkl'),'rb') as pickle_file:
                self.data = pickle.load(pickle_file)
            print("Successfully load the test data!")

    def load_motion(self,folder):
        # for file_name in os.listdir(folder):
        print('start to load motion')
        for file_name in ['027_ma_comb1_liu_comb2_joints_smpl','028_ma_comb3_liu_comb4_joints_smpl']:
            file_path = os.path.join(folder,file_name,'smplx_params/smplx_params.npy')
            
            # params = np.load(file_path,allow_pickle=True)[:100]
            params = np.load(file_path, allow_pickle=True)
            self.file_path[self.file_num] = file_path

            smplx_params = self.load_body_params(params[:,0])
            smplx_output = self.smplx_model(**smplx_params)

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
        
    def load_image(self,folder):
        # for file_name in os.listdir(folder):
        for file_name in ['027_ma_comb1_liu_comb2_joints_smpl','028_ma_comb3_liu_comb4_joints_smpl']:  
            if file_name == 'preprocessed_data':
                continue
            # image_folder = os.path.join(self.folder,file_name,'rgb')
            image_path = os.path.join(folder,file_name,'rgb/rgb.npy')

            # for image_name in os.listdir(image_folder):
            #     file_path = os.path.join(image_folder,image_name)
            #     image = cv2.imread(file_path)
            #     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #     self.image.append(image_rgb)

            # image_rgb = np.load(image_path,mmap_mode='r')[:100]
            image_rgb = np.load(image_path, mmap_mode='r')
            self.image.append(image_rgb[downsample:])

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
        return pose_rot.double().to('cuda'),root_pos


    def get_rel_position(self,pose_rot,root_pos):
        root_trans = root_pos[...,:3,3]
        root_rotation = root_pos[...,:3,:3]

        pose_pos = torch.einsum("...ij,...jk->...ik",root_rotation.transpose(-1,-2),pose_rot-root_trans)
        return pose_pos


    def get_motion_rel2prev(self, pose_pos, pose_rot):   
        # pose_pos:(1,frames,joints,3)——每个joint在全局坐标系的位置
        # pose_rot:每个joint相对父节点的rotation
        
        # transform from zup to yup ——z轴向上转为y轴向上
        pose_pos = apply_T_on_points(pose_pos, self.T_z2y)   # 坐标系间转换，T为[R t]
        pose_rot[:, :, 0] = self.T_z2y[..., :3, :3] @ pose_rot[:, :, 0]
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
        )

    def preprocess_data(self):  # shift smplx to relative and save in dict
        print('start to preprocess data')
        for file_idx in range(self.file_num):
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
            pickle.dump({'pose_pos':self.pose_pos, 'pose_rot':self.pose_rot, 'camera_params':self.camera_params, 'origin_file_path':self.file_path},pickle_file)
    
    def split_motion(self, motion_data, camera_params, file_idx):
        print('start to split data')
        train_data, test_data = [], []

        data_length = motion_data.pose_vel.size(1)

        # downsample 隔downsample取一帧，把这样作为相邻的两帧，降帧
        for d in range(downsample):
            frames, FN, TRAIN_FN = self.get_train_test_frames(data_length, d)
            train_data.append(dotdict(
                meta = self.get_meta(self.file_path[file_idx], d, 'train', file_idx, motion_data['root_mat'][0][frames[:TRAIN_FN]]),
                motion = self.get_actor_motion(motion_data, frames[:TRAIN_FN], squeeze=False),
                camera_params = camera_params[file_idx][frames[:TRAIN_FN]],
                # root_mat = motion_data['root_mat'][frames[:TRAIN_FN]]
            ))
            test_data.append(dotdict(
                meta = self.get_meta(self.file_path[file_idx], d, 'test', file_idx, motion_data['root_mat'][0][frames[TRAIN_FN:]]),
                motion = self.get_actor_motion(motion_data, frames[TRAIN_FN:], squeeze=False),
                camera_params = camera_params[file_idx][frames[TRAIN_FN:]],
                # root_mat = motion_data['root_mat'][frames[:TRAIN_FN]]
            ))
        return train_data, test_data
    
    def concat_params(self, meta, params, camera_params, image, squeeze):  
        # calculate local coodinate and concat params
        frames = list(range(params.pose_vel.size(1)))

        if squeeze:
            pose_series = LowlevelMoRepr.cal_pose_series(params, frames)[0]
        else:
            pose_series = LowlevelMoRepr.cal_pose_series(params, frames)

        return dotdict(
            meta = meta,
            pose_series = pose_series,
            camera_params = camera_params,
            image = image,
            # root_info = root_info,
        )

    def get_meta(self, origin_file_path, downsample, split, file_idx, root_mat):
        return dotdict(
            origin_file_path = origin_file_path,
            downsample = downsample,
            split = split,
            file_idx = file_idx,
            # origin_frames = frames,
            skeleton = skeleton,
            root_mat = root_mat,
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


    def pad_motion(self, pose_series, pose_sid, data_len):
        padding_motion = []
        # padding_motion[:data_len-pose_sid] = pose_series[pose_sid:data_len]
        # padding_motion[data_len-pose_sid:] = pose_series[data_len].repeat(self.block_size-data_len+pose_sid)
        
        padding_motion.extend(pose_series[pose_sid:data_len])
        padding_motion.extend(pose_series[data_len-1].reshape(1,-1).repeat(self.block_size-data_len+pose_sid,1))
        return padding_motion


    def pad_image(self, image, image_sid, data_len):
        padding_image = []
 
        padding_image.extend(image[image_sid:data_len])
        padding_image.extend(image[data_len-1].reshape(1,224,224,3).tile(self.block_size-data_len+image_sid,1,1,1))
        return padding_image


    def __len__(self):
        if self.split == 'train':
            return len(self.data)*100
        else:
            return len(self.data)

    def __getitem__(self, index):
        index = index % len(self.data)
        wholeseq = self.data[index]
        meta = wholeseq.meta
        pose_series = wholeseq.pose_series
        camera_params = wholeseq.camera_params
        image = torch.from_numpy(wholeseq.image)
        
        pose_sid = random.randint(0,len(wholeseq.pose_series)-1)   # start frame
        pose_eid = pose_sid + self.block_size   # end frame
       
        seq_meta = dotdict(
            original_file_path = meta.origin_file_path,
            downsample = meta.downsample,
            split = meta.split,
            skeleton = meta.skeleton,
            init_root_mat = meta.root_mat[pose_sid],
        )
        if pose_eid > len(wholeseq.pose_series):
            pose_series = self.pad_motion(pose_series, pose_sid, len(wholeseq.pose_series))
            image = self.pad_image(image, pose_sid, len(wholeseq.pose_series))
            camera_params = self.pad_motion(camera_params, pose_sid, len(wholeseq.pose_series))
            pose_eid = len(wholeseq.pose_series)

            pose_series = torch.stack(pose_series)
            camera_params = torch.stack(camera_params)
            image = torch.stack(image)

        else:
            pose_series = pose_series[pose_sid:pose_eid]
            image = image[pose_sid:pose_eid]
            camera_params = camera_params[pose_sid:pose_eid]

        mask = torch.ones(self.block_size)
        mask[pose_eid - pose_sid:] = 0
        
        return {
            'meta' : seq_meta,
            'mask' : mask,
            'pose_series' : pose_series,
            'camera_params' : camera_params,
            'image' : image,
        }

if __name__ == "__main__":
    #dataset=BoxingDataset(folder='/nas/share/ego_data/boxing/ego_render/')
    dataset = BoxingDataset(folder='EgoMoGen_data')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for idx,data in enumerate(dataloader):
        data = dotdict(
            meta = data['meta'],
            mask = data['mask'],
            pose_series = data['pose_series'],
            camera_params = data['camera_params'],
            image = data['image']
        )
        image = data['image']
        camera_params = data['camera_params']
        pose_series = data['pose_series']
        visualize(data)
