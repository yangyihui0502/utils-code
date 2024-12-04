import smplx
import torch
import random
import pickle
import trimesh
import tqdm
import pyrender
import numpy as np
import glob
import subprocess
import cv2
import copy
from PIL import Image
import pdb
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import sys
sys.path.append(".")
from scipy.spatial.transform import Rotation
import pyquaternion as pyquat
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import cm as cmx
from egomogen.utils.smplx_utils import make_smplx
from wis3d import Wis3D

def obj_vt(fname): # read texture coordinates: (u,v)
    res = []
    with open(fname) as f:
        for line in f:
            if line.startswith('vt '):
                tmp = line.split(' ')
                v = [float(i) for i in tmp[1:3]]
                res.append(v)
    return np.array(res, dtype=np.float32)

def obj_fv(fname): # read vertices id in faces: (vv1,vv2,vv3)
    res = []
    with open(fname) as f:
        for line in f:
            if line.startswith('f '):
                tmp = line.split(' ')
                if '/' in tmp[1]:
                    v = [int(i.split('/')[0]) for i in tmp[1:4]]
                else:
                    v = [int(i) for i in tmp[1:4]]
                res.append(v)
    return np.array(res, dtype=np.int32) - 1 # obj index from 1

def obj_ft(fname): # read texture id in faces: (vt1,vt2,vt3)
    res = []
    with open(fname) as f:
        for line in f:
            if line.startswith('f '):
                tmp = line.split(' ')
                if '/' in tmp[1]:
                    v = [int(i.split('/')[1]) for i in tmp[1:4]]
                else:
                    raise(Exception("not a textured obj file"))
                res.append(v)
    return np.array(res, dtype=np.int32) - 1 # obj index from 1

def save_rgb_image(rgb_image,folder):
    base_path = os.path.join(folder,'rgb')
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    np.save(os.path.join(base_path,'rgb.npy'),rgb_image)

def load_scene(scene_path):
    scene_mesh=trimesh.load(scene_path)
    return scene_mesh

def adjust_scene2ground(scene_mesh):
    vertices=scene_mesh.vertices

    x_max=vertices[:,0].max()
    index=np.where(vertices[:,0]==x_max)

    height=vertices[index,1].max()
    return height

def adjust_human2ground(vertices):
    y_dis=vertices[:,:,1]
    y_min=y_dis.min(axis=-1).reshape(-1,1)
    vertices[:,:,1]-=y_min
    return vertices,y_min

class EgoviewGeneratorboxing:
    def __init__(self):
        model_path = os.path.abspath("data/SMPL_Models") 
        self.scene_path='blender/assets/boxring_color_mesh.obj'
        self.data_folder='EgoMoGen_data/boxing/'
        self.device = torch.device('cuda')

        # load smplx
        '''
        self.smplx_model = smplx.create(model_path=model_path,
                          model_type='smplx',
                          gender="male",
                          use_pca=False,
                          batch_size=2,
                          ).to(self.device)
        '''
        self.smplx_model = make_smplx(type='boxing')
        # set light and pyrender
        ambient_intensity = np.random.uniform(0.5, 0.8)
        bg_color = np.random.uniform(0.9, 1.)
        self.scene = pyrender.Scene(ambient_light=[ambient_intensity, ambient_intensity, ambient_intensity], bg_color=[bg_color, bg_color, bg_color, 0.5])  
        self.renderer = pyrender.OffscreenRenderer(viewport_width=224, viewport_height=224)
        
        light = pyrender.DirectionalLight(color=[np.random.uniform(0.9, 1.), np.random.uniform(0.9, 1.), np.random.uniform(0.9, 1.)],\
                                        intensity=np.random.uniform(2., 6.))
        light_node = pyrender.Node(light=light)
        self.scene.add_node(light_node)

        # load scene
        self.scene_mesh=load_scene(self.scene_path)
        height=adjust_scene2ground(self.scene_mesh)
        self.scene_mesh.vertices[:,1]-=height

        self.add_scene(self.scene_mesh)
        
        # init camera
        # cx = np.random.uniform(942.543, 946.108)
        # cy = np.random.uniform(505.898, 510.081)
        # fx = np.random.uniform(1450.93, 1480.28)
        self.camera = pyrender.camera.IntrinsicsCamera(fx=140, fy=156, cx=112, cy=111)
        self.camera_node = None

        # load body textures
        self.smplx_vt = obj_vt('EgoMoGen_data/hood_data/bedlam/smplx_uv.obj')
        self.smplx_f = obj_fv('EgoMoGen_data/hood_data/bedlam/smplx_uv.obj')
        self.smplx_ft = obj_ft('EgoMoGen_data/hood_data/bedlam/smplx_uv.obj')
        # self.smplx_vt, self.smplx_f, self.smplx_ft = get_vt_f_ft('EgoMoGen_data//BEDLAM/smplx_uv.obj')

        self.body_texture_path = glob.glob('EgoMoGen_data/BEDLAM/smpl_texture/*.png')
        self.body_texture = {}
        for tex_path in self.body_texture_path:
            # im = Image.alpha_composite(Image.open(tex_path), eye_img)
            # im.save("/mnt/vlg-nfs/genli/datasets/bedlam/" + tex_path.split('/')[-1])
            self.body_texture[tex_path.split('/')[-1]] = Image.open(tex_path)
       
        self.body_node1=None
        self.body_node2=None


    def gen_egoview(self,params1,params2,save_params,smpl_texture_path,filename):
        self.gen_human_mesh(params1,params2)
        if save_params:
            self.smplx_dict = {}
            for key in self.smplx_dict1.keys():
                self.smplx_dict[key] = torch.stack([self.smplx_dict1[key],self.smplx_dict2[key]]).transpose(0,1)
            custom_smplx_params = np.zeros((self.smplx_dict['transl'].shape[0],2,99))
            custom_smplx_params[:,:,:3] = self.smplx_dict['transl'].cpu()
            custom_smplx_params[:,0,1]-=self.y1_min.reshape(-1)
            custom_smplx_params[:,1,1]-=self.y2_min.reshape(-1)
            custom_smplx_params[:,:,3:6] = self.smplx_dict['global_orient'].cpu()
            custom_smplx_params[:,:,6:69] = self.smplx_dict['body_pose'].cpu()
            # custom_smplx_params[:,0,69:85] = self.Rt.reshape(-1)
        '''
         npy文件：
         0:3 transl
         3:6 global_orient
         6:69 body_pose
         69:85 camera pose inverse matrix (used to transform to egocentric camera frame coordinates)
         #相机为[R t]取逆矩阵后展开存入(最后一行为0001)
         85:95 beta
         95 gender (male = 0, female = 1)
         '''
        all_rgb = []
        all_depth = []
        for frame_idx in tqdm.tqdm(range(self.vertices.shape[1])):
        # for frame_idx in tqdm.tqdm(range(2099,2300)):
            self.add_human_mesh(smpl_texture_path,frame_idx)
            self.set_camera(frame_idx)
            if save_params:
                custom_smplx_params[frame_idx,0,69:85] = self.Rt.reshape(-1)

            rgb,depth=self.renderer.render(self.scene)
            all_rgb.append(rgb)
            all_depth.append(depth)
        all_rgb = np.array(all_rgb)
        all_depth = np.array(all_depth)

        if not os.path.exists(os.path.join(save_folder,file_name[:-4],"smplx_params")):
            os.makedirs(os.path.join(save_folder,file_name[:-4],"smplx_params"))
        np.save(os.path.join(save_folder,file_name[:-4],f"smplx_params/smplx_params.npy"),custom_smplx_params)
 
        return all_rgb,all_depth
        
    def load_smplx_params(self,params1,params2):  #params为字典形式,含有某一frame的params
        batch_size = params1['betas'].shape[0]
        '''
        betas = np.stack([params1['betas'],params2['betas']]).reshape(batch_size,2,-1)
        transl = np.stack([params1['transl'],params2['transl']]).reshape(batch_size,2,-1)
        global_orient = np.stack([params1['global_orient'],params2['global_orient']]).reshape(batch_size,2,-1)
        body_pose = np.stack([params1['body_pose'],params2['body_pose']]).reshape(batch_size,2,-1)
        self.smplx_dict = {
            'betas': betas,
            'transl': transl,
            'global_orient': global_orient,
            'body_pose': body_pose
        }
        self.smplx_dict = self.params2torch(self.smplx_dict)
        '''
        self.smplx_dict1 = self.params2torch({
            'betas': params1['betas'].reshape(batch_size,-1),
            'transl': params1['transl'].reshape(batch_size,-1),
            'global_orient': params1['global_orient'].reshape(batch_size,-1),
            'body_pose': params1['body_pose'].reshape(batch_size,-1)
        })
        self.smplx_dict2 = self.params2torch({
            'betas': params2['betas'].reshape(batch_size,-1),
            'transl': params2['transl'].reshape(batch_size,-1),
            'global_orient': params2['global_orient'].reshape(batch_size,-1),
            'body_pose': params2['body_pose'].reshape(batch_size,-1)
        })
        
    
    def gen_human_mesh(self,params1,params2):
        self.load_smplx_params(params1,params2)
        output1 = self.smplx_model(**self.smplx_dict1)
        output2 = self.smplx_model(**self.smplx_dict2)
        output = {}
        for key in output1.keys():
            if output1[key] != None:
               output[key] = torch.stack([output1[key],output2[key]])

        self.vertices = output['vertices'].detach().cpu().numpy()
        self.vertices[0],self.y1_min=adjust_human2ground(self.vertices[0])
        self.vertices[1],self.y2_min=adjust_human2ground(self.vertices[1])
        
        self.joints = output['joints'].detach().cpu().numpy()
        self.joints[0,:,:,1]-=self.y1_min
        self.joints[1,:,:,1]-=self.y2_min


    def add_human_mesh(self,smpl_texture_path,frame_idx):
        for human_id in range(2):
            m = self.make_new_mesh(self.smplx_vt, self.smplx_f, self.smplx_ft, self.vertices[human_id][frame_idx], self.body_texture[smpl_texture_path])
            # m = trimesh.Trimesh(vertices=self.vertices[human_id][frame_idx],faces=self.smplx_model.bm.faces)
            #m = trimesh.Trimesh(vertices=self.vertices[human_id], faces=self.smplx_ft, \
            #                visual=trimesh.visual.TextureVisuals(uv=self.smplx_vt, image=self.body_texture[smpl_texture_path]), process=False)
            # wis3d = Wis3D('output/visualize_boxing',f'debug_{frame_idx}')
            # wis3d.add_mesh(m,name=f'body_{human_id}')
            # wis3d.add_mesh(self.scene_mesh,name='scene')

            body_mesh = pyrender.Mesh.from_trimesh(m, smooth=True)
            if human_id==0: 
                if self.body_node1 is not None:
                    self.scene.remove_node(self.body_node1)
                self.body_node1 = pyrender.Node(mesh=body_mesh, name='body1')
                self.scene.add_node(self.body_node1)
            else:
                if self.body_node2 is not None:
                    self.scene.remove_node(self.body_node2)
                self.body_node2 = pyrender.Node(mesh=body_mesh, name='body2')
                self.scene.add_node(self.body_node2)

    def set_camera(self,frame_idx):
        joint = self.joints[0][frame_idx]
        # joint = self.joints[1][frame_idx]
        # 57: leye 56: reye
        # look_front. approx. may not be vertical to look_right
        # look_at = joint[57] - joint[23] + joint[56] - joint[24]  # 平行视线向前看
        front = joint[57] - joint[23] + joint[56] - joint[24]
        front = front.astype(np.float64)
        front = front / np.linalg.norm(front)
        # 相机的view direction设置
        look_at = self.joints[1][frame_idx][9]-(joint[23]+joint[24])/2
        look_at = look_at.astype(np.float64)
        look_at = look_at / np.linalg.norm(look_at)
        # look_right
        leye_reye_dir = joint[23] - joint[24] 
        leye_reye_dir = leye_reye_dir.astype(np.float64)
        leye_reye_dir = leye_reye_dir / np.linalg.norm(leye_reye_dir)
        # look_up
        look_up_dir = np.cross(leye_reye_dir, look_at) 
        look_up_dir = look_up_dir.astype(np.float64)
        look_up_dir /= np.linalg.norm(look_up_dir)
        # only keep vertical componenet of look_at 
        
        look_at = np.cross(look_up_dir, leye_reye_dir)
        look_at = look_at.astype(np.float64)
        look_at /= np.linalg.norm(look_at)
        '''
        leye_reye_dir = np.cross(look_at,look_up_dir)
        leye_reye_dir = leye_reye_dir.astype(np.float64)
        leye_reye_dir/= np.linalg.norm(leye_reye_dir) 
        '''

        cam_pos = (joint[23] + joint[24]) / 2 + front*0.02   #左右眼中间
        # viewer.render_lock.acquire()
        #向pyrender中加入camera
        if self.camera_node is not None:
            self.scene.remove_node(self.camera_node)
        up = np.array([0,1,0])
        front = np.array([0,0,-1])
        right = np.cross(up, front)
        look_at_up = np.cross(look_at, leye_reye_dir)
        look_at_up = look_at_up.astype(np.float64)
        look_at_up /= np.linalg.norm(look_at_up)
        r1 = np.stack([leye_reye_dir, look_at_up, look_at])
        r2 = np.stack([right, up, front])
        quat = pyquat.Quaternion(matrix=(r1.T @ r2))
        quat_pyrender = [quat[1], quat[2], quat[3], quat[0]]

        camera_pose = np.eye(4)
        camera_pose[:3, :3] = r1.T @ r2   #旋转矩阵R
        camera_pose[:3, 3] = cam_pos  #平移矩阵t
        self.Rt = np.linalg.inv(camera_pose)

        self.camera_node = pyrender.Node(camera=self.camera, name='camera', rotation=quat_pyrender, translation=cam_pos)
        self.scene.add_node(self.camera_node)

        return look_at, cam_pos


    def add_scene(self,scene_mesh):
        m = pyrender.Mesh.from_trimesh(scene_mesh)
        object_node = pyrender.Node(mesh=m, name='scene')
        self.scene.add_node(object_node)
    
    def params2torch(self,params, dtype=torch.float32):
        return {k: torch.cuda.FloatTensor(v) if type(v) == np.ndarray else v for k, v in params.items()}

    def make_new_mesh(self,vt, f, ft, mesh, image):
        #增加人体纹理
        """
        Add missing vertices to the mesh such that it has the same number of vertices as the texture coordinates
        mesh: 3D vertices of the orginal mesh
        vt: 2D vertices of the texture map
        f: 3D faces of the orginal mesh (0-indexed)
        ft: 2D faces of the texture map (0-indexed)
        """
        #build a correspondance dictionary from the original mesh indices to the (possibly multiple) texture map indices
        f_flat = f.flatten()
        ft_flat = ft.flatten()
        correspondances = {}

        #traverse and find the corresponding indices in f and ft
        for i in range(len(f_flat)):
            if f_flat[i] not in correspondances:
                correspondances[f_flat[i]] = [ft_flat[i]]
            else:
                if ft_flat[i] not in correspondances[f_flat[i]]:
                    correspondances[f_flat[i]].append(ft_flat[i])

        #build a mesh using the texture map vertices
        new_mesh = np.zeros((vt.shape[0], 3))
        for old_index, new_indices in correspondances.items():
            for new_index in new_indices:
                new_mesh[new_index] = mesh[old_index]

        return trimesh.Trimesh(vertices=new_mesh, faces=ft, \
                                visual=trimesh.visual.TextureVisuals(uv=vt, image=image), process=True)
        #return trimesh.Trimesh(vertices=new_mesh,faces=ft,process=False)

def gen_one_file(filepath,filename,renderer:EgoviewGeneratorboxing):
    #file=np.load(os.path.join(data_folder,'027_ma_comb1_liu_comb2_joints_smpl.npy'),allow_pickle=True).item()
    file=np.load(filepath,allow_pickle=True).item()

    
    ethnicity = random.choice(["asian", "hispanic", "mideast", "white"])
    texture_paths = [tp[5:] for tp in body_texture_path if "m_" + ethnicity in tp]
    smpl_texture_path = random.choice(texture_paths).split('/')[-1] 
    
    sbj_names = list(file.keys())
    assert len(sbj_names) == 2
    origin_params1=file[sbj_names[0]]
    origin_params2=file[sbj_names[1]]

    # for frame_idx in tqdm.tqdm(range(len(origin_params1['transl']))):
    #     #if frame_idx>=2672:
    #     for key in origin_params1.keys():
    #         params1[key]=origin_params1[key][frame_idx:frame_idx+1,:]
    #         params2[key]=origin_params2[key][frame_idx:frame_idx+1,:]
            
    rgb,depth=renderer.gen_egoview(origin_params1,origin_params2,True,smpl_texture_path,filename)
    save_rgb_image(rgb,os.path.join(save_folder,filename[:-4]))

if __name__=="__main__":
    data_folder='EgoMoGen_data/boxing'
    #save_folder='/mnt/data/home/yangyihui/Datasets/output/egolocalmogen/render_boxing'
    save_folder='EgoMoGen_data/ego_render'

    body_texture_path = np.load('EgoMoGen_data/hood_data/body_texture_path.npy')
    body_texture = {}
    '''
    for tex_path in body_texture_path:
        # im = Image.alpha_composite(Image.open(tex_path), eye_img)
        # im.save("/mnt/vlg-nfs/genli/datasets/bedlam/" + tex_path.split('/')[-1])
        body_texture[tex_path.split('/')[-1]] = Image.open(tex_path[5:])
    '''
        
    renderer=EgoviewGeneratorboxing()

    for dir_name in tqdm.tqdm(os.listdir(data_folder)):
        if '20240412' in dir_name:
            continue
        for file_name in os.listdir(os.path.join(data_folder,dir_name,'smpl_params')):
            if '.npy' in file_name:
                gen_one_file(os.path.join(data_folder,dir_name,'smpl_params',file_name),file_name,renderer)
            
