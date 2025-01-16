import argparse
import os

from pathlib import Path
import yaml
from tqdm import tqdm
import time
import datetime
import random

import torch
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from torch.utils import data

import pytorch3d.transforms as transforms 

from egomogen.scripts.visualize_dataset import visualize_output
from ema_pytorch import EMA

from egomogen.utils.egoego.models.image_encoder import ImageEncoder
from egomogen.utils.data_utils import *
from egomogen.scripts.boxing_render import EgoviewGeneratorboxing

from egomogen.utils.egoego.runners.recorders import EgoMoGenTensorboardRecorder

import sys
sys.path.append('.')
from egomogen.dataloaders.datasets.boxing import BoxingDataset

from egomogen.utils.egoego.network.egoego_network import CondGaussianDiffusion

from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.console_utils import display_table

def cycle(dl):
    while True:
        for data in dl:
            yield data

class Trainer(object):
    def __init__(
        self,
        opt,
        diffusion_model,
        *,
        ema_decay = 0.995,
        train_batch_size = 32,
        train_lr = 1e-4,
        train_num_steps = 10000000,
        gradient_accumulate_every = 2,
        input_frames = 5,
        amp = False,
        step_start_ema = 2000,
        ema_update_every = 10,
        save_and_sample_every = 50,
        results_folder = './results',
        use_wandb=False,
        run_demo=False,
    ):
        super().__init__()

        self.use_wandb = use_wandb           
        if self.use_wandb:
            # Loggers
            wandb.init(config=opt, project='train_full_body_model', name='train_full_body_model', dir=opt.save_dir)

        self.encoder = ImageEncoder(opt, device=opt.device).to('cuda')
        if opt.split == 'train':
            self.recorder = EgoMoGenTensorboardRecorder(record_dir=f'EgoMoGen_data/record/{opt.record_exp}', resume=False)

        self.smplx_model = make_smplx('boxing')
        self.renderer = EgoviewGeneratorboxing(device='cuda')
        self.body_texture_path = np.load('EgoMoGen_data/hood_data/body_texture_path.npy')

        self.model = diffusion_model.to('cuda')
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every
        self.log_interval = opt.log_interval
        self.record_interval = opt.record_interval
        self.epoch_iter = opt.epoch_iter

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.input_frames = opt.input_frames

        self.optimizer = Adam(list(diffusion_model.parameters())+list(self.encoder.parameters()), lr=train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)

        self.results_folder = results_folder

        self.vis_folder = results_folder.replace("weights", "vis_res")

        self.opt = opt 

        if run_demo:
            self.ds = BoxingDataset(self.opt, split='train', folder=opt.data_folder, block_size=opt.block_size, run_demo=True) 
        else:
            self.prep_dataloader(window_size=opt.block_size)

        self.window = opt.window 

        # self.bm_dict = self.ds.bm_dict 

    def prep_dataloader(self, window_size):   # 准备数据集
        # Define dataset
        train_dataset = BoxingDataset(self.opt, split='train', folder=opt.data_folder, block_size=opt.block_size, run_demo=True) 
        val_dataset = BoxingDataset(self.opt, split='test', folder=opt.data_folder, test_frame_size=opt.test_size, run_demo=True) 

        self.ds = train_dataset 
        self.val_ds = val_dataset
        self.dl = data.DataLoader(self.ds, batch_size=self.batch_size, shuffle=True, pin_memory=False, num_workers=0)
        self.val_dl = data.DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=True, pin_memory=False, num_workers=0)
        # self.dl = cycle(data.DataLoader(self.ds, batch_size=self.batch_size, shuffle=True, pin_memory=False, num_workers=0))
        # self.val_dl = cycle(data.DataLoader(self.ds, batch_size=self.batch_size, shuffle=True, pin_memory=False, num_workers=0))

    def save(self, iter, milestone):
        data = {
            'iter': iter,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        save_path = os.path.join(self.results_folder, 'model-'+str(milestone)+'.pt')
        torch.save(data, save_path)
        print(f'Save model-{str(milestone)}.pt to\n{save_path}')

    def load(self, milestone):
        data = torch.load(os.path.join(self.results_folder, 'model-'+str(milestone)+'.pt'))

        self.step = data['iter']
        self.model.load_state_dict(data['model'], strict=False)
        self.ema.load_state_dict(data['ema'], strict=False)
        self.scaler.load_state_dict(data['scaler'])

    def select_texture(self):
        ethnicity = random.choice(["asian", "hispanic", "mideast", "white"])
        texture_paths = [tp[5:] for tp in self.body_texture_path if "m_" + ethnicity in tp]
        smpl_texture_path = random.choice(texture_paths).split('/')[-1] 
        return smpl_texture_path

    def load_weight_path(self, weight_path):
        data = torch.load(weight_path)

        self.step = data['iter']
        self.model.load_state_dict(data['model'], strict=False)
        self.ema.load_state_dict(data['ema'], strict=False)
        self.scaler.load_state_dict(data['scaler'])

    def train(self, continue_train=False):
        # torch.cuda.empty_cache()
        if continue_train:
            weights = os.listdir(self.results_folder)
            weights_paths = [os.path.join(self.results_folder, weight) for weight in weights]
            weight_path = max(weights_paths, key=os.path.getctime)
    
            print(f"Loaded weight: {weight_path}")

            milestone = weight_path.split("/")[-1].split("-")[-1].replace(".pt", "")
        
            self.load(milestone)
            self.ema.ema_model.train()
        
        print('begin to train!')
        init_step = self.step 
        epoch = init_step % self.epoch_iter
        start_time = time.perf_counter()

        for iter in range(init_step, self.train_num_steps):
            self.optimizer.zero_grad()

            nan_exists = False # If met nan in loss or gradient, need to skip to next data. 
            for idx,data_dict in enumerate(self.dl):
                if idx == self.gradient_accumulate_every:
                    break
                # next函数为从迭代器中获得下一个元素
                # data_dict = next(self.dl)
                # data = data_dict['motion'].cuda()

                # padding_mask = self.prep_padding_mask(data, data_dict['seq_len'])

                pose_series = data_dict['pose_series']
                view = data_dict['view']
                padding_mask = torch.ones((view.shape[0], view.shape[1]+1), device=view.device)
                padding_mask[:, 1:] = data_dict['mask']
                image = data_dict['image']

                pose_feats = torch.concat((pose_series, view), dim=-1)
                image_feats = self.encoder(image)
                feats = torch.zeros((image.shape[0], image.shape[1]-self.input_frames+1, (pose_feats.shape[-1]+image_feats.shape[-1])*self.input_frames), device=image.device)
                self.pose_len = 268

                for frame_idx in range(self.input_frames-1, image.shape[1], 1):
                    pose_feats_seq = []
                    image_feats_seq = []

                    for frame_number in range(frame_idx-self.input_frames+1, frame_idx+1, 1):
                        pose_feats_seq.append(pose_feats[:, frame_number])
                        image_feats_seq.append(image_feats[:, frame_number])
                    pose_feats_seq = torch.stack(pose_feats_seq).permute(1,0,2).reshape(feats.shape[0], -1)
                    image_feats_seq = torch.stack(image_feats_seq).permute(1,0,2).reshape(feats.shape[0], -1)    
                    
                    feats[:, frame_idx-self.input_frames+1] = torch.concat((pose_feats_seq, image_feats_seq), dim=-1)
                padding_mask = padding_mask[:, self.input_frames-1:]

                with autocast(enabled = self.amp):
                    # 创建cond_mask，作用于当前帧动作
                    cond_mask = self.prep_condition_mask(feats, self.pose_len) # BS X T' X D 
                    # 随机取样一次，预测噪声并计算loss
                    loss_diffusion = self.model(feats, cond_mask, self.pose_len, padding_mask=padding_mask[:, None])
                    
                    loss = loss_diffusion

                    if torch.isnan(loss).item():
                        print('WARNING: NaN loss. Skipping to next data...')
                        nan_exists = True 
                        torch.cuda.empty_cache()
                        continue

                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()
                    
                    # check gradients
                    parameters = [p for p in self.model.parameters() if p.grad is not None]
                    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).to(view.device) for p in parameters]), 2.0)
                    if torch.isnan(total_norm):
                        print('WARNING: NaN gradients. Skipping to next data...')
                        nan_exists = True 

                        torch.cuda.empty_cache()
                        continue

                    # if self.use_wandb:
                    #     log_dict = {
                    #         "Train/Loss/Total Loss": loss.item(),
                    #         "Train/Loss/Diffusion Loss": loss_diffusion.item(),
                    #     }
                    #     wandb.log(log_dict)

            if nan_exists:
                continue

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.ema.update()

            end_time = time.perf_counter()
            batch_time = end_time - start_time
            start_time = end_time  # note that all logging and profiling time are accumuated into data_time
            
            if iter % self.log_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']  # MARK: skechy lr query, only lr of the first param will be saved
                max_mem = torch.cuda.max_memory_allocated() / 2**20
                scalar_stats = dotdict(
                    batch = batch_time,
                    lr = lr,
                    max_mem = max_mem,
                    loss = loss.item()
                    )

                self.recorder.iter = iter
                self.recorder.epoch = epoch
                self.recorder.update_scalar_stats(scalar_stats)

                # For logging
                eta_seconds = self.recorder.scalar_stats.batch.global_avg * (self.train_num_steps - self.recorder.iter)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                log_stats = dotdict()
                log_stats.eta = eta_string
                log_stats.update(self.recorder.log_stats)

                # Render table to screen
                display_table(log_stats)  # render dict as a table (live, console, table)

            if iter % self.record_interval == 0:
                self.recorder.record('train')

            # save checkpoint
            if iter != 0 and iter % self.save_and_sample_every == 0:
                milestone = iter // self.save_and_sample_every
                self.save(iter, milestone)
            '''
            eval and visualize:
                self.ema.ema_model.eval()
                with torch.no_grad():
                  
                    data_dict = next(self.val_dl)

                    pose_series = data_dict['pose_series']
                    view = data_dict['view']
                    padding_mask = torch.ones((view.shape[0], view.shape[1]+1))
                    padding_mask[:, 1:] = data_dict['mask']
                    image = data_dict['image']

                    pose_feats = torch.concat((pose_series, view), dim=-1)
                    image_feats = self.encoder(image)
                    feats = torch.zeros((image.shape[0], image.shape[1]-self.input_frames+1, (pose_feats.shape[-1]+image_feats.shape[-1])*self.input_frames), device=image.device)
                    self.pose_len = pose_feats.shape[-1]

                    for frame_idx in range(self.input_frames-1, image.shape[1], 1):
                        pose_feats_seq = []
                        image_feats_seq = []

                        for frame_number in range(frame_idx-self.input_frames+1, frame_idx+1, 1):
                            pose_feats_seq.extend(pose_feats[:, frame_number])
                            image_feats_seq.extend(image_feats[:, frame_number])
                        pose_feats_seq = torch.stack(pose_feats_seq).reshape(feats.shape[0], -1)
                        image_feats_seq = torch.stack(image_feats_seq).reshape(feats.shape[0], -1)    
                        
                        feats[:, frame_idx-self.input_frames+1] = torch.concat((pose_feats_seq, image_feats_seq), dim=-1)
                    padding_mask = padding_mask[:, self.input_frames-1:]

                    cond_mask = self.prep_condition_mask(feats) # BS X T X D 
                    
                    all_res_list = self.ema.ema_model.sample(feats, cond_mask, padding_mask=padding_mask)
                

                # Visualization
                bs_for_vis = 4
                for_vis_gt_data = feats[:bs_for_vis]
                self.gen_vis_res(for_vis_gt_data, self.step, vis_gt=True)

                self.gen_vis_res(all_res_list[:bs_for_vis], self.step)
                '''

            if (iter + 1) % self.epoch_iter == 0:
                # Actual start of the execution
                epoch = epoch + 1

        print('training complete')

        if self.use_wandb:
            wandb.run.finish()
    
    def prep_condition_mask(self, data, pose_len):
        # data: BS X T' X D  
        # Condition part is zeros, while missing part is ones. 
        mask = torch.ones_like(data).to(data.device)

        current_pose_sid = (pose_len + 9) * (self.input_frames-1)
        mask[:, :, current_pose_sid:current_pose_sid+pose_len] = torch.zeros(data.shape[0], data.shape[1], pose_len).to(data.device)

        return mask

    def prep_padding_mask(self, val_data, seq_len):
        # Generate padding mask 
        actual_seq_len = seq_len + 1 # BS, + 1 since we need additional timestep for noise level 
        tmp_mask = torch.arange(self.window+1).expand(val_data.shape[0], \
        self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
        # BS X max_timesteps
        padding_mask = tmp_mask[:, None, :].to(val_data.device)

        return padding_mask 

    def cond_sample_res(self):
        weights = os.listdir(self.results_folder)
        weights_paths = [os.path.join(self.results_folder, weight) for weight in weights]
        weight_path = max(weights_paths, key=os.path.getctime)
    
        print(f"Loaded weight: {weight_path}")

        milestone = weight_path.split("/")[-1].split("-")[-1].replace(".pt", "")
        
        self.load(milestone)
        self.ema.ema_model.eval()
        num_sample = 1  
        with torch.no_grad():
            # for s_idx in range(num_sample):
                # val_data_dict = next(self.val_dl)
                # val_data_dict = next(self.dl)
            for idx, val_data_dict in enumerate(self.val_dl):
                if idx == num_sample:
                    break

                pose_series = val_data_dict['pose_series']
                oppo_params = val_data_dict['meta'].oppo_params
                view = val_data_dict['view']
                padding_mask = torch.ones((view.shape[0], view.shape[1]+1), device=view.device)
                padding_mask[:, 1:] = val_data_dict['mask']
                image = val_data_dict['image'].clone()
                meta = val_data_dict['meta']

                pose_feats = torch.concat((pose_series, view), dim=-1)
                image_feats = self.encoder(image)
                feats = torch.zeros((image.shape[0], 1, (pose_feats.shape[-1]+image_feats.shape[-1])*self.input_frames), device=image.device)
                self.pose_len = 268
                # padding_mask = padding_mask[:, self.input_frames-1:]
                all_pos_output = torch.zeros((image.shape[0], image.shape[1]-self.input_frames+1, self.pose_len), device=image.device)
                all_image = torch.zeros((image.shape[0],image.shape[1]-self.input_frames+1, 224, 224, 3))
                all_ego_mesh = []
                all_oppo_mesh = []
                all_current_pos = []

                texture_path = self.select_texture()
                camera_norm = torch.from_numpy(np.load('EgoMoGen_data/normalize/camera_trans_norm.npy')).to('cuda')

                for frame_idx in tqdm.tqdm(range(self.input_frames-1, image.shape[1], 1), desc='generating the motion'):
                    pose_feats_seq = []
                    image_feats_seq = []

                    for frame_number in range(frame_idx-self.input_frames+1, frame_idx+1, 1):
                        pose_feats_seq.append(pose_feats[:, frame_number])  # masked later
                        image_feats_seq.append(image_feats[:, frame_number])
                    pose_feats_seq = torch.stack(pose_feats_seq).permute(1,0,2).reshape(feats.shape[0], -1)
                    image_feats_seq = torch.stack(image_feats_seq).permute(1,0,2).reshape(feats.shape[0], -1)    
                    
                    feats[:] = torch.concat((pose_feats_seq, image_feats_seq), dim=-1).unsqueeze(1)
                    current_padding_mask = padding_mask[:, frame_idx, None, None]
                    cond_mask = self.prep_condition_mask(feats, self.pose_len)

                    # generate motion and save to feats
                    all_pos_output[:, frame_idx-self.input_frames+1:frame_idx-self.input_frames+2] = self.ema.ema_model.sample(x_start=feats, cond_mask=cond_mask, padding_mask=current_padding_mask)
                    if frame_idx == self.input_frames-1:
                    # if True:
                        current_pos, current_rot, init_rot_mat, loc_pos = recover_data(all_pos_output[:, frame_idx-self.input_frames+1].unsqueeze(1), meta.init_root_mat.double())
                    else:
                        current_pos, current_rot, init_rot_mat, loc_pos = recover_data(all_pos_output[:, frame_idx-self.input_frames+1].unsqueeze(1), init_rot_mat[:, 0])       
                    
                    # current_rot[:, 0, 0, :, 1] = torch.tensor([0, 1, 0])
                    # init_rot_mat[:, 0, :3, 1] = torch.tensor([0, 1, 0])

                    # current_rot = matrix.normalized_matrix(current_rot)
                    # init_rot_mat = torch.eye(4).repeat(cond_mask.shape[0], 1, 1)
                    # init_rot_mat[:, :3, :3] = current_rot[:, 0, 0]
                    # init_rot_mat[:, :3, 3] = current_pos[:, 0, 0]
                    last_rgb_image = torch.zeros((cond_mask.shape[0], 224, 224, 3))

                    for batch_idx in range(cond_mask.shape[0]):
                        vertices, faces, ego_joints= skeleton2mesh(current_pos[batch_idx][0], current_rot[batch_idx][0], parents)
                        ego_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

                        smplx_output = self.smplx_model(
                            global_orient = np.double(oppo_params.global_orient[batch_idx, frame_idx].reshape(1,-1)),
                            transl = np.double(oppo_params.transl[batch_idx, frame_idx].reshape(1,-1)),
                            body_pose = np.double(oppo_params.body_pose[batch_idx, frame_idx].reshape(1,-1,3)),
                        )
                        oppo_mesh = trimesh.Trimesh(vertices=smplx_output.vertices[0], faces=self.smplx_model.bm.faces)

                        if batch_idx == 0:
                            all_current_pos.append(current_pos[0])
                            all_ego_mesh.append(ego_mesh)
                            all_oppo_mesh.append(oppo_mesh)
                        # "render and save image to image_feats"
                        rot_matrix, cam_pos, rgb_image = self.renderer.gen_egoview_from_mesh(ego_mesh, oppo_mesh, ego_joints[0], smplx_output.joints[0][9], texture_path, with_texture=False)
                        normalized_cam_pos = Normalize(cam_pos, camera_norm)

                        last_rgb_image[batch_idx] = rgb_image
                        all_image[batch_idx, frame_idx-self.input_frames+1] = rgb_image

                        # 给定第一张的flow，后面图片render得到
                        if frame_idx == self.input_frames-1:
                            normalized_flow = image[batch_idx, frame_idx]
                        else:
                            flow = rgb2flow(rgb_image, last_rgb_image[batch_idx])
                            normalized_flow = self.normalize_flow(flow)

                        image_feats[batch_idx, frame_idx] = self.encoder(normalized_flow[None][None])
                        pose_feats[batch_idx, frame_idx] = torch.concat((all_pos_output[batch_idx][frame_idx-self.input_frames+1], transforms.matrix_to_rotation_6d(rot_matrix[:3,:3]), normalized_cam_pos))
                
                # data = {
                #     'image':all_image,
                #     'view': None,
                #     'pose_series': all_pos_output,
                #     'meta': meta
                # }
                # visualize_dataset(data, image_type='rgb')
                visualize_output(all_pos_output, meta, input_frames=self.input_frames, image=all_image, mesh1=all_ego_mesh, mesh2=all_oppo_mesh, vis_output_mesh=True, current_pos=all_current_pos, image_type='rgb', vis_gt=True, gt=val_data_dict, vis_mesh=True, loc_pos=loc_pos)


    def normalize_flow(self, flow):
        shape = flow.shape
        mean, var = np.load('EgoMoGen_data/normalize/image_flow_norm.npy').astype(np.float32)
        normalized_flow = Normalize(flow.reshape(-1), (mean,var)).reshape(shape)

        return torch.from_numpy(normalized_flow)


    def full_body_gen_cond_head_pose_sliding_window(self, head_pose, seq_name):
        # head_pose: BS X T X 7 
        self.ema.ema_model.eval()

        global_head_jpos = head_pose[:, :, :3] # BS X T X 3 
        global_head_quat = head_pose[:, :, 3:] # BS X T X 4 

        data = torch.zeros(head_pose.shape[0], head_pose.shape[1], 22*3+22*6).to(head_pose.device) # BS X T X D 

        with torch.no_grad():
            cond_mask = self.prep_head_condition_mask(data) # BS X T X D 

            local_aa_rep, seq_root_pos = self.ema.ema_model.sample_sliding_window_w_canonical(self.ds, \
            global_head_jpos, global_head_quat, x_start=data, cond_mask=cond_mask) 
            # BS X T X 22 X 3, BS X T X 3       

        return local_aa_rep, seq_root_pos # T X 22 X 3, T X 3  

    def gen_vis_res(self, all_res_list, step, vis_gt=False):
        # all_res_list: N X T X D 
        num_seq = all_res_list.shape[0]
        normalized_global_jpos = all_res_list[:, :, :22*3].reshape(num_seq, -1, 22, 3)

        global_jpos = self.ds.de_normalize_jpos_min_max(normalized_global_jpos.reshape(-1, 22, 3))
      
        global_jpos = global_jpos.reshape(num_seq, -1, 22, 3) # N X T X 22 X 3 
        global_root_jpos = global_jpos[:, :, 0, :].clone() # N X T X 3 

        global_rot_6d = all_res_list[:, :, 22*3:].reshape(num_seq, -1, 22, 6)
        
        global_rot_mat = transforms.rotation_6d_to_matrix(global_rot_6d) # N X T X 22 X 3 X 3 

        for idx in range(num_seq):
            curr_global_rot_mat = global_rot_mat[idx] # T X 22 X 3 X 3 
            curr_local_rot_mat = quat_ik_torch(curr_global_rot_mat) # T X 22 X 3 X 3 
            curr_local_rot_aa_rep = transforms.matrix_to_axis_angle(curr_local_rot_mat) # T X 22 X 3 
            
            curr_global_root_jpos = global_root_jpos[idx] # T X 3
            move_xy_trans = curr_global_root_jpos.clone()[0:1] # 1 X 3 
            move_xy_trans[:, 2] = 0 
            root_trans = curr_global_root_jpos - move_xy_trans # T X 3 

            # Generate global joint position 
            bs = 1
            betas = torch.zeros(bs, 16).to(root_trans.device)
            gender = ["male"] * bs 

            mesh_jnts, mesh_verts, mesh_faces = \
            run_smpl_model(root_trans[None], \
            curr_local_rot_aa_rep[None], betas, gender, \
            self.bm_dict)
            # BS(1) X T' X 22 X 3, BS(1) X T' X Nv X 3
            
            dest_mesh_vis_folder = os.path.join(self.vis_folder, "blender_mesh_vis", str(step))
            if not os.path.exists(dest_mesh_vis_folder):
                os.makedirs(dest_mesh_vis_folder)

            if vis_gt:
                mesh_save_folder = os.path.join(dest_mesh_vis_folder, \
                                "objs_step_"+str(step)+"_bs_idx_"+str(idx)+"_gt")
                out_rendered_img_folder = os.path.join(dest_mesh_vis_folder, \
                                "imgs_step_"+str(step)+"_bs_idx_"+str(idx)+"_gt")
                out_vid_file_path = os.path.join(dest_mesh_vis_folder, \
                                "vid_step_"+str(step)+"_bs_idx_"+str(idx)+"_gt.mp4")
            else:
                mesh_save_folder = os.path.join(dest_mesh_vis_folder, \
                                "objs_step_"+str(step)+"_bs_idx_"+str(idx))
                out_rendered_img_folder = os.path.join(dest_mesh_vis_folder, \
                                "imgs_step_"+str(step)+"_bs_idx_"+str(idx))
                out_vid_file_path = os.path.join(dest_mesh_vis_folder, \
                                "vid_step_"+str(step)+"_bs_idx_"+str(idx)+".mp4")

            # Visualize the skeleton 
            if vis_gt:
                dest_skeleton_vis_path = os.path.join(dest_mesh_vis_folder, \
                                "vid_step_"+str(step)+"_bs_idx_"+str(idx)+"_skeleton_gt.gif")
            else:
                dest_skeleton_vis_path = os.path.join(dest_mesh_vis_folder, \
                                "vid_step_"+str(step)+"_bs_idx_"+str(idx)+"_skeleton.gif")

            channels = global_jpos[idx:idx+1] # 1 X T X 22 X 3 
            # show3Dpose_animation_smpl22(channels.data.cpu().numpy(), dest_skeleton_vis_path) 

            # For visualizing human mesh only 
            save_verts_faces_to_mesh_file(mesh_verts.data.cpu().numpy()[0], mesh_faces.data.cpu().numpy(), mesh_save_folder)
            run_blender_rendering_and_save2video(mesh_save_folder, out_rendered_img_folder, out_vid_file_path)

    def gen_full_body_vis(self, root_trans, curr_local_rot_aa_rep, dest_mesh_vis_folder, seq_name, vis_gt=False):
        # root_trans: T X 3 
        # curr_local_rot_aa_rep: T X 22 X 3 

        # Generate global joint position 
        bs = 1
        betas = torch.zeros(bs, 16).to(root_trans.device)
        gender = ["male"] * bs 

        mesh_jnts, mesh_verts, mesh_faces = run_smpl_model(root_trans[None].float(), \
        curr_local_rot_aa_rep[None].float(), betas.float(), gender, self.ds.bm_dict)
        # BS(1) X T' X 22 X 3, BS(1) X T' X Nv X 3
    
        if vis_gt:
            mesh_save_folder = os.path.join(dest_mesh_vis_folder, seq_name, \
                            "objs_gt")
            out_rendered_img_folder = os.path.join(dest_mesh_vis_folder, seq_name, \
                            "imgs_gt")
            out_vid_file_path = os.path.join(dest_mesh_vis_folder, \
                            seq_name+"_vid_gt.mp4")
        else:
            mesh_save_folder = os.path.join(dest_mesh_vis_folder, seq_name, \
                            "objs")
            out_rendered_img_folder = os.path.join(dest_mesh_vis_folder, seq_name, \
                            "imgs")
            out_vid_file_path = os.path.join(dest_mesh_vis_folder, \
                            seq_name+"_vid.mp4")

        # For visualizing human mesh only 
        save_verts_faces_to_mesh_file(mesh_verts.data.cpu().numpy()[0], \
        mesh_faces.data.cpu().numpy(), mesh_save_folder)
        run_blender_rendering_and_save2video(mesh_save_folder, \
        out_rendered_img_folder, out_vid_file_path)

        return mesh_jnts, mesh_verts 

def run_train(opt, device):
    # Prepare Directories
    # save_dir = Path(opt.save_dir)
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)

    # Save run settings
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=True)

    # Define model  
    repr_dim = 3945
    out_dim = 268
    input_frames = opt.input_frames
  
    loss_type = "l1"
    
    diffusion_model = CondGaussianDiffusion(d_feats=repr_dim, d_model=opt.d_model, \
                n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v, \
                input_frames=input_frames, max_timesteps=opt.window+1, out_dim=out_dim, timesteps=1000, \
                objective="pred_x0", loss_type=loss_type, \
                batch_size=opt.batch_size)
  
    diffusion_model.to(device)

    trainer = Trainer(
        opt,
        diffusion_model,
        train_batch_size=opt.batch_size, # 32
        train_lr=opt.learning_rate, # 1e-4
        # train_num_steps=700000,         # 700000, total training steps
        train_num_steps=opt.train_steps,
        gradient_accumulate_every=2,    # gradient accumulation steps
        input_frames=input_frames,
        ema_decay=0.995,                # exponential moving average decay
        amp=True,                        # turn on mixed precision
        results_folder=str(wdir),
    )

    trainer.train(opt.continue_train)

    torch.cuda.empty_cache()

def run_sample(opt, device):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'

    # Define model     
    repr_dim = 3945
    out_dim = 268
   
    loss_type = "l1"
  
    diffusion_model = CondGaussianDiffusion(d_feats=repr_dim, d_model=opt.d_model, \
                input_frames=opt.input_frames, n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v, \
                max_timesteps=opt.window+1, out_dim=out_dim, timesteps=1000, \
                objective="pred_x0", loss_type=loss_type, \
                batch_size=opt.batch_size)

    diffusion_model.to(device)

    trainer = Trainer(
        opt,
        diffusion_model,
        train_batch_size=opt.batch_size, # 32
        train_lr=opt.learning_rate, # 1e-4
        train_num_steps=800,         # 700000, total training steps
        gradient_accumulate_every=2,    # gradient accumulation steps
        ema_decay=0.995,                # exponential moving average decay
        amp=True,                        # turn on mixed precision
        results_folder=str(wdir),
        use_wandb=False 
    )
  
    trainer.cond_sample_res()

    torch.cuda.empty_cache()

def get_trainer(opt, run_demo=False):
    opt.window = opt.diffusion_window 

    opt.diffusion_save_dir = os.path.join(opt.diffusion_project, opt.diffusion_exp_name)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    # Prepare Directories
    save_dir = Path(opt.diffusion_save_dir)
    wdir = save_dir / 'weights'

    # Define model 
    repr_dim = 3945
    out_dim = 268
   
    transformer_diffusion = CondGaussianDiffusion(d_feats=repr_dim, d_model=opt.diffusion_d_model, \
                input_frames=opt.input_frames, n_dec_layers=opt.diffusion_n_dec_layers, n_head=opt.diffusion_n_head, \
                d_k=opt.diffusion_d_k, d_v=opt.diffusion_d_v, \
                max_timesteps=opt.diffusion_window+1, out_dim=out_dim, timesteps=1000, objective="pred_x0", \
                batch_size=opt.diffusion_batch_size)

    transformer_diffusion.to(device)

    trainer = Trainer(
        opt,
        transformer_diffusion,
        train_batch_size=opt.diffusion_batch_size, # 32
        train_lr=opt.diffusion_learning_rate, # 1e-4
        train_num_steps=8000000,         # 700000, total training steps
        gradient_accumulate_every=2,    # gradient accumulation steps
        ema_decay=0.995,                # exponential moving average decay
        amp=True,                        # turn on mixed precision
        results_folder=str(wdir),
        use_wandb=False,
        run_demo=run_demo,
    )

    return trainer 

def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--project', default="exp/stage2_motion_diffusion_amass_runs/train", help='project/name')
    # parser.add_argument('--wandb_pj_name', type=str, default="stage2_cond_motion_diffusion_amass", help='project name')
    # parser.add_argument('--entity', default="yihuiyang", help='W&B entity')
    # parser.add_argument('--exp_name', default="stage2_cond_motion_diffusion_amass_set1", help='save to project/name')
    parser.add_argument('--save_dir', default="EgoMoGen_data/egoego")

    parser.add_argument('--record_exp', default="egoego_small_sample", help='exp name to save record')
    parser.add_argument('--device', default='cuda', help='cuda device')

    parser.add_argument('--data_folder', default="EgoMoGen_data", help='')

    parser.add_argument('--window', type=int, default=100, help='horizon')
    parser.add_argument('--block_size', type=int, default=40)
    parser.add_argument('--test_size', type=int, default=120)

    parser.add_argument('--input_of_feats', default=False)
    parser.add_argument('--input_frames', default=5)
    parser.add_argument('--train_batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=4e-4, help='generator_learning_rate')
    parser.add_argument('--freeze_of_cnn', default=False)

    parser.add_argument('--checkpoint', type=str, default="", help='checkpoint')
    parser.add_argument('--continue_train', default=False)

    parser.add_argument('--train_steps', default=100000)
    parser.add_argument('--n_dec_layers', type=int, default=4, help='the number of decoder layers')
    parser.add_argument('--n_head', type=int, default=4, help='the number of heads in self-attention')
    parser.add_argument('--d_k', type=int, default=256, help='the dimension of keys in transformer')
    parser.add_argument('--d_v', type=int, default=256, help='the dimension of values in transformer')
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of intermediate representation in transformer')
    
    # For testing sampled results 
    parser.add_argument("--split", default="test")
    parser.add_argument("--record_interval", default=10)
    parser.add_argument("--log_interval", default=5)
    parser.add_argument("--epoch_iter", default=1000)

    # For data representation
    parser.add_argument("--canonicalize_init_head", action="store_true",default=True)
    parser.add_argument("--use_min_max", action="store_true")

    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    # opt.save_dir = os.path.join(opt.project, opt.exp_name)
    opt.exp_name = opt.save_dir.split('/')[-1]
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    if opt.split == 'test':
        opt.batch_size = opt.test_batch_size
        run_sample(opt, device)
    else:
        opt.batch_size = opt.train_batch_size
        run_train(opt, device)
