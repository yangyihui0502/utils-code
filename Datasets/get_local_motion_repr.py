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