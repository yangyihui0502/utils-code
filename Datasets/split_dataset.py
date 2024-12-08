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