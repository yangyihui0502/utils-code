def __len__(self):
        if self.split == 'train':
            return len(self.data)*100   # 数据集较小，*100来扩大数据集
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
    if pose_eid > len(wholeseq.pose_series):    # 做padding
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

def pad_motion(self, pose_series, pose_sid, data_len):
        padding_motion = []
 
        padding_motion.extend(pose_series[pose_sid:data_len])
        padding_motion.extend(pose_series[data_len-1].reshape(1,-1).repeat(self.block_size-data_len+pose_sid,1))
        return padding_motion