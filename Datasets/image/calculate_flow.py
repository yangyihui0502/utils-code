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