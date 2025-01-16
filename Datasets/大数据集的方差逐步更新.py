n = 0
mean = 0
M2 = 0
for path in tqdm.tqdm(image_path):
    image = np.load(path).astype(np.float32)[downsample:]
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

var = M2 / (n-1)
mean = mean