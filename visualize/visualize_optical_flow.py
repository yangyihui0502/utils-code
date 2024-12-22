import numpy as np
import cv2

def visualize_flow(flow):
    # flow是形状为(H, W, 2)的光流图像，包含水平(U)和垂直(V)分量

    # 计算光流的角度和幅度——flow需要是float32/64！
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # 创建HSV图像
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2  # 角度映射到色相
    hsv[..., 1] = 255  # 饱和度固定为255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # 幅度归一化为亮度

    # 转换为BGR图像
    bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return bgr_flow

# 示例
flow = np.random.randn(224, 224, 2)  # 假设这是计算得到的光流
flow_vis = visualize_flow(flow)

cv2.imshow("Optical Flow Visualization", flow_vis)
cv2.waitKey(0)
cv2.destroyAllWindows()