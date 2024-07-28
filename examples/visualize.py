import open3d as o3d

import numpy as np
import random


def get_ball(num_points, r):
    point_cloud = []
    for i in range(num_points):
        t = random.random()
        t = np.arcsin(1 - 2 * t)
        u = random.random() * 2 * np.pi - np.pi
        x = np.cos(t) * np.cos(u) * r
        y = np.cos(t) * np.sin(u) * r
        z = np.sin(t) * r
        point_cloud.append([x, y, z])
    return np.array(point_cloud)


# 点群作成
num_points = 1000
r = 0.5
test_data = get_ball(num_points, r)
print(test_data.shape)

points = np.fromfile(
    "data/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin",
    dtype=np.float32,
).reshape(-1, 5)[:, :3]

print(points.shape)

# numpy open3d に変換
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Open3dで可視化
o3d.visualization.draw_geometries([pcd])
