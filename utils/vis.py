import open3d as o3d


def vis(points):
    """
    Visualization
    :param points: np array of points
    :return: None
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([point_cloud])


if __name__ == '__main__':
    import numpy as np

    pc = np.load("../data/e3a631d5ec512e997827c1e59b032cf8.npy")
    vis(pc)
