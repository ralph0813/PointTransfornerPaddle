import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np


class KNN:
    def __init__(self, k):
        self.k = k

    def __call__(self, query, ref):
        B, N, C = ref.shape
        _, M, _ = query.shape
        ref = ref.reshape((B, N, 1, C))
        query = query.reshape((B, 1, M, C))
        dist = paddle.sum((ref - query) ** 2, axis=-1)
        idx = dist.topk(k=self.k, axis=-1, largest=False)[1]  # (B, N, k)
        return dist, idx


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    xyz = xyz.numpy()
    B, N, C = xyz.shape
    centroids = np.zeros((B, npoint), dtype=np.int64)
    distance = np.ones((B, N)) * 1e10
    farthest = paddle.randint(0, N, (B,), dtype=np.int64).numpy()
    batch_indices = paddle.arange(B, dtype=np.int64).numpy()
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].reshape((B, 1, 3))
        dist = np.sum((xyz - centroid) ** 2, axis=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    return centroids


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points: indexed points data, [B, S, C]
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = paddle.arange(B, dtype=paddle.int64).reshape(view_shape).tile(repeat_shape).numpy()
    points = points.numpy()
    new_points = points[batch_indices, idx, :]
    return paddle.to_tensor(new_points)


def fps(data, number):
    fps_idx = farthest_point_sample(data, number)
    fps_data = index_points(data, fps_idx)
    return fps_data


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * paddle.matmul(src, dst.transpose((0, 2, 1)))
    dist += paddle.sum(src ** 2, -1).reshape((B, N, 1))
    dist += paddle.sum(dst ** 2, -1).reshape((B, 1, M))
    return dist


class PointNetFeaturePropagation(nn.Layer):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.LayerList()
        self.mlp_bns = nn.LayerList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1D(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1D(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.transpose((0, 2, 1))
        xyz2 = xyz2.transpose((0, 2, 1))

        points2 = points2.transpose((0, 2, 1))
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists= paddle.sort(dists, axis=-1)
            idx = paddle.argsort(dists, axis=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = paddle.sum(dist_recip, axis=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = paddle.sum(index_points(points2, idx) * weight.reshape((B, N, 3, 1)), axis=2)

        if points1 is not None:
            points1 = points1.transpose((0, 2, 1))
            new_points = paddle.concat([points1, interpolated_points], axis=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.transpose((0, 2, 1))
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


if __name__ == '__main__':
    import time

    query = paddle.randn((4, 2048, 3))
    ref = paddle.randn((4, 128, 3))
    k = 100
    knn = KNN(k)
    s = time.time()
    for i in range(100):
        dist, idx = knn(query, ref)
        fps(query, 128)
    e = time.time()
    print(e - s)
