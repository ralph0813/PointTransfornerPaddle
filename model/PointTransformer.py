import time

import paddle
import paddle.nn as nn
from tqdm import tqdm

from .Transformer import TransformerEncoder
from .pointnet import KNN, fps, PointNetFeaturePropagation


class Group:
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=group_size)

    def __call__(self, x):
        """

        :param x: shape (B, N, C)
        :return: shape (B, N, num_group, group_size)
        """
        batch_size, num_points, _ = x.shape
        center = fps(x, self.num_group)
        # knn to get the neighborhood
        _, idx = self.knn(x, center)
        assert idx.shape[1] == self.num_group
        assert idx.shape[2] == self.group_size
        idx_base = paddle.arange(0, batch_size, dtype=paddle.int64).reshape((-1, 1, 1)) * num_points
        idx = idx + idx_base
        idx = idx.reshape([-1, ]).numpy()
        neighborhood = paddle.to_tensor(x.reshape((batch_size * num_points, -1)).numpy()[idx, :])
        neighborhood = neighborhood.reshape((batch_size, self.num_group, self.group_size, 3))
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class Encoder(nn.Layer):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.layer1 = nn.Sequential(
            nn.Conv1D(3, 128, 1),
            nn.BatchNorm1D(128),
            nn.ReLU(),
            nn.Conv1D(128, 256, 1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1D(512, 512, 1),
            nn.BatchNorm1D(512),
            nn.ReLU(),
            nn.Conv1D(512, self.encoder_channel, 1),
        )

    def forward(self, x):
        """
        :param x: shape B G N 3
        :return: shape B G C
        """
        batch_size, num_group, num_points, _ = x.shape
        x = x.reshape((batch_size * num_group, num_points, 3))
        feature = self.layer1(x.transpose((0, 2, 1)))
        feature_global = paddle.max(feature, axis=2)
        expend = paddle.tile(feature_global, (1, num_points)).reshape((batch_size * num_group, -1, num_points))
        feature_with_global = paddle.concat([expend, feature], axis=1)
        feature = self.layer2(feature_with_global)
        feature_global = paddle.max(feature, axis=2)
        return feature_global.reshape((batch_size, num_group, self.encoder_channel))


class PointTransformer(nn.Layer):
    def __init__(self, cls_dim):
        super().__init__()
        self.trans_dim = 384
        self.depth = 12
        self.drop_path_rate = 0.1
        self.cls_dim = cls_dim
        self.num_heads = 6

        self.group_size = 64
        self.num_group = 128
        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # define the encoder
        self.encoder_dims = 384
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.norm = nn.LayerNorm(self.trans_dim)
        # bridge encoder and transformer
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.transformer = TransformerEncoder(
            dim=self.trans_dim,
            heads=self.num_heads,
            dropout=0.1
        )
        self.propagation_0 = PointNetFeaturePropagation(
            in_channel=1152 + 3, mlp=[self.trans_dim * 4, 1024]
        )
        self.seg_head = nn.Sequential(
            nn.Conv1D(3328, 512, 1),
            nn.BatchNorm1D(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1D(512, 256, 1),
            nn.BatchNorm1D(256),
            nn.ReLU(),
            nn.Conv1D(256, self.cls_dim, 1)
        )

    def forward(self, pts):
        B, N, C = pts.shape
        # pts = pts.transpose(-1, -2).contiguous()  # B N 3
        # divide the point cloud in the same form. This is important
        # 先根据FPS把点分组，再用KNN找到每组点的邻域
        neighborhood, center = self.group_divider(pts)
        # encode the point cloud
        group_input_tokens = self.encoder(neighborhood)  # B G N
        # add the position embedding
        group_input_tokens = group_input_tokens + self.pos_embed(center)
        # transformer
        feature_list = self.transformer(group_input_tokens)
        feature_list = [self.norm(x).transpose((0, 2, 1)) for x in feature_list]
        x = paddle.concat((feature_list[0], feature_list[1], feature_list[2]), axis=1)
        # get the global feature
        x_max = paddle.max(x, axis=2)
        x_avg = paddle.mean(x, axis=2)
        x_max_feature = x_max.reshape((B, -1)).tile((1, N)).reshape((B, -1, N))
        x_avg_feature = x_avg.reshape((B, -1)).tile((1, N)).reshape((B, -1, N))
        x_global_feature = paddle.concat((x_max_feature, x_avg_feature), axis=1)

        f_level_0 = self.propagation_0(pts.transpose((0, 2, 1)), center.transpose((0, 2, 1)), pts.transpose((0, 2, 1)),
                                       x)
        x = paddle.concat((f_level_0, x_global_feature), 1)
        x = self.seg_head(x)
        x = nn.functional.log_softmax(x, axis=1)
        x = x.transpose((0, 2, 1))
        return x


if __name__ == '__main__':
    x = paddle.rand((8, 2048, 3))
    model = PointTransformer(33)
    start = time.time()
    optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    for i in tqdm(range(1000)):
        y = model(x)
        loss = paddle.mean(y)
        loss.backward()
        optimizer.step()

    end = time.time()
    print(end - start)
    # y = model(x)
    # print(y.shape)
