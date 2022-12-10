import os
import numpy as np
import paddle
from paddle.io import Dataset


class ToothSegData(Dataset):
    def __init__(self, root, n_points=8192, split='train', label_type='txt', feature_size=3):
        super().__init__()
        self.root = root
        self.n_points = n_points
        self.split = split
        self.samples = []
        self.label_type = label_type
        self.feature_size = feature_size

        self.__load_data__()

    def __load_data__(self):
        with open(os.path.join(self.root, f"{self.split}.txt"), "r") as f:
            paths = f.readlines()
        for path in paths:
            path = path.strip()
            if path.endswith(f".{self.label_type}"):
                self.samples.append(path)
        with open(os.path.join(self.root, "labels.txt"), "r") as f:
            self.names = f.readlines()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        file_path = os.path.join(self.root, "data", self.samples[item])
        pc_data = np.loadtxt(file_path) if self.label_type == 'txt' else np.load(file_path)
        pts = pc_data[:, :self.feature_size].astype(np.float32)
        seg = pc_data[:, self.feature_size].astype(np.int32)

        if self.n_points != pts.shape[0]:
            replace = False if self.n_points < pts.shape[0] else True
            choice = np.random.choice(len(pts), self.n_points, replace=replace)
            pts = pts[choice, :]
            seg = seg[choice]

        return pts, seg

    def label_to_name(self, idx):
        return self.names[idx]


if __name__ == '__main__':
    data = ToothSegData("../data/ToothDataset/seg_face_normal", split='test')
    for idx, (pts, seg) in enumerate(data):
        print(idx, pts.shape, seg.shape)
