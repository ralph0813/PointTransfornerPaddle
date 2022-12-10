import os
import shutil

import cv2
import numpy as np
import paddle
import skimage


def save_checkpoint(state, is_best, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, f'checkpoint_{epoch}.pdparams')
    paddle.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pdparams')
        shutil.copyfile(checkpoint_filename, best_filename)


class Logger:
    def __init__(self, log_dir=None):
        from visualdl import LogWriter
        from datetime import datetime
        if log_dir is None:
            log_dir = f'./log/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = LogWriter(logdir=log_dir)

    def add_scalar(self, tag, value, step, walltime=None):
        self.writer.add_scalar(tag, value, step, walltime)

    def write_image(self, tag, image, step, **kwargs):
        self.writer.add_image(tag, image, step, **kwargs)

    def write_metrics_log(self, epoch, metrics, mode):
        for key in metrics.keys():
            self.writer.add_scalar(f'{mode}/{key}', metrics[key], epoch)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.writer.close()


def get_vis_depth_img(img):
    stretch = skimage.exposure.rescale_intensity(img, in_range='image', out_range=(0, 255)).astype(np.uint8)
    stretch = cv2.merge([stretch, stretch, stretch])

    # define colors
    color1 = (0, 0, 255)  # red
    color2 = (0, 165, 255)  # orange
    color3 = (0, 255, 255)  # yellow
    color4 = (255, 255, 0)  # cyan
    color5 = (255, 0, 0)  # blue
    color6 = (128, 64, 64)  # violet
    colorArr = np.array([[color1, color2, color3, color4, color5, color6]], dtype=np.uint8)
    lut = cv2.resize(colorArr, (256, 1), interpolation=cv2.INTER_LINEAR)

    # apply lut
    result = cv2.LUT(stretch, lut)
    grad = np.linspace(0, 255, 512, dtype=np.uint8)
    grad = np.tile(grad, (20, 1))
    grad = cv2.merge([grad, grad, grad])

    # apply lut to gradient for viewing
    grad_colored = cv2.LUT(grad, lut)

    return result, grad_colored


def get_out_img(pred_img, gt_img):
    if isinstance(pred_img, paddle.Tensor):
        pred_img = pred_img.numpy()
    if isinstance(gt_img, paddle.Tensor):
        gt_img = gt_img.numpy()

    pred_img, grad_colored = get_vis_depth_img(pred_img)
    gt_img, _ = get_vis_depth_img(gt_img)
    out_img = cv2.hconcat([gt_img, pred_img])
    return out_img
