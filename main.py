import argparse
import os

import paddle
from paddle import optimizer, nn
from paddle.io import DataLoader
from tqdm import tqdm

import utils
from dataloader import ToothSegData
from model import PointTransformer as Model
from utils.metrics import ConfusionMatrix, get_mious


def parse_args():
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--root', type=str, default='./data/ToothDataset/seg_data_10k', help='data root')
    parser.add_argument('--device', type=str, default='gpu', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size in training')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--epoch', default=40, type=int, help='number of epoch in training')
    parser.add_argument('--interval', default=1, type=float, help='interval of save model')
    parser.add_argument('--n_sample', default=500, type=float, help='learning rate in training')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate in training')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum in training')
    parser.add_argument('--nesterov', '-n', action='store_true', help='enables Nesterov momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay in training')
    parser.add_argument('--save_path', type=str, default='weights/', help='path to save the checkpoints')
    parser.add_argument('--log_dir', type=str, default=None, help='path to save the log')
    parser.add_argument('--pretrain', type=str, default='weights/model_best.pdparams',
                        help='path to load the pretrain model')
    return parser.parse_args()


def train_epoch(model, data_loader, loss_fn, optim, epoch):
    model.train()
    loss_sum = 0
    cm = ConfusionMatrix(33)

    for i, data in tqdm(enumerate(data_loader), desc='Train Epoch: {}'.format(epoch), total=len(data_loader)):
        optim.clear_grad()
        points, target = data
        target = target.astype('int64')
        outputs = model(points)
        outputs_flatten = outputs.reshape((-1, outputs.shape[-1]))  # [B, N, C] -> [B*N, C]
        target_flatten = target.reshape((-1, 1))[:, 0]  # [B, N] -> [B*N]
        loss = loss_fn(outputs_flatten, target_flatten)
        loss.backward()

        cm.update(outputs_flatten.argmax(axis=1), target_flatten)
        loss_sum += loss.numpy()
        optim.step()


    miou, macc, oa, ious, accs = get_mious(cm.tp, cm.union, cm.count)
    metrics = {'loss': loss_sum / len(data_loader), 'miou': miou, 'macc': macc, 'oa': oa}
    logger.write_metrics_log(epoch, metrics, "train")
    return metrics


@paddle.no_grad()
def val_epoch(model, data_loader, loss_fn, epoch):
    loss_sum = 0
    model.eval()
    cm = ConfusionMatrix(33)

    for i, data in tqdm(enumerate(data_loader), desc='Val Epoch: {}'.format(epoch), total=len(data_loader)):
        points, target = data
        target = target.astype('int64')
        outputs = model(points)
        outputs_flatten = outputs.reshape((-1, outputs.shape[-1]))  # [B, N, C] -> [B*N, C]
        target_flatten = target.reshape((-1, 1))[:, 0]  # [B, N] -> [B*N]
        loss = loss_fn(outputs_flatten, target_flatten)

        cm.update(outputs_flatten.argmax(axis=1), target_flatten)
        loss_sum += loss.numpy()


    miou, macc, oa, ious, accs = get_mious(cm.tp, cm.union, cm.count)
    metrics = {'loss': loss_sum / len(data_loader), 'miou': miou, 'macc': macc, 'oa': oa}
    logger.write_metrics_log(epoch, metrics, "train")
    return metrics


def train(args):
    paddle.device.set_device(args.device)
    # load data
    train_set = ToothSegData(args.root, n_points=2048, split='train')
    val_set = ToothSegData(args.root, n_points=2048, split='test')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # define model
    model = Model(33)
    model_named_params = [p for _, p in model.named_parameters() if not p.stop_gradient]
    # define loss
    lose_fn = nn.NLLLoss()
    # define lr_scheduler and optimizer
    lr_scheduler = optimizer.lr.CosineAnnealingDecay(
        learning_rate=args.lr,
    )
    optim = optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model_named_params,
        weight_decay=args.weight_decay,
    )
    # load pretrain model
    start_epoch = 0
    # best_error = {
    #     'MSE': float('inf'), 'RMSE': float('inf'), 'ABS_REL': float('inf'), 'LG10': float('inf'),
    #     'MAE': float('inf'),
    #     'DELTA1.02': 0, 'DELTA1.05': 0, 'DELTA1.10': 0,
    #     'DELTA1.25': 0, 'DELTA1.25^2': 0, 'DELTA1.25^3': 0
    # }
    best_error = {'loss': float('inf'), 'miou': 0, 'macc': 0, 'oa': 0}
    if args.pretrain and os.path.exists(args.pretrain):
        try:
            checkpoints = paddle.load(args.pretrain)
            model.set_state_dict(checkpoints['model'])
            optim.set_state_dict(checkpoints['optimizer'])
            lr_scheduler.set_state_dict(checkpoints['lr_scheduler'])
            start_epoch = checkpoints['epoch']
            best_error = checkpoints['val_metrics']
            print(f'load pretrain model from {args.pretrain}')
        except Exception as e:
            print(f"{e} load pretrain model failed")

    # train
    for epoch in range(start_epoch, args.epoch):
        train_loss = train_epoch(model, train_loader, lose_fn, optim, epoch)
        val_loss = val_epoch(model, val_loader, lose_fn, epoch)
        print(f"Epoch: {epoch}\ntrain_loss:\t{train_loss}\nval_loss:\t{val_loss}")

        lr_scheduler.step(train_loss['loss'])
        #
        # logger.write_log(epoch, train_loss, "train_epoch")
        # logger.write_log(epoch, val_metrics, "val_epoch")

        is_best = False
        if val_loss['loss'] < best_error['loss']:
            best_error = val_loss
            is_best = True
        if epoch % args.interval == 0 or is_best:
            utils.save_checkpoint({
                'args': args,
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optim.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'train_metrics': train_loss,
                'val_metrics': val_loss,
            }, is_best, epoch, args.save_path)
            print(f"save model at epoch {epoch}")


if __name__ == '__main__':
    args = parse_args()
    with utils.Logger(args.log_dir) as logger:
        train(args)
