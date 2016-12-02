import argparse
import mxnet as mx
from core.imdb import IMDB
from train import train_net
from core.symbol import R_Net

def train_R_net(image_set, root_path, dataset_path, prefix, ctx,
                 pretrained, epoch, begin_epoch,
                 end_epoch, frequent, lr, resume):
    imdb = IMDB("mtcnn", image_set, root_path, dataset_path)
    gt_imdb = imdb.gt_imdb()
    gt_imdb = imdb.append_flipped_images(gt_imdb)
    sym = R_Net()

    train_net(sym, prefix, ctx, pretrained, epoch, begin_epoch, end_epoch, gt_imdb,
              24, frequent, not resume, lr)

def parse_args():
    parser = argparse.ArgumentParser(description='Train refine net(24-net)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image_set', dest='image_set', help='training set',
                        default='train_24', type=str)
    parser.add_argument('--root_path', dest='root_path', help='output data folder',
                        default='data', type=str)
    parser.add_argument('--dataset_path', dest='dataset_path', help='dataset folder',
                        default='data/mtcnn', type=str)
    parser.add_argument('--prefix', dest='prefix', help='new model prefix',
                        default='model/rnet', type=str)
    parser.add_argument('--gpus', dest='gpu_ids', help='GPU device to train with',
                        default='0', type=str)
    parser.add_argument('--pretrained', dest='pretrained', help='pretrained prefix',
                        default='model/rnet', type=str)
    parser.add_argument('--epoch', dest='epoch', help='load epoch',
                        default=0, type=int)
    parser.add_argument('--begin_epoch', dest='begin_epoch', help='begin epoch of training',
                        default=0, type=int)
    parser.add_argument('--end_epoch', dest='end_epoch', help='end epoch of training',
                        default=16, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=200, type=int)
    parser.add_argument('--lr', dest='lr', help='learning rate',
                        default=0.01, type=float)
    parser.add_argument('--resume', dest='resume', help='continue training', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print 'Called with argument:'
    print args
    ctx = [mx.gpu(int(i)) for i in args.gpu_ids.split(',')]
    train_R_net(args.image_set, args.root_path, args.dataset_path, args.prefix, ctx,
                 args.pretrained, args.epoch,
                 args.begin_epoch, args.end_epoch, args.frequent, args.lr, args.resume)
