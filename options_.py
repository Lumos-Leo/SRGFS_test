import argparse

parser = argparse.ArgumentParser()

# Hardware specifications
parser.add_argument('--cpu', action="store_true", help='use cpu only')

# Data specifications
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='Set5',
                    help='test dataset name')
parser.add_argument('--scale', type=int, default=4,
                    help='super resolution scale')
parser.add_argument('--hr_path', type=str, default='/datassd/liao/SR/datasets/train/original/DIV2K/DIV2K_train_HR',
                    help='train high resolution images path')
parser.add_argument('--img_size', type=int, default=64,
            help='crop patch size')           

# Model specifications
parser.add_argument('--model', default='SESR',
                    help='model name')
parser.add_argument('--pre_train', type=str,
                    help='pre-trained model directory')
parser.add_argument('--fine_tune', action="store_true",
                    help='fine_tune model')
parser.add_argument('--n_resblocks', type=int, default=3,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=16,
                    help='number of feature maps')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='idel split rate')
parser.add_argument('--collapse_rate', type=int, default=16,
                    help='expand rate')
parser.add_argument('--gumbel_tau', type=int, default=500,
                    help='gumbel decrease')
parser.add_argument('--sparse', action="store_true", help='use sparse loss')

# multi-GPUs setting
parser.add_argument('--train_devices', type=list, default=[1,2,3,4],
                    help='use GPUs')
parser.add_argument('-g', '--gpus', default=1, type=int,
                    help='number of gpus per node')
parser.add_argument('--num_works', default=4, type=int,
                    help='ranking within the nodes')

# Training specifications
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--test_only', action="store_true",
                    help='run or not')
parser.add_argument('--sparsity_target', type=float, default=1.0,
                    help='sparsity rate')
parser.add_argument('--sparsity_weight', type=float, default=0.1,
                    help='sparsity weight')
parser.add_argument('--log_dir', type=str, default='./logs',
                    help='log path')

# Optimization specifications
parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=300,
                    help='learning rate decay per N epochs')
parser.add_argument('--scheduler', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='Adam',
                    choices=('SGD', 'Adam'),
                    help='optimizer to use (SGD | Adam)')
parser.add_argument('--start_epoch', type=int, default=1,
                    help='resume from the snapshot, and the start_epoch')

# Ablation Study
parser.add_argument('--test_model_path', type=str, default='prior',
                    help='')
parser.add_argument('--test_data_path', type=str, default='./data',
                    help='')
parser.add_argument('--test_type', type=str, default='benchmark',
                    help='')
parser.add_argument('--super_resolution', type=str, default='2160x3840x3',
                    help='')
parser.add_argument('--savefig', action="store_true", help='save the output images')
args = parser.parse_args()
