import argparse

# Training settings
parser = argparse.ArgumentParser(description='PyTorch ADMM training for C3D')
parser.add_argument('--logger', action='store_true', default=True,
                    help='whether to use logger')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--seed', type=int, default=2019, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 12)')
parser.add_argument('--multi-gpu', action='store_true', default=False,
                    help='for multi-gpu training')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--arch', type=str, default='r2+1d',
                    help='[c3d, r2+1d, s3d, mf3d]')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--admm-epochs', type=int, default=2, metavar='N',
                    help='number of interval epochs to update admm (default: 10)')

# parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
#                     help='learning rate (default: 0.1)')
parser.add_argument('--lr-decay', type=int, default=30, metavar='LR_decay',
                    help='how many every epoch before lr drop (default: 30)')
# parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
#                     help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')

parser.add_argument('--sparsity-type', type=str, default='random-pattern',
                    help ='define sparsity_type: [filter, channel, column]')
parser.add_argument('--config-file', type=str, default='c3d',
                    help ='config file name')
parser.add_argument('--admm', action='store_true', default=True,
                    help='for admm training')
parser.add_argument('--masked-retrain', action='store_true', default=False,
                    help='for masked retrain')
parser.add_argument('--combine-progressive', action='store_true', default=False,
                    help='for filter pruning after column pruning')
parser.add_argument('--rho', type=float, default=0.0001,
                    help='define rho for ADMM')
# Tricks
parser.add_argument('--lr-scheduler', type=str, default='default',
                    help='define lr scheduler')
parser.add_argument('--alpha', type=float, default=0.0, metavar='M',
                    help='for mixup training, lambda = Beta(alpha, alpha) distribution. Set to 0.0 to disable')
parser.add_argument('--no-tricks', action='store_true', default=False,
                    help='disable all training tricks and restore original classic training process')

#

parser.add_argument('--resume',  action='store_true', default=False,
                    help='resume from last epoch if model exists')
parser.add_argument("--data_label_path", help="data label path", required=True, type=str)
parser.add_argument("--init_path", help="init path", required=True, type=str)
parser.add_argument("--label_path", help="label path", required=True, type=str)
parser.add_argument("--lr", help="learning rate", required=True, type=float)
parser.add_argument("--momentum", help="momentum", required=True, type=float)
parser.add_argument("--weight_decay", help="weight decay", required=True, type=float)
parser.add_argument("--batch_size", help="batch size", required=True, type=int)
parser.add_argument("--num_epochs", help="num_epochs", required=True, type=int)
parser.add_argument("--split_ratio", help="training/testing split ratio", required=True, type=float)
parser.add_argument("--checkpoint_dir", help="checkpoint_dir", required=True, type=str)
parser.add_argument("--load_from_main_checkpoint", type=str)
parser.add_argument("--model_checkpoint_name", help="model checkpoint name", required=True, type=str)
parser.add_argument("--image_save_folder", type=str, required=True)
parser.add_argument("--eval_only", dest='eval_only', action='store_true')