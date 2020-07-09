import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
    

class AverageMeter(object):
    '''Computes and stores the average and current value'''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    '''Computes the accuracy over the k top predictions for the specified values of k'''
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(optimizer, epoch, args):
    '''Sets the learning rate to the initial LR decayed by 10 every 30 epochs'''
    # only in masked retrain

    # adjust learning rate
    if args.warmup and epoch - 1 <= args.warmup_epochs:
        lr = args.warmup_lr + (args.lr - args.warmup_lr) / args.warmup_epochs * (epoch - 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    args.lr_decay = max(1, int(args.epochs * 0.2))
    #lr = args.lr * (0.3 ** (epoch // args.lr_decay))
    lr = args.lr * (0.5 ** ((epoch - 1) // args.lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class CrossEntropyLossMaybeSmooth(nn.CrossEntropyLoss):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    def __init__(self, smooth_eps=0.0):
        super(CrossEntropyLossMaybeSmooth, self).__init__()
        self.smooth_eps = smooth_eps

    def forward(self, output, target, smooth=False):
        if not smooth:
            return F.cross_entropy(output, target)

        target = target.contiguous().view(-1)  # 此处target变为一维
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        smooth_one_hot = one_hot * (1 - self.smooth_eps) + (1 - one_hot) * self.smooth_eps / (n_class - 1)
        log_prb = F.log_softmax(output, dim=1)
        loss = -(smooth_one_hot * log_prb).sum(dim=1).mean()
        return loss


def mixup_data(x, y, alpha=1.0):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam, smooth):
    return lam * criterion(pred, y_a, smooth=smooth) + \
           (1 - lam) * criterion(pred, y_b, smooth=smooth)


class GradualWarmupScheduler(_LRScheduler):
    ''' Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_iter: target learning rate is reached at total_iter, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    '''

    def __init__(self, optimizer, multiplier, total_iter, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_iter = total_iter
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_iter:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_iter + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            return self.after_scheduler.step(epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


# https://gist.github.com/spezold/42a451682422beb42bc43ad0c0967a30
def torch_percentile(t: torch.tensor, q: float):
    """
    Return the ``q``-th percentile of the flattened input tensor's data.
    
    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.
       
    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.cpu().view(-1).kthvalue(k).values.item()
    return result


def reshape_matrix2block(matrix, blk_h, blk_w):
    block = torch.cat(torch.split(matrix, blk_h), dim=1)
    block = torch.split(block, blk_w, dim=1)
    block = torch.stack([i.reshape(-1) for i in block])
    return block
def reshape_block2matrix(block, num_blk_h, num_blk_w, blk_h, blk_w):
    matrix = []
    for i in range(num_blk_h):
        for j in range(blk_h):
            matrix.append(block[num_blk_w*i:num_blk_w*(i+1), blk_w*j:blk_w*(j+1)].reshape(-1))
    matrix = torch.stack(matrix)
    return matrix


def reshape_matrix2block_kernel(matrix, blk_h, blk_w):
    block = torch.cat(torch.split(matrix, blk_h), dim=1)
    block = torch.split(block, blk_w, dim=1)
    block = torch.cat([i.permute(2, 0, 1).reshape(-1, blk_h*blk_w) for i in block])
    return block
def reshape_block2matrix_kernel(block, num_blk_h, num_blk_w, blk_h, blk_w, kernel_size):
    matrix = []
    blocks = torch.stack(torch.split(block, kernel_size), dim=1).permute(1, 2, 0)
    for i in range(num_blk_h):
        for j in range(blk_h):
            matrix.append(blocks[num_blk_w*i:num_blk_w*(i+1), blk_w*j:blk_w*(j+1)].reshape(-1))
    matrix = torch.stack(matrix)
    return matrix


if __name__ == '__main__':
    def test_block():
        shape = torch.Size([4, 8, 2, 1])
        block_shape = (3, 3)
        weight = torch.arange(torch.prod(torch.tensor(shape))).reshape(tuple(shape)).cuda()
        print(weight[:, :, 0, 0])

        ext_shape = [(shape[i] + block_shape[i] - 1) // block_shape[i] * block_shape[i] for i in range(2)] + list(shape[2:])

        blk_h, blk_w = block_shape
        num_blk_h, num_blk_w = ext_shape[0]//blk_h, ext_shape[1]//blk_w
        kernel_size = torch.prod(torch.tensor(shape[2:]))
        valid = torch.Tensor(num_blk_h * num_blk_w)
        for i in range(num_blk_h):
            for j in range(num_blk_w):
                valid_x = min(blk_h*(i+1), shape[0]) - blk_h*i
                valid_y = min(blk_w*(j+1), shape[1]) - blk_w*j
                valid[i*num_blk_w+j] = valid_x * valid_y * kernel_size
        print(valid)
        valid1 = torch.Tensor(num_blk_h * num_blk_w)
        for i in range(num_blk_h * num_blk_w):
            valid1[i] = len((weight_[i]==0).nonzero())
        print(valid1)

        padding = nn.ZeroPad2d((0, ext_shape[1] - shape[1], 0, ext_shape[0] - shape[0]))
        assert len(shape) >= 2
        if len(shape) == 2:
            weight_ = padding(weight)
        else:
            weight_ = weight.reshape(shape[0], shape[1], -1)
            weight_ = torch.stack([padding(weight_[:, :, i]) for i in range(kernel_size)], dim=2)
        print(weight_[:, :, 0])

        weight_ = reshape_matrix2block(weight_, blk_h, blk_w)
        print(weight_)

        '''norm & pruning'''
        weight_[0] = torch.tensor([-1 for _ in weight_[0]])
        print(weight_)

        weight_ = reshape_block2matrix(weight_, num_blk_h, num_blk_w, blk_h, blk_w * kernel_size)
        weight = weight_.reshape(ext_shape)[:shape[0], :shape[1]]
        print(weight[:, :, 0, 0])
    # test_block()

    def test_block_kernel():
        shape = torch.Size([4, 8, 2, 1])
        block_shape = (3, 3)
        weight = torch.arange(torch.prod(torch.tensor(shape))).reshape(tuple(shape)).cuda()
        print(weight[:, :, 0, 0])

        ext_shape = [(shape[i] + block_shape[i] - 1) // block_shape[i] * block_shape[i] for i in range(2)] + list(shape[2:])

        blk_h, blk_w = block_shape
        num_blk_h, num_blk_w = ext_shape[0]//blk_h, ext_shape[1]//blk_w
        kernel_size = torch.prod(torch.tensor(shape[2:]))
        valid = torch.Tensor(num_blk_h * num_blk_w * int(kernel_size))
        for i in range(num_blk_h):
            for j in range(num_blk_w):
                for k in range(kernel_size):
                    valid_x = min(blk_h*(i+1), shape[0]) - blk_h*i
                    valid_y = min(blk_w*(j+1), shape[1]) - blk_w*j
                    valid[(i*num_blk_w+j)*kernel_size+k] = valid_x * valid_y
        print(valid)
        valid1 = torch.Tensor(num_blk_h * num_blk_w * int(kernel_size))
        for i in range(num_blk_h * num_blk_w * kernel_size):
            valid1[i] = len((weight_[i]==0).nonzero())
        print(valid1)

        padding = nn.ZeroPad2d((0, ext_shape[1] - shape[1], 0, ext_shape[0] - shape[0]))
        assert len(shape) >= 2
        if len(shape) == 2:
            weight_ = padding(weight)
        else:
            weight_ = weight.reshape(shape[0], shape[1], -1)
            weight_ = torch.stack([padding(weight_[:, :, i]) for i in range(kernel_size)], dim=2)
        # print(weight_[:, :, 0])

        weight_ = reshape_matrix2block_kernel(weight_, blk_h, blk_w)
        print(weight_)

        '''norm & pruning'''
        weight_[0] = torch.tensor([-1 for _ in weight_[0]])
        weight_[3] = torch.tensor([-2 for _ in weight_[3]])
        # print(weight_)

        weight_ = reshape_block2matrix_kernel(weight_, num_blk_h, num_blk_w, blk_h, blk_w, int(kernel_size))
        weight = weight_.reshape(ext_shape)[:shape[0], :shape[1]]
        print(weight[:, :, 0, 0])
        print(weight[:, :, 1, 0])
    test_block_kernel()