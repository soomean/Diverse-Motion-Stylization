import numpy as np
from torch.utils.tensorboard import SummaryWriter

from .helper import make_dir, denormalize
from .helper import restore_animation, to_bvh_cmu, remove_fs


parents = [-1,
            0, 1, 2, 3,
            0, 5, 6, 7,
            0, 9, 10, 11,
            10, 13, 14, 15,
            10, 17, 18, 19]


class Logger:
    def __init__(self, opt):
        self.opt = opt
        self.mode = opt.mode
        if opt.mode == 'train':
            tsblog_name = make_dir(opt.save_dir, 'tensorboard_log')
            self.writer = SummaryWriter(tsblog_name)
        self.weights_name = ['adv', 'reg', 'con', 'sty', 'ds', 'cyc']

        preprocess = np.load(opt.preproot)
        self.Xmean = preprocess['Xmean']
        self.Xstd = preprocess['Xstd']

    def print_current_losses(self, iter, losses):
        message = ''
        for name, loss in losses.items():
            for k, v in loss.items():
                rename = "%s_%s_%s" % (name.split('_')[0].capitalize(), name.split('_')[2], k)
                message += "%s: %.3f\t" % (rename, v)
                self.writer.add_scalar('Loss/%s' % rename, v, iter)
            message += '\n'
        return message

    def print_current_lrs(self, lrs):
        message = ''
        for name, lr in lrs.items():
            message += '%s: %f\t' % (name, lr)
        message += '\n'
        return message

    def print_current_weights(self):
        message = ''
        for name in self.weights_name:
            message += 'lambda_%s: %.3f\t' % (name, getattr(self.opt, 'lambda_%s' % name))
        message += '\n'
        return message

    def save_output(self, output, traj, feet=None, filename='output.bvh', fs_fix=False):
        output = output.detach().cpu().numpy().copy()
        output = output[0]
        output = denormalize(output, self.Xmean[:7], self.Xstd[:7])
        output = np.transpose(output, (1, 2, 0))

        traj = traj.detach().cpu().numpy().copy()
        traj = traj[0]
        traj = denormalize(traj, self.Xmean[-4:], self.Xstd[-4:])
        traj = np.transpose(traj, (1, 2, 0))

        # original output
        positions = restore_animation(output[:, :, :3], traj)

        # foot skate clean-up
        if fs_fix:
            filename = filename[:-4] + '_fs.bvh'
            positions = remove_fs(positions, feet)

        print('Saving animation of %s in bvh...' % filename)
        to_bvh_cmu(positions, filename=filename, frametime=1.0/30.0)
