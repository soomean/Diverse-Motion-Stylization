import os
from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def __init__(self):
        BaseOptions.__init__(self)

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # dataset parameter
        parser.add_argument('--num_workers', type=int, default=0)
        parser.add_argument('--test_dataroot', type=str, default='./datasets/styletransfer_classify.npz', help='path to training set')
        parser.add_argument('--preproot', type=str, default='./datasets/preprocess_styletransfer_generate.npz', help='path to preprocess')
        parser.add_argument('--clip_size', type=int, nargs='+', default=[64, 21])

        # test parameters
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--load_iter', type=int, default=-1)
        parser.add_argument('--load_latest', action='store_true')

        return parser

    def check(self, opt):
        assert opt.mode == 'test', 'Not a test mode!'
        assert opt.num_domains == len(opt.domains), 'Number of domains does not match!'
        assert opt.load_latest == (opt.load_iter == -1), 'Specify either load_iter or load_latest!'

    def print_options(self, opt):
        message = BaseOptions.print_options(self, opt)
        file_name = os.path.join(opt.sub_dir, '%s_opt.txt' % opt.mode)

        with open(file_name, 'w') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
