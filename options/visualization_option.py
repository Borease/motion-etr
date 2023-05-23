from .base_options import BaseOptions


class VisualOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./exp_results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=5000, help='how many test images to run')
        self.parser.add_argument('--verbose', action='store_true', help='print map')
        self.parser.add_argument('--offset_grid', type=int, default=17, help='Grid length of visualizing recovered motion offset')
        self.parser.add_argument('--visualize_traj', type=bool, default=False, help='Print recovered trajectory')
        self.parser.add_argument('--visualize_flow', type=bool, default=False, help='Print flow, set true when reblur, false when deblur')
        self.parser.add_argument('--extract_frame', type=bool, default=False, help='Extract frames, currently set this always false')
        self.parser.add_argument('--visualize_testing_flow_offset',type=bool, default=True, help='[Debug only] Save offset and flow testing image')
        self.parser.add_argument('--set_how_many', type=bool, default=False, help='use how_many to limit input images or not')
        self.isTrain = False