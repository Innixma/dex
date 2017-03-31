# By Nick Erickson
# Contains parameter functions


class Hyperparam:
    def __init__(
                 self,
                 framerate=40,
                 gamma=0.99,
                 batch=16,
                 observe=100,
                 explore=300,
                 epsilon_init=1.0,
                 epsilon_final=0.01,
                 memory_size=20000,
                 save_rate=10000,
                 neg_regret_frames=1,
                 img_channels=2,
                 update_rate=1000,
                 ):
        
        self.framerate = framerate
        self.gamma = gamma
        self.batch = batch
        self.observe = observe
        self.explore = explore
        self.epsilon_init = epsilon_init
        self.epsilon_final = epsilon_final
        self.memory_size = memory_size
        self.save_rate = save_rate
        self.neg_regret_frames = neg_regret_frames
        self.img_channels = img_channels
        self.update_rate = update_rate
        
class Screenparam:
    def __init__(self,
                 app=None,
                 size=[140,140],
                 zoom=[0,0]
                 ):
        self.app = app
        self.size = size
        self.zoom = zoom
        
class Args:
    def __init__(self,
                 mode='train',
                 game='default',
                 env='real',
                 data='default',
                 screen='default',
                 hyper='default',
                 directory='default',
                 memory_delay=4
                 ):
        self.mode = mode
        self.game = game
        self.env = env
        self.data = data
        self.screen = screen
        self.hyper = hyper
        self.directory = directory
        self.memory_delay = memory_delay