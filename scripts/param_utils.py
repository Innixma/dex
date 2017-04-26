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
                 learning_rate=0.00025,
                 extra=None,
                 model=None
                 ):
        
        self.framerate = framerate # REAL
        self.gamma = gamma # ALL
        self.batch = batch # ALL
        self.observe = observe # DDQN
        self.explore = explore # ALL
        self.epsilon_init = epsilon_init # ALL
        self.epsilon_final = epsilon_final # ALL
        self.memory_size = memory_size # ALL
        self.save_rate = save_rate # ALL
        self.neg_regret_frames = neg_regret_frames # DDQN
        self.img_channels = img_channels # ALL
        self.update_rate = update_rate # DDQN
        self.learning_rate = learning_rate # ALL
        self.extra = extra # Additional hyperparmeters for algorithms
        self.model = model
        
class Hyper_a3c: # Additional hyperparameters for a3c
    def __init__(
                 self,
                 loss_v=0.5,
                 loss_entropy=0.01,
                 brain_memory_size=8
                 ):
        
        self.loss_v = loss_v
        self.loss_entropy = loss_entropy
        self.brain_memory_size = brain_memory_size

# TODO: Use this
class Hyper_ddqn: # Additional hyperparameters for ddqn
    def __init__(
                 self,
                 observe=100,
                 update_rate=1000,
                 neg_regret_frames=1
                 ):
        
        self.observe = observe
        self.update_rate = update_rate
        self.neg_regret_frames = neg_regret_frames

class Gym_param:
    def __init__(self,
                 problem='CartPole-v0',
                 wrapper='Gym_base_wrapper'
                 ):
        self.type = 'gym'
        self.problem = problem
        self.wrapper = wrapper

class Real_param:
    def __init__(self,
                 problem='Hexagon',
                 wrapper=None
                 ):
        self.type = 'real'
        self.problem = problem
        self.wrapper = wrapper
                
        
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
                 algorithm,
                 mode='train',
                 game='default',
                 env='real',
                 data='default',
                 screen='default',
                 hyper='default',
                 directory='default',
                 memory_delay=4,
                 run_count_load=0 # Temp
                 ):
        self.algorithm = algorithm
        self.mode = mode
        self.game = game
        self.env = env
        self.data = data
        self.screen = screen
        self.hyper = hyper
        self.directory = directory
        self.memory_delay = memory_delay
        self.run_count_load = run_count_load
        self.weight_override = None