# By Nick Erickson
# Contains parameter functions


class Hyperparam:
    def __init__(
                 self,
                 gamma=0.99, # Fine
                 batch=16, # Fine
                 explore=300, # Fine
                 epsilon_init=1.0, # Fine
                 epsilon_final=0.01, # Fine
                 memory_size=20000, # Fine
                 save_rate=10000, # Fine, could be in metrics?
                 img_channels=1, # Specific to game
                 learning_rate=0.00025, # Fine
                 extra=None, # Fine
                 ):
        
        self.gamma = gamma # ALL
        self.batch = batch # ALL
        self.explore = explore # ALL
        self.epsilon_init = epsilon_init # ALL
        self.epsilon_final = epsilon_final # ALL
        self.memory_size = memory_size # ALL
        self.save_rate = save_rate # ALL
        self.img_channels = img_channels # ALL
        self.learning_rate = learning_rate # ALL
        self.extra = extra # Additional hyperparmeters for algorithms
        
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
                 ):
        
        self.observe = observe
        self.update_rate = update_rate

class Gym_param:
    def __init__(self,
                 problem='CartPole-v0',
                 wrapper='Gym_base_wrapper'
                 ):
        self.type = 'gym'
        self.problem = problem
        self.wrapper = wrapper

    def generate_env(self):
        import env_wrappers
        return getattr(env_wrappers,self.wrapper)(self.problem)
       
        
class Real_param:
    def __init__(self,
                 problem='Hexagon',
                 wrapper=None,
                 module_name=None,
                 class_name=None,
                 game_args=None
                 ):
        self.type = 'real'
        self.problem = problem
        self.wrapper = wrapper
        self.module_name = module_name
        self.class_name = class_name
        self.game_args = game_args
        
    def generate_env(self):
        import env_wrappers
        return getattr(env_wrappers,self.wrapper)(self.problem, self.module_name, self.class_name, self.game_args)
                
        
class Screenparam:
    def __init__(
                 self,
                 app=None,
                 size=[140,140],
                 zoom=[0,0],
                 framerate=30
                 ):
        self.app = app
        self.size = size
        self.zoom = zoom
        self.framerate = framerate
        
class Args:
    def __init__(self,
                 algorithm,
                 mode='train',
                 env='real',
                 data='default',
                 screen='default',
                 hyper='default',
                 directory='default',
                 memory_delay=4,
                 run_count_load=0, # Temp
                 model=None
                 ):
        self.algorithm = algorithm
        self.mode = mode
        self.env = env
        self.data = data
        self.screen = screen
        self.hyper = hyper
        self.directory = directory
        self.memory_delay = memory_delay
        self.run_count_load = run_count_load
        self.weight_override = None
        self.model = model