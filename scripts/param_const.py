# By Nick Erickson
# Contains parameters for games and levels

from param_utils import Args, Hyperparam, Screenparam



hex_hyperparams = Hyperparam(
                 framerate=40,
                 gamma=0.99,
                 batch=16,
                 observe=1000,
                 explore=3000,
                 epsilon_init=1.0,
                 epsilon_final=0.01,
                 memory_size=20000,
                 save_rate=10000,
                 neg_regret_frames=1,
                 img_channels=2,
                 update_rate=1000,
                 )

hex_screen = Screenparam(
                         app='Open Hexagon 1.92 - by vittorio romeo',
                         size=[140,140],
                         zoom=[28,18]
                         )

hex_incongruence = Args(
                        mode='train',
                        game='default',
                        env='real',
                        data='default',
                        screen=hex_screen,
                        hyper=hex_hyperparams,
                        directory='default',
                        memory_delay=0.5
                        )

hex_base = Args(
                        mode='train',
                        game='default',
                        env='real',
                        data='default',
                        screen=hex_screen,
                        hyper=hex_hyperparams,
                        directory='default',
                        memory_delay=4
                        )

gym_cart_hyperparams = Hyperparam(
                             framerate=40,
                             gamma=0.99,
                             batch=64,
                             observe=100000,
                             explore=30000,
                             epsilon_init=1.0,
                             epsilon_final=0.01,
                             memory_size=100000,
                             save_rate=100000,
                             neg_regret_frames=0,
                             img_channels=1,
                             update_rate=1000
                           )

gym_cart = Args(
                        mode='train',
                        game='CartPole-v0',
                        env='gym',
                        data='default',
                        screen='default',
                        hyper=gym_cart_hyperparams,
                        directory='default',
                        memory_delay=4
                        )

hex_gather_hyperparams = Hyperparam(
                             framerate=40,
                             gamma=0.99,
                             batch=64,
                             observe=0,
                             explore=10000,
                             epsilon_init=1.0,
                             epsilon_final=0.01,
                             memory_size=50000,
                             save_rate=10000,
                             neg_regret_frames=1,
                             img_channels=2,
                             update_rate=1000
                           )