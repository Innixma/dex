# By Nick Erickson
# Contains parameters for games and levels

from param_utils import Args, Hyperparam, Screenparam, Hyper_a3c



hex_hyperparams = Hyperparam(
                 framerate=40,
                 gamma=0.99,
                 batch=16,
                 observe=1000,
                 explore=3000,
                 epsilon_init=1.0,
                 epsilon_final=0.01,
                 memory_size=50000,
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
                        algorithm='ddqn',
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
                        algorithm='ddqn',
                        mode='train',
                        game='default',
                        env='real',
                        data='default',
                        screen=hex_screen,
                        hyper=hex_hyperparams,
                        directory='default',
                        memory_delay=4
                        )

gym_cart_ddqn_hyperparams = Hyperparam(
                             framerate=40,
                             gamma=0.99,
                             batch=8,
                             observe=7500,
                             explore=30000,
                             epsilon_init=0.4,
                             epsilon_final=0.15,
                             memory_size=7500,
                             save_rate=10000,
                             neg_regret_frames=0,
                             img_channels=1,
                             update_rate=1000,
                             learning_rate=5e-3
                           )

gym_cart_ddqn = Args(
                        algorithm='ddqn',
                        mode='train',
                        game='CartPole-v0',
                        env='gym',
                        data='default',
                        screen='default',
                        hyper=gym_cart_ddqn_hyperparams,
                        directory='default',
                        memory_delay=4
                        )

gym_cart_a3c_hyperspecific = Hyper_a3c(
                                       loss_v=0.5,
                                       loss_entropy=0.01,
                                       )

gym_cart_a3c_hyperparams = Hyperparam(
                             framerate=40,
                             gamma=0.99,
                             batch=8,
                             observe=0,
                             explore=75000,
                             epsilon_init=0.4,
                             epsilon_final=0.15,
                             memory_size=8,
                             save_rate=100000,
                             neg_regret_frames=0,
                             img_channels=1,
                             update_rate=1000,
                             learning_rate=5e-3,
                             extra=gym_cart_a3c_hyperspecific
                           )

hex_base_a3c_hyperspecific = Hyper_a3c(
                                       loss_v=0.5,
                                       loss_entropy=0.01,
                                       )

hex_base_a3c_hyperparams = Hyperparam(
                             framerate=40,
                             gamma=0.99,
                             batch=64,
                             observe=0,
                             explore=100000,
                             epsilon_init=1,
                             epsilon_final=0.05,
                             memory_size=8,
                             save_rate=5000,
                             neg_regret_frames=0,
                             img_channels=2,
                             update_rate=1000,
                             learning_rate=2.5e-3,
                             extra=hex_base_a3c_hyperspecific
                           )

hex_base_a3c = Args(
                        algorithm='a3c',
                        mode='train',
                        game='default',
                        env='real',
                        data='default',
                        screen=hex_screen,
                        hyper=hex_base_a3c_hyperparams,
                        directory='default',
                        memory_delay=4
                        )


gym_cart_a3c = Args(
                        algorithm='a3c',
                        mode='train',
                        game='CartPole-v0',
                        env='gym',
                        data='default',
                        screen='default',
                        hyper=gym_cart_a3c_hyperparams,
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