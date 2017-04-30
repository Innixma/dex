# By Nick Erickson
# Contains parameters for games and levels

from param_utils import Args, Hyperparam, Screenparam, Hyper_a3c, Hyper_ddqn, Gym_param, Real_param

hex_screen = Screenparam(
                         app='Open Hexagon 1.92 - by vittorio romeo',
                         size=[140,140],
                         zoom=[12,2],
                         framerate=50
                         )

ddqn_hyperspecific = Hyper_ddqn(
                                observe=7500,
                                update_rate=1000,
                                )

gym_a3c_hyperspecific = Hyper_a3c(
                                       loss_v=0.5,
                                       loss_entropy=0.01,
                                       brain_memory_size=16
                                       )

gym_cart_env = Gym_param()
gym_pong_env = Gym_param(
                         problem='Pong-v0',
                         wrapper='Gym_pong_wrapper'
                         )
hex_env = Real_param(
                     problem='Hexagon',
                     wrapper='Real_base_wrapper',
                     module_name='OpenHexagonEmulator',
                     class_name='HexagonEmulator',
                     game_args=hex_screen
                     )

hex_hyperparams = Hyperparam(
                 gamma=0.99,
                 batch=16,
                 explore=3000,
                 epsilon_init=1.0,
                 epsilon_final=0.01,
                 memory_size=50000,
                 save_rate=10000,
                 img_channels=2,
                 extra=ddqn_hyperspecific
                 )

hex_incongruence = Args(
                        algorithm='ddqn',
                        mode='train',
                        env=hex_env,
                        data='default',
                        screen=hex_screen,
                        hyper=hex_hyperparams,
                        directory='default',
                        memory_delay=0.5,
                        model='model_mid_cnn'
                        )

hex_base = Args(
                        algorithm='ddqn',
                        mode='train',
                        env=hex_env,
                        data='default',
                        screen=hex_screen,
                        hyper=hex_hyperparams,
                        directory='default',
                        memory_delay=4.5,
                        model='model_mid_cnn'
                        )



gym_cart_ddqn_hyperparams = Hyperparam(
                             gamma=0.99,
                             batch=8,
                             explore=30000,
                             epsilon_init=0.4,
                             epsilon_final=0.15,
                             memory_size=7500,
                             save_rate=10000,
                             img_channels=1,
                             learning_rate=5e-3,
                             extra=ddqn_hyperspecific
                           )

gym_cart_ddqn = Args(
                        algorithm='ddqn',
                        mode='train',
                        env=gym_cart_env,
                        data='default',
                        screen='default',
                        hyper=gym_cart_ddqn_hyperparams,
                        directory='default',
                        memory_delay=4,
                        model='model_mid_default'
                        )

gym_pong_ddqn_hyperparams = Hyperparam(
                             gamma=0.99,
                             batch=8,
                             explore=30000,
                             epsilon_init=0.4,
                             epsilon_final=0.15,
                             memory_size=7500,
                             save_rate=10000,
                             img_channels=1,
                             learning_rate=2.5e-4,
                             extra=ddqn_hyperspecific
                           )

gym_pong_ddqn = Args(
                        algorithm='ddqn',
                        mode='train',
                        env=gym_pong_env,
                        data='default',
                        screen='default',
                        hyper=gym_pong_ddqn_hyperparams,
                        directory='default',
                        memory_delay=4,
                        model='model_mid_atari'
                        )

gym_pong_a3c_hyperparams = Hyperparam(
                             gamma=0.99,
                             batch=16,
                             explore=30000,
                             epsilon_init=0.4,
                             epsilon_final=0.01,
                             memory_size=2,
                             save_rate=10000,
                             img_channels=4,
                             learning_rate=2.5e-4,
                             extra=gym_a3c_hyperspecific
                           )

gym_pong_a3c = Args(
                        algorithm='a3c',
                        mode='train',
                        env=gym_pong_env,
                        data='default',
                        screen='default',
                        hyper=gym_pong_a3c_hyperparams,
                        directory='default',
                        model='model_mid_atari'
                        )

gym_cart_a3c_hyperparams = Hyperparam(
                             gamma=0.99,
                             batch=16,
                             explore=3000,
                             epsilon_init=0.4,
                             epsilon_final=0.1,
                             memory_size=2,
                             save_rate=100000,
                             img_channels=1,
                             learning_rate=5e-3,
                             extra=gym_a3c_hyperspecific,
                           )

gym_cart_a3c = Args(
                        algorithm='a3c',
                        mode='train',
                        env=gym_cart_env,
                        data='default',
                        screen='default',
                        hyper=gym_cart_a3c_hyperparams,
                        directory='default',
                        memory_delay=4,
                        model='model_mid_default'
                        )

hex_base_a3c_hyperspecific = Hyper_a3c(
                                       loss_v=0.5,
                                       loss_entropy=0.01,
                                       brain_memory_size=40000
                                       )

hex_base_a3c_hyperparams = Hyperparam(
                             gamma=0.99,
                             batch=64,
                             explore=1000,
                             epsilon_init=1,
                             epsilon_final=0.01,
                             memory_size=4,
                             save_rate=5000,
                             img_channels=2,
                             learning_rate=2.5e-4, # 2.5e-4
                             extra=hex_base_a3c_hyperspecific,
                           )

hex_base_a3c = Args(
                        algorithm='a3c',
                        mode='train',
                        env=hex_env,
                        data='default',
                        screen=hex_screen,
                        hyper=hex_base_a3c_hyperparams,
                        directory='default',
                        memory_delay=4.5,
                        model='model_mid_cnn'
                        )

hex_incongruence_a3c_hyperspecific = Hyper_a3c(
                                       loss_v=0.5,
                                       loss_entropy=0.01,
                                       brain_memory_size=40000
                                       )

hex_incongruence_a3c_hyperparams = Hyperparam(
                             gamma=0.99,
                             batch=64,
                             explore=10000,
                             epsilon_init=1,
                             epsilon_final=0.01,
                             memory_size=4,
                             save_rate=5000,
                             img_channels=2,
                             learning_rate=2.5e-4, # 2.5e-4
                             extra=hex_incongruence_a3c_hyperspecific,
                           )

hex_incongruence_a3c = Args(
                        algorithm='a3c',
                        mode='train',
                        env=hex_env,
                        data='default',
                        screen=hex_screen,
                        hyper=hex_incongruence_a3c_hyperparams,
                        directory='default',
                        memory_delay=0.5,
                        model='model_mid_cnn'
                        )

hex_base_a3c_load = Args(
                        algorithm='a3c',
                        mode='train_old',
                        env=hex_env,
                        data='default',
                        screen=hex_screen,
                        hyper=hex_base_a3c_hyperparams,
                        directory='hex_acer_hexreal_v1',
                        memory_delay=4.5,
                        run_count_load=3387,
                        model='model_mid_cnn'
                        )

hex_incongruence_a3c_load = Args(
                        algorithm='a3c',
                        mode='train_old',
                        env=hex_env,
                        data='default',
                        screen=hex_screen,
                        hyper=hex_incongruence_a3c_hyperparams,
                        directory='hex_acer_end_v1',
                        memory_delay=0.5,
                        run_count_load=1511,
                        model='model_mid_cnn'
                        )

hex_pi_acer_load = Args(
                        algorithm='a3c',
                        mode='train_old',
                        env=hex_env,
                        data='default',
                        screen=hex_screen,
                        hyper=hex_incongruence_a3c_hyperparams,
                        directory='hex_acer_pi_v1',
                        memory_delay=0.5,
                        run_count_load=6412,
                        model='model_mid_cnn'
                        )



hex_gather_hyperparams = Hyperparam(
                             gamma=0.99,
                             batch=64,
                             explore=1000,
                             epsilon_init=1.0,
                             epsilon_final=0.01,
                             memory_size=1000,
                             save_rate=10000,
                             img_channels=2,
                           )

hex_base_gather = Args(
                        algorithm='random',
                        mode='train',
                        env='memory', # Fix this later
                        data='default',
                        screen=hex_screen,
                        hyper=hex_gather_hyperparams,
                        directory='default',
                        memory_delay=4.5
                        )