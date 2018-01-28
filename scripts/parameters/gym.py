# By Nick Erickson
# Contains parameters for games and levels

from parameters.param_utils import Args, Hyperparam, HyperA3C, HyperDDQN, GymParam

a3c_hyperspecific = HyperA3C(
                                       loss_v=0.5,
                                       loss_entropy=0.01,
                                       brain_memory_size=128
                                       )

ddqn_hyperspecific = HyperDDQN(
                                observe=7500,
                                update_rate=1000,
                                )

cart_env = GymParam()
pong_env = GymParam(
                         problem='Pong-v0',
                         wrapper='GymPongWrapper'
                         )
breakout_env = GymParam(
                         problem='Breakout-v0',
                         wrapper='GymBreakoutWrapper'
                         )

cart_ddqn_hyperparams = Hyperparam(
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

cart_ddqn = Args(
                        algorithm='ddqn',
                        mode='train',
                        env=cart_env,
                        data=None,
                        hyper=cart_ddqn_hyperparams,
                        directory='default',
                        memory_delay=4,
                        model='model_mid_default'
                        )

pong_ddqn_hyperparams = Hyperparam(
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

pong_ddqn = Args(
                        algorithm='ddqn',
                        mode='train',
                        env=pong_env,
                        data=None,
                        hyper=pong_ddqn_hyperparams,
                        directory='default',
                        memory_delay=4,
                        model='model_mid_cnn_84x84_pool'
                        )

breakout_a3c_hyperparams = Hyperparam(
                             gamma=0.99,
                             batch=128,
                             explore=100000,
                             epsilon_init=1,
                             epsilon_final=0.05,
                             memory_size=1,
                             save_rate=100000,
                             img_channels=4,
                             learning_rate=1e-3,
                             extra=a3c_hyperspecific
                           )

breakout_a3c = Args(
                        algorithm='a3c',
                        mode='train',
                        env=breakout_env,
                        data=None,
                        hyper=breakout_a3c_hyperparams,
                        directory='default',
                        model='model_mid_cnn_84x84_pool'
                        )

pong_a3c_hyperparams = Hyperparam(
                             gamma=0.99,
                             batch=16,
                             explore=100000,
                             epsilon_init=1,
                             epsilon_final=0.15,
                             memory_size=2,
                             save_rate=10000,
                             img_channels=4,
                             learning_rate=2.5e-4,
                             extra=a3c_hyperspecific
                           )

pong_a3c = Args(
                        algorithm='a3c',
                        mode='train',
                        env=pong_env,
                        data=None,
                        hyper=pong_a3c_hyperparams,
                        directory='default',
                        model='model_mid_cnn_84x84_pool'
                        )

cart_a3c_hyperparams = Hyperparam(
                             gamma=0.99,
                             batch=16,
                             explore=3000,
                             epsilon_init=0.4,
                             epsilon_final=0.1,
                             memory_size=2,
                             save_rate=100000,
                             img_channels=1,
                             learning_rate=5e-3,
                             extra=a3c_hyperspecific,
                           )

cart_a3c = Args(
                        algorithm='a3c',
                        mode='train',
                        env=cart_env,
                        data=None,
                        screen='default',
                        hyper=cart_a3c_hyperparams,
                        directory='default',
                        memory_delay=4,
                        model='model_mid_default'
                        )
