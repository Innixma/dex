# By Nick Erickson
# Contains parameters for games and levels

from parameters.param_utils import Args, Hyperparam, Screenparam, Hyper_a3c, Real_param

screen = Screenparam(
                         app='Open Hexagon 1.92 - by vittorio romeo',
                         size=[188,188],
                         #zoom=[12,2],
                         zoom=[10,10],
                         framerate=50,
                         scale=4
                         )

a3c_hyperspecific = Hyper_a3c(
                                       loss_v=0.5,
                                       loss_entropy=0.01,
                                       brain_memory_size=40000
                                       )

gather_a3c_hyperspecific = Hyper_a3c(brain_memory_size=500)

env = Real_param(
                     problem='Hexagon',
                     wrapper='Real_base_wrapper',
                     module_name='OpenHexagonEmulator',
                     class_name='HexagonEmulator',
                     game_args=screen
                     )

a3c_hyperparams = Hyperparam(
                             gamma=0.99,
                             batch=64,
                             explore=10000,
                             epsilon_init=1,
                             epsilon_final=0.00,
                             memory_size=4,
                             save_rate=5000,
                             img_channels=1,
                             learning_rate=2.5e-4, # 2.5e-4
                             extra=a3c_hyperspecific,
                           )

gather_a3c_hyperparams = Hyperparam(
                             gamma=0.99,
                             batch=64,
                             explore=1000000,
                             epsilon_init=1,
                             epsilon_final=1,
                             memory_size=4,
                             save_rate=999999,
                             img_channels=2,
                             learning_rate=2.5e-4, # 2.5e-4
                             extra=gather_a3c_hyperspecific,
                           )

base_a3c = Args(
                        algorithm='a3c',
                        mode='train',
                        env=env,
                        data=None,
                        screen=screen,
                        hyper=a3c_hyperparams,
                        directory='default',
                        memory_delay=4,
                        model='model_mid_cnn_42x42'
                        )

gather_a3c = Args(
                        algorithm='a3c',
                        mode='gather',
                        env=env,
                        data=None,
                        screen=screen,
                        hyper=gather_a3c_hyperparams,
                        directory='default',
                        memory_delay=3.5,
                        model='model_mid_cnn_42x42'
                        )

incongruence_a3c = Args(
                        algorithm='a3c',
                        mode='train',
                        env=env,
                        #data='hex_gather_rotfast',
                        screen=screen,
                        hyper=a3c_hyperparams,
                        directory='default',
                        memory_delay=0.5,
                        model='model_mid_cnn_42x42'
                        )

hex_base_a3c_load = Args(
                        algorithm='a3c',
                        mode='train_old',
                        env=env,
                        data=None,
                        screen=screen,
                        hyper=a3c_hyperparams,
                        directory='hex_a3c_base_hard_v1',
                        memory_delay=4,
                        run_count_load=629,
                        model='model_mid_cnn_42x42'
                        )

hex_thinkfast_a3c = Args(
                        algorithm='a3c',
                        mode='train',
                        env=env,
                        #data='hex_gather_rotfast',
                        screen=screen,
                        hyper=a3c_hyperparams,
                        directory='default',
                        memory_delay=3.5,
                        model='model_mid_cnn_42x42'
                        )

hex_incongruence_a3c_load = Args(
                        algorithm='a3c',
                        mode='run',
                        env=env,
                        data=None,
                        screen=screen,
                        hyper=a3c_hyperparams,
                        directory='hex_acer_rotfast_v2_1channel',
                        memory_delay=0.5,
                        run_count_load=1511,
                        model='model_mid_cnn_42x42',
                        weight_override='model_frame_12452'
                        )
