import tensorflow as tf
from tensorflow.keras import layers, optimizers, callbacks
import time
import itertools

from multiprocessing import Pool, Process, cpu_count
from plot_efficient_frontier import plot_frontier_carlos as plot_frontier
import numpy as np
from auxiliary_functions import *
from life_cycle_gradient_descent import optimize



def construct_grid():

    interval_grid = np.random.rand(10000, 3)

    #grid = np.array(list(itertools.product(interval_grid, interval_grid, interval_grid)))
    grid = interval_grid
    results = []
    x0 = 1
    lam_x0 = 0
    lam_g0 = 0
    for i, tax_policy in enumerate(grid):
        util, revenue, _,  x0, lam_x0, lam_g0 = optimize(lb_x, ub_x, S, tax_policy, wage, r, conf, x0, lam_x0, lam_g0)
        if S[1].stats()["success"]:
            results.append([util, revenue])
        else:
            continue

    results = np.array(results)
    return grid, results

def init_tf_model(NN_structure = [100,100, 50]):
    # model inputs
    inputs = tf.keras.Input(shape=(3,))
    activation_function = 'relu'

    # first layer after inputs
    x = layers.Dense(NN_structure[0], activation=activation_function)(inputs)
    # build hidden layers
    for nodes_in_layer in NN_structure[1:]:
        x = layers.Dense(nodes_in_layer, activation=activation_function)(x)
    # output layer
    prediction = layers.Dense(1, activation='linear')(x)

    model = tf.keras.Model(inputs=inputs, outputs=prediction)

    # The compile step specifies the training configuration.
    model.compile(optimizer=tf.keras.optimizers.Adadelta(lr=1.0, rho=0.95),
                  loss='mse',
                  metrics=['mae', 'accuracy'])

    return model

def fit_model(model):
    # define callbacks to be executed during training
    checkpoints = callbacks.ModelCheckpoint("checkpoint_model.h5", monitor='val_loss', verbose=0, save_best_only=True,
                                            save_weights_only=False, mode='auto', period=1000)

    tensorboard = callbacks.TensorBoard(log_dir='./tensorboard_data', histogram_freq=100, batch_size=32,
                                        write_graph=True, write_grads=False, write_images=True,
                                        embeddings_freq=0, embeddings_layer_names=None,
                                        embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=1E-4, patience=200, verbose=0,
                                             mode='min', baseline=None, restore_best_weights=False)

    callback_list = [checkpoints, tensorboard, early_stopping]

    model.fit(data_x, data_y, batch_size=len(data_x)//2, epochs=20000,
              validation_split=0.15, shuffle=True, verbose=1)

    model.save("trained_model.h5")


    return model

if __name__ == '__main__':
    conf = {
        'tax_setting': 5,  # tax setting: see below for detailed explanation
        'beta': 0.95,  # Utility discounting
        'r_debt': 0.15,  # debt interest rate
        'r': 0.1,
        'T': 60,  # agent's maturity
        'T_ret': 45,  # retirement age
        'L_max': 10,  # maximum labour
        'gamma': 0.5,  # Agent's utility from consumption
        'eta': 8,  # Agent's disutility from labour
        'plot': False,
        'plot_steps': False,
        'plot_GD_steps': False,
        'h': 1E-6,  # step size finite difference
        'n_steps_max': 100,  # max number of steps for 1 point on pareto frontier
        'k_max': 100,  # max number of steps for bi-linear search
        'tol': 1E-3,  # tolerance for gradient stopping
        'tol_line_search': 1E-8,  # tolerance to abort line search
        'serial': False,
        'max_debt': 2,
        'costb': 0.02,
        'borrowing': np.inf,  # np.inf -> allow unlimited borrowing // 0 -> allow no borrowing
        'n_starting_points': 150  # number of starting points for "frontier searching"
    }
    timestr = time.strftime("%m-%d-%H%M")

    # extract parameters from configuration dictionary
    gamma = conf["gamma"]
    eta = conf["eta"]
    n_points = conf["n_starting_points"]
    r = conf["r"]
    T = conf["T"]

    # build interest rate vector of length
    r = np.full(T, r)

    # set n_procs
    n_procs = cpu_count() - 1
    #    n_procs = 1

    # create gamma eta pairs from input
    #gammas_etas = itertools.product(gamma, eta)

    # n points per process
    n = n_points // n_procs
    wage = wage_function(conf)

    # initialize NLP problem - S = casadi optimization object - lb_x ub_x are the lower and upper bounds
    S, lb_x, ub_x = init_optimization_problem(r, wage, conf)

    data_x,  data_y = construct_grid()
    model = init_tf_model()

    fit_model(model)


