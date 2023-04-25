import mnist
import model_utils
import csv
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from mapping import get_scaling_f, g_to_w_at_g_max, w_to_g_at_g_max
import math

def show_nonlinearity():
    input = np.arange(0,1,0.01)

    g_1 = np.ones(input.shape) / 1000.0
    g_2 = g_1 + 1.0 / 1000.0
    #g_2 = g_1 + np.arange(0,1,0.01) / 1000.0

    out_linear_1 = g_1 * input
    out_linear_2 = g_2 * input


    # actual code
    # g_w_nonlinear = tf.math.add(g_w, tf.math.multiply(extended_input_2 * tf.constant(0.7), 2-g_w))


    # dummy implementation
    constant = 0.8
    g_nonlinear_1 = g_1 + ((input * constant) * (0.001 * 0.001/g_1))
    g_nonlinear_2 = g_2 + ((input * constant) * (0.001 * 0.001/g_2))


    out_nonlinear_1 = g_nonlinear_1 * input
    out_nonlinear_2 = g_nonlinear_2 * input

    ax = plt.subplots()
    plt.plot(input,out_linear_1, color="red")
    plt.plot(input,out_nonlinear_1, color="orange")
    plt.plot(input,out_linear_2, color="green")
    plt.plot(input,out_nonlinear_2, color="blue")
    plt.legend(["linear g=1mS", "nonlinear g=1mS", "linear g=2mS", "nonlinear g=2mS"])
    plt.xlabel("input")
    plt.ylabel("conductance, S")
    plt.show()



def show_training():
    input = np.arange(0, 1, 0.01)

    g_1 = np.ones(input.shape) / 1000.0
    g_2 = g_1 + 1.0 / 1000.0

    constant = 0.8
    g_nonlinear_1 = g_1 + ((input * constant) * (0.001 * 0.001 / g_1))
    g_nonlinear_2 = g_2 + ((input * constant) * (0.001 * 0.001 / g_2))

    w_pos = (g_nonlinear_2 - g_nonlinear_1) * 1000
    out_nonlinear_1 = w_pos * input

    w = 1.0
    z = 0.5
    w_x = w * (1 - z * constant * (1 - 0.5 * w))
    #w_x = w * (1.0 - z * (1 - constant * 0.5 * w))
    out_3 = w_x * input


    g_3 = g_1 + 0.5 / 1000.0
    g_nonlinear_3 = g_3 + ((input * constant) * (0.001 * 0.001 / g_3))
    w_pos_2 = (g_nonlinear_3 - g_nonlinear_1) * 1000
    out_nonlinear_2 = w_pos_2 * input

    w_2 = 0.5
    w_x_2 = w_2 * (1 - z * constant * (1 - 0.5 * w_2))  # constant = 0.7 at w=0.5; 0.9 for 0.1; 0.5 for 1
    #w_x_2 = w_2 * (1 - 4 * 0.5 * constant)
    out_4 = w_x_2 * input


    ax = plt.subplots()
    plt.plot(input, out_nonlinear_1, color="red")
    plt.plot(input, out_3, color="green")
    plt.plot(input, out_nonlinear_2, color="red")
    plt.plot(input, out_4, color="green")
    plt.legend(["simulated w nonlinearity","approximation"])
    plt.xlabel("input")
    plt.ylabel("effective weight")
    plt.show()

def show_w_w():

    input_2 = np.arange(0, 1, 0.01)
    input = np.ones(input_2.shape) * 0.5
    g_1 = np.ones(input.shape) / 1000.0
    g_2 = g_1 + np.arange(0, 1, 0.01) / 1000.0


    constant = 1
    g_nonlinear_1 = g_1 + ((input * constant) * (0.001 * 0.001 / g_1))
    g_nonlinear_2 = g_2 + ((input * constant) * (0.001 * 0.001 / g_2))

    w_pos_nonlinear = (g_nonlinear_2 - g_nonlinear_1) * 1000
    w_pos = (g_2 - g_1) * 1000

    out_nonlinear_1 = w_pos_nonlinear * input

    w = np.arange(0, 1, 0.01)
    #w_x = w * (0.5 + 0.25 * constant * w)      # constant = 0.7 at w=0.5; 0.9 for 0.1; 0.5 for 1
    #w_x = w * (1 - 0.4 * constant * w * w)
    z = 0.5
    constant2 = 0.7
    w_x = w * (1 - z * constant2 * (1 - 0.5 * w))
    #w_x = w * (1.0 - z * (1 - constant * 0.5 * w))
    out_3 = w_x * input


    g_3 = g_1 + 0.5 / 1000.0
    g_nonlinear_3 = g_3 + ((input * constant) * (0.001 * 0.001 / g_3))
    w_pos_2 = (g_nonlinear_3 - g_nonlinear_1) * 1000
    out_nonlinear_2 = w_pos_2 * input


    #w_x_2 = w * (1 - 0.4 * constant * 1.0 / 0.25)  # constant = 0.7 at w=0.5; 0.9 for 0.1; 0.5 for 1
    z=0.5
    constant3 = 0.6
    w_x_2 = w * (1 - z * constant3 * (1 - 0.5 * w))
    out_4 = w_x_2 * input

    z=0.5
    constant3 = 0.8
    w_x_3 = w * (1 - z * constant3 * (1 - 0.5 * w))
    out_5 = w_x_3 * input


    ax = plt.subplots()
    plt.plot(w, w_x, color="blue")
    plt.plot(w, w_x_2, color="green")
    plt.plot(w, w_x_3, color="orange")
    plt.plot(w_pos, w_pos_nonlinear, color="red")
    plt.xlabel("actual weight")
    plt.ylabel("effective weight")
    plt.legend(["training","simulation"])
    plt.show()

def show_ratio_w():


    input_val = 1
    constant = 1

    input_2 = np.arange(0.01, 1, 0.01)
    input = np.ones(input_2.shape) * input_val
    g_1 = np.ones(input.shape) / 1000.0
    g_2 = g_1 + np.arange(0.01, 1, 0.01) / 1000.0
    g_nonlinear_1 = g_1 + ((input * constant) * (0.001 * 0.001 / g_1))
    g_nonlinear_2 = g_2 + ((input * constant) * (0.001 * 0.001 / g_2))
    w_pos_nonlinear = (g_nonlinear_2 - g_nonlinear_1) * 1000
    w = np.arange(0.01, 1, 0.01)


    ratio_1 = w_pos_nonlinear/w


    input_val = 1
    constant = 0.5

    input_2 = np.arange(0.01, 1, 0.01)
    input = np.ones(input_2.shape) * input_val
    g_1 = np.ones(input.shape) / 1000.0
    g_2 = g_1 + np.arange(0.01, 1, 0.01) / 1000.0
    g_nonlinear_1 = g_1 + ((input * constant) * (0.001 * 0.001 / g_1))
    g_nonlinear_2 = g_2 + ((input * constant) * (0.001 * 0.001 / g_2))
    w_pos_nonlinear = (g_nonlinear_2 - g_nonlinear_1) * 1000
    w = np.arange(0.01, 1, 0.01)


    ratio_2 = w_pos_nonlinear/w

    input_val = 1
    constant = 0.25

    input_2 = np.arange(0.01, 1, 0.01)
    input = np.ones(input_2.shape) * input_val
    g_1 = np.ones(input.shape) / 1000.0
    g_2 = g_1 + np.arange(0.01, 1, 0.01) / 1000.0
    g_nonlinear_1 = g_1 + ((input * constant) * (0.001 * 0.001 / g_1))
    g_nonlinear_2 = g_2 + ((input * constant) * (0.001 * 0.001 / g_2))
    w_pos_nonlinear = (g_nonlinear_2 - g_nonlinear_1) * 1000
    w = np.arange(0.01, 1, 0.01)


    ratio_3 = w_pos_nonlinear/w

    input_val = 0.48
    constant = 1

    input_2 = np.arange(0.01, 1, 0.01)
    input = np.ones(input_2.shape) * input_val
    g_1 = np.ones(input.shape) / 1000.0
    g_2 = g_1 + np.arange(0.01, 1, 0.01) / 1000.0
    g_nonlinear_1 = g_1 + ((input * constant) * (0.001 * 0.001 / g_1))
    g_nonlinear_2 = g_2 + ((input * constant) * (0.001 * 0.001 / g_2))
    w_pos_nonlinear = (g_nonlinear_2 - g_nonlinear_1) * 1000
    w = np.arange(0.01, 1, 0.01)

    ratio_4 = w_pos_nonlinear / w

    input_val = 0.23
    constant = 1

    input_2 = np.arange(0.01, 1, 0.01)
    input = np.ones(input_2.shape) * input_val
    g_1 = np.ones(input.shape) / 1000.0
    g_2 = g_1 + np.arange(0.01, 1, 0.01) / 1000.0
    g_nonlinear_1 = g_1 + ((input * constant) * (0.001 * 0.001 / g_1))
    g_nonlinear_2 = g_2 + ((input * constant) * (0.001 * 0.001 / g_2))
    w_pos_nonlinear = (g_nonlinear_2 - g_nonlinear_1) * 1000
    w = np.arange(0.01, 1, 0.01)

    ratio_5 = w_pos_nonlinear / w

    input_val = 0.98
    constant = 1

    input_2 = np.arange(0.01, 1, 0.01)
    input = np.ones(input_2.shape) * input_val
    g_1 = np.ones(input.shape) / 1000.0
    g_2 = g_1 + np.arange(0.01, 1, 0.01) / 1000.0
    g_nonlinear_1 = g_1 + ((input * constant) * (0.001 * 0.001 / g_1))
    g_nonlinear_2 = g_2 + ((input * constant) * (0.001 * 0.001 / g_2))
    w_pos_nonlinear = (g_nonlinear_2 - g_nonlinear_1) * 1000
    w = np.arange(0.01, 1, 0.01)

    ratio_6 = w_pos_nonlinear / w

    # # w_x = w * (0.5 + 0.25 * constant * w)      # constant = 0.7 at w=0.5; 0.9 for 0.1; 0.5 for 1
    # # w_x = w * (1 - 0.4 * constant * w * w)
    # z = 0.25
    # w_x = w * (1 - z * (1 - 0.5 * constant * w))
    # out_3 = w_x * input
    #
    # g_3 = g_1 + 0.5 / 1000.0
    # g_nonlinear_3 = g_3 + ((input * constant) * (0.001 * 0.001 / g_3))
    # w_pos_2 = (g_nonlinear_3 - g_nonlinear_1) * 1000
    # out_nonlinear_2 = w_pos_2 * input
    #
    # w_x_2 = w * (1 - 0.4 * constant * 1.0 / 0.25)  # constant = 0.7 at w=0.5; 0.9 for 0.1; 0.5 for 1
    # out_4 = w_x_2 * input

    ax = plt.subplots()
    plt.plot(w, ratio_1)
    plt.plot(w, ratio_2)
    plt.plot(w, ratio_3)
    # plt.plot(w, ratio_4)
    # plt.plot(w, ratio_5)
    # plt.plot(w, ratio_6)
    #plt.plot(w_pos, w_pos_nonlinear, color="red")
    plt.xlabel("weight")
    plt.ylabel("w_eff/w")
    plt.legend(["C=1", "C=0.5", "C=0.25"])
    plt.show()



#show_training()
show_w_w()
#show_ratio_w()
