import mnist
import model_utils
import csv
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from mapping import get_scaling_f, g_to_w_at_g_max, w_to_g_at_g_max
import math
import tensorflow_probability as tfp

#
# a = np.random.normal(0,0.05, 10000)
# plt.hist(a, 100)
# plt.show()

# std = 0.1
# g = np.random.normal(0, 0.2, (100,100))
#
# disturbed_g = np.where(g != 0, np.random.lognormal(g, math.sqrt(math.log((math.pow(std, 2) / math.pow(g, 2)) + 1))), 0)


# g = np.random.normal(1.5, 0.1, (100,100))
# ax = plt.subplots()
# plt.hist(g.flat, 100, density=True)
# plt.show()

# print(g)

#convert mean and std to distribution parameters mu and sigma
# mean = 1000
# std = 0.5 * mean
#
# variance = math.pow(std,2)
# sigma_sqr = math.log((variance/math.pow(mean,2)) + 1)
# sigma = math.sqrt(sigma_sqr)
# mu = math.log(mean) - (sigma_sqr)/2
#
#
# dist_1 = np.random.lognormal(mu, sigma, 10000)
# dist_clean_1 = np.where(dist_1 > 5000, 5000, dist_1)
#
# mean = 2000
# std = 0.5 * mean
# variance = math.pow(std,2)
# sigma_sqr = math.log((variance/math.pow(mean,2)) + 1)
# sigma = math.sqrt(sigma_sqr)
# mu = math.log(mean) - (sigma_sqr)/2
#
#
# dist_2 = np.random.lognormal(mu, sigma, 10000)
# dist_clean_2 = np.where(dist_2 > 5000, 5000, dist_2)
#
# ax = plt.subplots()
# plt.hist(dist_clean_1, 100, density=True, alpha=0.5)
# plt.hist(dist_clean_2, 100, density=True, alpha=0.5, color="red")
# plt.show()

# r = 1/(0.001*g)
# r_disturbed = np.zeros(r.shape)
# for i in range(r.shape[0]):
#     for x in range(r.shape[1]):
#
#         mean = r[i,x]
#         std = 0.25 * mean
#
#         variance = math.pow(std,2)
#         sigma_sqr = math.log((variance/math.pow(mean,2)) + 1)
#         sigma = math.sqrt(sigma_sqr)
#         mu = math.log(mean) - (sigma_sqr)/2
#
#         r_disturbed[i,x] = np.random.lognormal(mu, sigma)
#
# g_disturbed = 1000/r_disturbed
# print(g_disturbed)
#
# fig, ax = plt.subplots()
# ax.hist(g.flat, 100, density=True, alpha=0.5)
# ax.hist(g_disturbed.flat, 100, density=True, color="red", alpha=0.5)
# plt.show()



std = 0.1
r = tf.constant([2.0, 2.0])
#
# mean = r
# this_std = std * mean
#
# variance = tf.pow(this_std, 2.0)
# sigma_sqr = tf.math.log((variance / tf.pow(mean, 2.0)) + 1.0)
# sigma = tf.sqrt(sigma_sqr)
# mu = tf.math.log(mean) - (sigma_sqr) / 2.0
#
# r_disturbed = tf.random.lognormal(mu, sigma)
#
# g_disturbed = 1.0 / r_disturbed
#
# print(g_disturbed)