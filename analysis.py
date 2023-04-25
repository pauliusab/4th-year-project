import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib.colors import ListedColormap
import math
import tensorflow as tf



def read_summary(dir):
    # read history csv file
    file = open(dir, "r")
    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)

    rows = []
    for row in csvreader:
        rows.append(row)

    file.close()


    # extract required data
    history_dict = {}
    collumn = []

    for i in range(len(header)):
        for x in range(len(rows)):
            collumn.append(float(rows[x][i]))       # need to add float(), otherwise reads as string
        history_dict.update({header[i]: collumn})
        collumn = []

    return history_dict

def get_acc(dict, **kwargs):
    x = kwargs.pop("x", "std")
    y = kwargs.pop("y", "acc")

    print(dict[x])
    std_list = list(set(dict[x]))
    std_list.sort()


    acc_list = []

    for i in std_list:
        n_list = []
        for n in range(len(dict[x])):
            if dict[x][n] == i:
                n_list.append(dict[y][n])
        acc_list.append(n_list)


    print(std_list)
    print(np.asarray(acc_list).shape)

    return std_list, acc_list

def plot_acc_std_analysis(std_list, mean_list, quantiles, quantile_list, title, **kwargs):
    x = std_list


    fig, ax = plt.subplots()

    ax.plot(x, mean_list, "-o")


    if (quantiles == 2):
        y_0 = quantile_list[0]
        y_25 = quantile_list[1]
        y_50 = quantile_list[2]
        y_75 = quantile_list[3]
        y_100 = quantile_list[4]
        ax.fill_between(x, y_0, y_100, alpha=.25, linewidth=0)
        ax.legend(['0-100%', '25-75%', "mean"])
    elif (quantiles == 1):
        y_25 = quantile_list[0]
        y_50 = quantile_list[1]
        y_75 = quantile_list[2]
        ax.legend(['25-75%', "mean"])

    ax.fill_between(x, y_25, y_75, alpha=.5, linewidth=0, color="orange")

    if (quantiles == 2):
        ax.legend(["mean", '0-100% percentiles', '25-75% percentiles'])
    elif (quantiles == 1):
        ax.legend(["mean", '25-75% percentiles'])

    # ax.set_title(title, fontsize=16)
    ax.set_ylabel(kwargs.pop("ylabel", "Accuracy"))
    ax.set_xlabel(kwargs.pop("xlabel", 'Standard deviation'))
    ax.grid(visible=True)


    # ax.set_xlim(xmin=xmin, xmax=xmax)
    # ax.set_ylim(ymin=ymin, ymax=ymax)

    plt.show()

def get_acc_analysis(dir, title, **kwargs):

    quantiles = kwargs.pop("quantiles", None)
    dict = read_summary(dir)
    std_list, acc_list = get_acc(dict, **kwargs)

    if(quantiles == 2):
        quantile_list = np.quantile(acc_list, [0, 0.25, 0.50, 0.75, 1], axis=1)
        mean_list = []
        for i in range(np.asarray(acc_list).shape[0]):
            mean_list.append(np.mean(acc_list[i]))

        print(quantile_list)
        print(quantile_list.shape)
        plot_acc_std_analysis(std_list, mean_list, quantiles, quantile_list, title, **kwargs)
    elif(quantiles == 1):
        quantile_list = np.quantile(acc_list, [0.25, 0.50, 0.75], axis=1)
        mean_list = []
        for i in range(np.asarray(acc_list).shape[0]):
            mean_list.append(np.mean(acc_list[i]))

        plot_acc_std_analysis(std_list, mean_list, quantiles, quantile_list, title, **kwargs)

    else:
        fig, ax = plt.subplots()
        ax.plot(std_list, acc_list)

        #ax.set_title(title, fontsize=16)
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Standard deviation')
        ax.grid(visible=True)
        plt.show()

    return std_list, acc_list, quantile_list

def get_acc_analysis2(dir, **kwargs):

    plt.figure(figsize=(8,6))
    title = kwargs.pop("title", None)

    for i in dir:
        dict = read_summary(i)
        std_list, acc_list = get_acc(dict, **kwargs)
        mean_list = []
        for i in range(np.asarray(acc_list).shape[0]):
            mean_list.append(np.mean(acc_list[i]))
        plt.plot(std_list, mean_list, "-o")

    if title != None:
        plt.title(title, fontsize=16)
    plt.ylabel(kwargs.pop("ylabel", "Accuracy"), fontsize=16)
    plt.xlabel(kwargs.pop("xlabel", 'Standard deviation'), fontsize=16)
    plt.legend(kwargs.pop("legend", ["mean"]), loc='upper right', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(visible=True)
    plt.show()


def get_acc_analysis3(dir, n_test, **kwargs):

    plt.figure(figsize=(8,6))
    title = kwargs.pop("title", None)
    x = kwargs.get("x", "std")

    #print(dict[x])

    for i in dir:
        dict = read_summary(i)
        std_list, acc_list = get_acc(dict, **kwargs)
        print(acc_list)
        mean_list = []
        for a in range(np.asarray(acc_list).shape[0]):
            for b in range(0, np.asarray(acc_list).shape[1], n_test):
                this_arr = []
                for c in range(0, n_test):
                    this_acc = acc_list[a][b+c]
                    this_arr.append(this_acc)
                mean_list.append(np.mean(this_arr))

        print(mean_list)
        print(np.asarray(mean_list).shape)

        this_dict = np.repeat(np.asarray(std_list), np.asarray(mean_list).shape[0] / np.asarray(acc_list).shape[0], -1)
        print(this_dict)
        print(this_dict.shape)

        plt.plot(mean_list, "-o")

    if title != None:
        plt.title(title, fontsize=16)
    plt.ylabel(kwargs.pop("ylabel", "Accuracy"), fontsize=16)
    plt.xlabel(kwargs.pop("xlabel", 'Standard deviation'), fontsize=16)
    plt.legend(kwargs.pop("legend", ["mean"]), fontsize=16)
    plt.tick_params(axis='y', which='major', labelsize=14)
    # plt.tick_params(
    #     axis='x',  # changes apply to the x-axis
    #     which='both',  # both major and minor ticks are affected
    #     bottom=False,  # ticks along the bottom edge are off
    #     top=False,  # ticks along the top edge are off
    #     labelbottom=False)  # labels along the bottom edge are off
    plt.grid(visible=True)
    plt.show()

# std_list, acc_list, quantile_list = get_acc_analysis(
#     "nonideality_analysis/L1/d2d_std_0.2/1.csv", 'Inference accuracy with L1 regularization for d2d std 0.2', quantiles=1, x="alpha", xlabel="alpha")

# std_list, acc_list, quantile_list = get_acc_analysis(
#     "custom_training/noise_analysis2/d2d_0.1.csv", 'noisy training for d2d std 0.1', quantiles=1, x="training_std", xlabel="training stddev")
#
# std_list, acc_list, quantile_list = get_acc_analysis(
#     "custom_training/noise_analysis2/d2d_0.2.csv", 'noisy training for d2d std 0.2', quantiles=1, x="training_std", xlabel="training stddev")
#
# std_list, acc_list, quantile_list = get_acc_analysis(
#     "custom_training/noise_analysis2/d2d_0.3.csv", 'noisy training for d2d std 0.3', quantiles=1, x="training_std", xlabel="training stddev")

# std_list, acc_list, quantile_list = get_acc_analysis(
#     "custom_training/noise_analysis2/read_std_0.1.csv", 'noisy training for read std 0.1', quantiles=1, x="training_std", xlabel="training stddev")
#
# std_list, acc_list, quantile_list = get_acc_analysis(
#     "custom_training/noise_analysis2/read_std_0.2.csv", 'noisy training for read std 0.2', quantiles=1, x="training_std", xlabel="training stddev")
#
# std_list, acc_list, quantile_list = get_acc_analysis(
#     "custom_training/noise_analysis2/read_std_0.3.csv", 'noisy training for read std 0.3', quantiles=1, x="training_std", xlabel="training stddev")


# std_list, acc_list, quantile_list = get_acc_analysis(
#     "custom_training/nonlinearity_analysis2/2c_0.7.csv", 'noisy training for read std 0.1', quantiles=1, x="training_constant", xlabel="training stddev")
# std_list, acc_list, quantile_list = get_acc_analysis(
#     "custom_training/nonlinearity_analysis2/2c_0.8.csv", 'noisy training for read std 0.1', quantiles=1, x="training_constant", xlabel="training stddev")
# std_list, acc_list, quantile_list = get_acc_analysis(
#     "custom_training/nonlinearity_analysis2/2c_0.9.csv", 'noisy training for read std 0.1', quantiles=1, x="training_constant", xlabel="training stddev")
# std_list, acc_list, quantile_list = get_acc_analysis(
#     "custom_training/nonlinearity_analysis2/2c_1.0.csv", 'noisy training for read std 0.1', quantiles=1, x="training_constant", xlabel="training stddev")
# std_list, acc_list, quantile_list = get_acc_analysis(
#     "custom_training/nonlinearity_analysis2/2c_1.1.csv", 'noisy training for read std 0.1', quantiles=1, x="training_constant", xlabel="training stddev")

# std_list, acc_list, quantile_list = get_acc_analysis(
#     "custom_training/noisy_w_analysis/std_0.11.csv", 'noisy training for read std 0.1', quantiles=1, x="training_std", xlabel="training stddev")


def get_nonideality_analysis():
    std_list, acc_list, quantile_list = get_acc_analysis(
        "asd/nonideality_analysis/read_noise/ideal_analysis.csv", 'noisy training for read std 0.1', quantiles=1, x="read_noise_stddev", xlabel="read noise stddev factor")

    std_list, acc_list, quantile_list = get_acc_analysis(
        "asd/nonideality_analysis/d2d/ideal_analysis.csv", 'noisy training for read std 0.1', quantiles=1, x="d2d_stddev", xlabel="device-to-device stddev factor")

    std_list, acc_list, quantile_list = get_acc_analysis(
        "asd/nonideality_analysis/nonlinearity/ideal_analysis.csv", 'noisy training for read std 0.1', quantiles=1, x="nonlinearity", xlabel="nonlinearity constant")

# get_acc_analysis2(
#     ["asd/nonideality_analysis/d2d/input_noise/stddev_0.1.csv"],
#     x="input_stddev", xlabel="input noise stddev")


#get_nonideality_analysis()



#
# get_acc_analysis2(
#     ["asd/nonideality_analysis/d2d/input_noise/stddev_0.1.csv",
#      "asd/nonideality_analysis/d2d/input_noise/stddev_0.15.csv",
#      "asd/nonideality_analysis/d2d/input_noise/stddev_0.2.csv"],
#     x="input_stddev", xlabel="input noise stddev",
#     legend=["low d2d", "medium d2d", "high d2d"])
#
# get_acc_analysis2(
#     ["asd/nonideality_analysis/read_noise/input_noise/stddev_0.1.csv",
#      "asd/nonideality_analysis/read_noise/input_noise/stddev_0.15.csv",
#      "asd/nonideality_analysis/read_noise/input_noise/stddev_0.2.csv"],
#     x="input_stddev", xlabel="input noise stddev",
#     legend=["low read noise", "medium read noise", "high read noise"])
#
#
# get_acc_analysis2(
#     ["asd/nonideality_analysis/d2d/weights_noise/stddev_0.1.csv",
#      "asd/nonideality_analysis/d2d/weights_noise/stddev_0.15.csv",
#      "asd/nonideality_analysis/d2d/weights_noise/stddev_0.2.csv"],
#     x="weights_stddev", xlabel="weights noise stddev",
#     legend=["low d2d", "medium d2d", "high d2d"])
#
# get_acc_analysis2(
#     ["asd/nonideality_analysis/read_noise/weights_noise/stddev_0.1.csv",
#      "asd/nonideality_analysis/read_noise/weights_noise/stddev_0.15.csv",
#      "asd/nonideality_analysis/read_noise/weights_noise/stddev_0.2.csv"],
#     x="weights_stddev", xlabel="weights noise stddev",
#     legend=["low read noise", "medium read noise", "high read noise"])

#
# get_acc_analysis2(
#     ["asd/nonideality_analysis/nonlinearity/clean/c_0.7.csv",
# "asd/nonideality_analysis/nonlinearity/clean/c_0.9.csv",
# "asd/nonideality_analysis/nonlinearity/clean/c_1.0.csv",
# ],
#     x="nonlinearity", xlabel="training nonlinearity constant",
#     legend=["low nonlinearity", "medium nonlinearity", "high nonlinearity"])


# get_acc_analysis2(
#     ["asd/nonideality_analysis/d2d/input_noise/stddev_0.1.csv",
#      "asd/nonideality_analysis/d2d/input_noise/stddev_0.15.csv",
#      "asd/nonideality_analysis/d2d/input_noise/stddev_0.2.csv",
# "asd/nonideality_analysis/d2d/mixed_noise/stddev_0.1.csv",
# "asd/nonideality_analysis/d2d/mixed_noise/stddev_0.15.csv",
# "asd/nonideality_analysis/d2d/mixed_noise/stddev_0.2.csv"
#      ],
#     x="input_stddev", xlabel="input noise stddev",
#     legend=["low d2d input", "medium d2d input", "high d2d input", "low d2d mixed", "medium d2d mixed", "high d2d mixed"])
#
# get_acc_analysis2(
#     ["asd/nonideality_analysis/read_noise/input_noise/stddev_0.1.csv",
#      "asd/nonideality_analysis/read_noise/input_noise/stddev_0.15.csv",
#      "asd/nonideality_analysis/read_noise/input_noise/stddev_0.2.csv",
#      "asd/nonideality_analysis/read_noise/mixed_noise/stddev_0.1.csv",
#      "asd/nonideality_analysis/read_noise/mixed_noise/stddev_0.15.csv",
#      "asd/nonideality_analysis/read_noise/mixed_noise/stddev_0.2.csv"
#      ],
#     x="input_stddev", xlabel="input noise stddev",
#     legend=["low d2d input", "medium d2d input", "high d2d input", "low d2d mixed", "medium d2d mixed", "high d2d mixed"])

#
# get_acc_analysis2(
#     ["asd/nonideality_analysis/d2d/mixed_noise2/stddev_0.1.csv",
# "asd/nonideality_analysis/d2d/mixed_noise2/stddev_0.15.csv",
# "asd/nonideality_analysis/d2d/mixed_noise2/stddev_0.2.csv",
#      ],
#     x="weights_stddev", xlabel="weights noise stddev",
#     legend=["low d2d", "medium d2d", "high d2d"])
#
# get_acc_analysis2(
#     ["asd/nonideality_analysis/d2d/mixed_noise2/stddev_0.1.csv",
# "asd/nonideality_analysis/d2d/mixed_noise2/stddev_0.15.csv",
# "asd/nonideality_analysis/d2d/mixed_noise2/stddev_0.2.csv",
#      ],
#     x="input_stddev", xlabel="input noise stddev",
#     legend=["low d2d", "medium d2d", "high d2d"])
#
# get_acc_analysis2(
#     ["asd/nonideality_analysis/read_noise/mixed_noise2/stddev_0.1.csv",
# "asd/nonideality_analysis/read_noise/mixed_noise2/stddev_0.15.csv",
# "asd/nonideality_analysis/read_noise/mixed_noise2/stddev_0.2.csv",
#      ],
#     x="weights_stddev", xlabel="weights noise stddev",
#     legend=["low d2d", "medium d2d", "high d2d"])
#
# get_acc_analysis2(
#     ["asd/nonideality_analysis/read_noise/mixed_noise2/stddev_0.1.csv",
# "asd/nonideality_analysis/read_noise/mixed_noise2/stddev_0.15.csv",
# "asd/nonideality_analysis/read_noise/mixed_noise2/stddev_0.2.csv",
#      ],
#     x="input_stddev", xlabel="input noise stddev",
#     legend=["low d2d", "medium d2d", "high d2d"])


get_acc_analysis3(
    ["asd/nonideality_analysis/d2d/mixed_noise2/stddev_0.1.csv",
"asd/nonideality_analysis/d2d/mixed_noise2/stddev_0.15.csv",
"asd/nonideality_analysis/d2d/mixed_noise2/stddev_0.2.csv",
     ], n_test=9,

    x="weights_stddev", xlabel="input noise stddev",
    legend=["low d2d", "medium d2d", "high d2d"])

get_acc_analysis3(
    ["asd/nonideality_analysis/d2d/mixed_noise2/stddev_0.1.csv",
"asd/nonideality_analysis/d2d/mixed_noise2/stddev_0.15.csv",
"asd/nonideality_analysis/d2d/mixed_noise2/stddev_0.2.csv",
     ], n_test=9,

    x="input_stddev", xlabel="weights noise stddev",
    legend=["low d2d", "medium d2d", "high d2d"])
#
#
#
# get_acc_analysis3(
#     ["asd/nonideality_analysis/read_noise/mixed_noise2/stddev_0.1.csv",
# "asd/nonideality_analysis/read_noise/mixed_noise2/stddev_0.15.csv",
# "asd/nonideality_analysis/read_noise/mixed_noise2/stddev_0.2.csv",
#      ], n_test=9,
#
#     x="weights_stddev", xlabel="input noise stddev",
#     legend=["low d2d mixed", "medium d2d mixed", "high d2d mixed"])





# # L2 analysis
# get_acc_analysis2(
#     ["asd/nonideality_analysis/d2d/L2_input_noise/stddev_0.1.csv",
#      "asd/nonideality_analysis/d2d/L2_input_noise/reg_stddev_0.1.csv",],
#     x="input_stddev", xlabel="input noise stddev",
#     legend=["low d2d, no regularization", "low d2d, regularized"])
#
# get_acc_analysis2(
#     ["asd/nonideality_analysis/read_noise/L2_input_noise/stddev_0.1.csv",
#      "asd/nonideality_analysis/read_noise/L2_input_noise/reg_stddev_0.1.csv",],
#     x="input_stddev", xlabel="input noise stddev",
#     legend=["low read noise, no regularization", "low read noise, regularized"])
#
#
# get_acc_analysis2(
#     ["asd/nonideality_analysis/d2d/L2_weights_noise/stddev_0.1.csv",
#      "asd/nonideality_analysis/d2d/L2_weights_noise/reg_stddev_0.1.csv",],
#     x="weights_stddev", xlabel="input noise stddev",
#     legend=["low d2d, no regularization", "low d2d, regularized"])
#
# get_acc_analysis2(
#     ["asd/nonideality_analysis/read_noise/L2_weights_noise/stddev_0.1.csv",
#      "asd/nonideality_analysis/read_noise/L2_weights_noise/reg_stddev_0.1.csv",],
#     x="weights_stddev", xlabel="input noise stddev",
#     legend=["low read noise, no regularization", "low read noise, regularized"])
#
#
#
# get_acc_analysis2(
#     ["asd/nonideality_analysis/nonlinearity/L2/c_0.9.csv",
# "asd/nonideality_analysis/nonlinearity/L2/reg_c_0.9.csv",
# ],
#     x="nonlinearity", xlabel="training nonlinearity constant",
#     legend=["low nonlinearity, no regularization", "low nonlinearity, regularized"])






# dict = read_summary("asd/nonideality_analysis/all_low/ideal.csv")
# print(np.mean(np.asarray(dict["acc"])))
#
# dict = read_summary("asd/nonideality_analysis/all_low/aware.csv")
# print(np.mean(np.asarray(dict["acc"])))

# dict = read_summary("saved_models/naive/summary.csv")
# print(np.mean(np.asarray(dict["accuracy"])))

def plot_all_accuracies():
    # naive, input, weights, mixed
    # low, med, high
    ideal_acc = [0.9579]

    d2d_acc = [0.848, 0.597, 0.454,
               0.89, 0.815, 0.7325,
               0.904, 0.834, 0.737,
               0.904, 0.855, 0.774]

    read_acc = [0.842, 0.627, 0.476,
                0.892, 0.808, 0.71,
                0.907, 0.842, 0.76,
                0.905, 0.866, 0.78]

    nonlinearity_acc = [0.896, 0.648, 0.452,
                        0.937, 0.851, 0.711]

    mixed_acc = [0.539, 0.7]

    all_acc = np.concatenate((ideal_acc, d2d_acc, read_acc, nonlinearity_acc, mixed_acc), axis=None)


    labels = ["naive ideal",
              "naive low d2d", "mid d2d", "high d2d",
              "input low d2d", "mid d2d", "high d2d",
              "weights low d2d", "mid d2d", "high d2d",
              "mixed low d2d", "mid d2d", "high d2d",

              "naive low read", "mid read", "high read",
              "input low", "mid", "high",
              "weights low", "mid", "high",
              "mixed low", "mid", "high",

              "naive low nonlinearity", "mid", "high",
              "nonlineartty aware low", "mid", "high",

              "naive all nonidealities", "modified all nonidealities"
              ]

    labels.reverse()
    all_acc = np.flip(all_acc) * 100
    nr_labels = np.arange(0, len(labels), 1)

    plt.figure(figsize=(8, 6))
    plt.barh(nr_labels, all_acc)
    plt.yticks(nr_labels, labels)
    plt.xlabel("Accuracy")
    plt.grid(visible=True)
    plt.show()


def plot_all_accuracies_2():
    ideal_acc = [0.9579]
    d2d_acc_2 = [0.848, 0.89, 0.904, 0.904, 0.597, 0.815, 0.834, 0.855, 0.454, 0.7325, 0.737, 0.774]
    read_acc_2 = [0.842, 0.892, 0.907, 0.905, 0.627, 0.808, 0.842, 0.866, 0.476, 0.71, 0.76, 0.78]
    nonlinearity_acc_2 = [0.896, 0.937, 0.648, 0.851, 0.452, 0.711]
    mixed_acc = [0.539, 0.7]
    nonideal_acc_2 = np.flip(np.concatenate((ideal_acc, d2d_acc_2, read_acc_2, nonlinearity_acc_2, mixed_acc), axis=None)) * 100

    labels_2 = ["ideal naive",
                "d2d low: naive", "input", "weights", "mixed",
                "d2d mid: naive", "input", "weights", "mixed",
                "d2d high: naive", "input", "weights", "mixed",

                "read low: naive", "input", "weights", "mixed",
                "read mid: naive", "input", "weights", "mixed",
                "read high: naive", "input", "weights", "mixed",

                "lin low: naive", "aware",
                "lin mid: naive", "aware",
                "lin high: naive", "aware",

                "all naive", "all aware"
                ]

    colors = ["tab:red",
              "tab:blue", "tab:blue", "tab:blue", "tab:blue",
              "tab:blue", "tab:blue", "tab:blue", "tab:blue",
              "tab:blue", "tab:blue", "tab:blue", "tab:blue",

              "tab:green", "tab:green", "tab:green", "tab:green",
              "tab:green", "tab:green", "tab:green", "tab:green",
              "tab:green", "tab:green", "tab:green", "tab:green",

              "tab:orange", "tab:orange",
              "tab:orange", "tab:orange",
              "tab:orange", "tab:orange",

              "tab:cyan", "tab:cyan"
              ]

    labels_2.reverse()
    colors.reverse()
    nr_labels_2 = np.arange(0, len(labels_2), 1)

    plt.figure(figsize=(8, 6))
    plt.barh(nr_labels_2, 100.0-nonideal_acc_2, color=colors)
    plt.yticks(nr_labels_2, labels_2)
    plt.xlabel("incorrect classification, %")
    plt.grid(visible=True, axis="x")
    plt.show()

# plot_all_accuracies_2()


