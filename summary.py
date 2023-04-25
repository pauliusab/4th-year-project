import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import math
import tensorflow as tf
import os

def read_history(dir):
    # read history csv file
    file = open(dir + "/history.csv", "r")
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

def plot_accuracy(dir, **kwargs):

    xmin = kwargs.get("xmin", None)
    xmax = kwargs.get("xmax", None)
    ymin = kwargs.get("ymin", None)
    ymax = kwargs.get("ymax", None)

    history_dict = read_history(dir)

    fig, ax = plt.subplots()
    ax.plot(history_dict["accuracy"])
    ax.plot(history_dict["val_accuracy"])
    ax.set_title('model accuracy')
    ax.set_ylabel('accuracy')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'val'], loc='upper left')
    ax.set_xlim(xmin=xmin, xmax=xmax)
    ax.set_ylim(ymin=ymin, ymax=ymax)
    plt.show()

def plot_loss(dir, **kwargs):

    xmin = kwargs.get("xmin", None)
    xmax = kwargs.get("xmax", None)
    ymin = kwargs.get("ymin", None)
    ymax = kwargs.get("ymax", None)

    history_dict = read_history(dir)

    fig, ax = plt.subplots()
    ax.plot(history_dict["loss"])
    ax.plot(history_dict["val_loss"])
    ax.set_title('model loss')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'val'], loc='upper left')

    ax.set_xlim(xmin=xmin, xmax=xmax)
    ax.set_ylim(ymin=ymin, ymax=ymax)

    plt.show()

def create_ideal_summary():
    n_models = 30

    dir = "saved_models_interim/ideal"
    # read history csv file

    all_eval = []
    rows = []

    for i in range(1, n_models+1):
        dir_model = dir + "/model_" + str(i)
        file = open(dir_model + "/evaluation.csv", "r")
        csvreader = csv.reader(file)
        header = []
        header = next(csvreader)


        for row in csvreader:
            rows.append(row)

        all_eval.append(rows)
        file.close()

    print(all_eval)

    # extract required data
    history_dict = {}
    collumn = []

    for i in range(len(header)):
        for x in range(len(rows)):
            collumn.append(float(rows[x][i]))  # need to add float(), otherwise reads as string
        history_dict.update({header[i]: collumn})
        collumn = []

    print(history_dict)




    dir = "saved_models_interim/ideal"

    history = history_dict


    history.update({"model": list(range(1, n_models+1))})

    key_order = ["model", "loss", "accuracy"]
    ordered_hist = {k: history[k] for k in key_order}
    history = ordered_hist

    header = key_order

    print(history)

    row = []
    rows = []
    for i in range(len(history["loss"])):
        for a in history.keys():
            row.append(history[a][i])
        rows.append(row)
        row = []


    file = open(dir + "/summary.csv", "w")
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(rows)
    file.close



#create_ideal_summary()

def create_abs_dist_summary():
    n_std = 11
    n_dist = 10
    n_models = 30

    std_list = [0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 1, 2]        # for abs_disturb

    dir = "saved_models/abs_disturb_"

    full_summary = {}

    for i in range(len(std_list)):
        for a in range(n_models):
            for b in range(n_dist):
                try:
                    full_summary["std"].extend([std_list[i]])
                except KeyError:
                    full_summary["std"] = [std_list[i]]

    for u in range(1, n_std+1):

        dir_t = dir + str(u)

        dist_summary = {}

        for i in range(1, n_models+1):
            for a in range(n_dist):
                try:
                    dist_summary["model"].extend([i])
                except KeyError:
                    dist_summary["model"] = [i]

        for x in range(1, n_models+1):

            dir_model = dir_t + "/model_" + str(x)
            # read history csv file

            all_eval = []
            rows = []

            for i in range(1, n_dist + 1):
                dir_this = dir_model + "/" + str(i)
                file = open(dir_this + "/evaluation.csv", "r")
                csvreader = csv.reader(file)
                header = []
                header = next(csvreader)

                for row in csvreader:
                    rows.append(row)

                all_eval.append(rows)
                file.close()

            print(all_eval)

            # extract required data
            history_dict = {}
            collumn = []

            for i in range(len(header)):
                for x in range(len(rows)):
                    collumn.append(float(rows[x][i]))  # need to add float(), otherwise reads as string
                history_dict.update({header[i]: collumn})
                collumn = []

            print("history_dict " + str(x) + ":")
            print(history_dict)


            history = history_dict
            history.update({"n_dist": list(range(1, n_dist + 1))})

            key_order = ["n_dist", "loss", "accuracy"]
            ordered_hist = {k: history[k] for k in key_order}
            history = ordered_hist

            header = key_order


            for i in range(len(history.keys())):
                try:
                    dist_summary[list(history.keys())[i]].extend(history[list(history.keys())[i]])
                except KeyError:
                    dist_summary[list(history.keys())[i]] = history[list(history.keys())[i]]


            row = []
            rows = []
            for i in range(len(history["loss"])):
                for a in history.keys():
                    row.append(history[a][i])
                rows.append(row)
                row = []

            file = open(dir_model + "/summary.csv", "w")
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)
            file.close

        print("dist_summary " + str(u) + ":")
        print(dist_summary)

        row = []
        rows = []
        for i in range(len(dist_summary["loss"])):
            for a in dist_summary.keys():
                row.append(dist_summary[a][i])
            rows.append(row)
            row = []

        header = dist_summary.keys()

        file = open(dir_t + "/summary.csv", "w")
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)
        file.close

        for i in range(len(dist_summary.keys())):
            try:
                full_summary[list(dist_summary.keys())[i]].extend(dist_summary[list(dist_summary.keys())[i]])
            except KeyError:
                full_summary[list(dist_summary.keys())[i]] = dist_summary[list(dist_summary.keys())[i]]

    print("full_summary:")
    print(full_summary)

    row = []
    rows = []
    for i in range(len(full_summary["loss"])):
        for a in full_summary.keys():
            row.append(full_summary[a][i])
        rows.append(row)
        row = []

    header = full_summary.keys()
    file = open("saved_models/abs_disturb_summary.csv", "w")
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(rows)
    file.close

#create_abs_dist_summary()

def create_rel_dist_summary():
    n_std = 9
    n_dist = 10
    n_models = 30

    std_list = [0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5]        # for abs_disturb

    dir = "saved_models/rel_disturb_"

    full_summary = {}

    for i in range(len(std_list)):
        for a in range(n_models):
            for b in range(n_dist):
                try:
                    full_summary["std"].extend([std_list[i]])
                except KeyError:
                    full_summary["std"] = [std_list[i]]

    for u in range(1, n_std+1):

        dir_t = dir + str(u)

        dist_summary = {}

        for i in range(1, n_models+1):
            for a in range(n_dist):
                try:
                    dist_summary["model"].extend([i])
                except KeyError:
                    dist_summary["model"] = [i]

        for x in range(1, n_models+1):

            dir_model = dir_t + "/model_" + str(x)
            # read history csv file

            all_eval = []
            rows = []

            for i in range(1, n_dist + 1):
                dir_this = dir_model + "/" + str(i)
                file = open(dir_this + "/evaluation.csv", "r")
                csvreader = csv.reader(file)
                header = []
                header = next(csvreader)

                for row in csvreader:
                    rows.append(row)

                all_eval.append(rows)
                file.close()

            print(all_eval)

            # extract required data
            history_dict = {}
            collumn = []

            for i in range(len(header)):
                for x in range(len(rows)):
                    collumn.append(float(rows[x][i]))  # need to add float(), otherwise reads as string
                history_dict.update({header[i]: collumn})
                collumn = []

            print("history_dict " + str(x) + ":")
            print(history_dict)


            history = history_dict
            history.update({"n_dist": list(range(1, n_dist + 1))})

            key_order = ["n_dist", "loss", "accuracy"]
            ordered_hist = {k: history[k] for k in key_order}
            history = ordered_hist

            header = key_order


            for i in range(len(history.keys())):
                try:
                    dist_summary[list(history.keys())[i]].extend(history[list(history.keys())[i]])
                except KeyError:
                    dist_summary[list(history.keys())[i]] = history[list(history.keys())[i]]


            row = []
            rows = []
            for i in range(len(history["loss"])):
                for a in history.keys():
                    row.append(history[a][i])
                rows.append(row)
                row = []

            file = open(dir_model + "/summary.csv", "w")
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)
            file.close

        print("dist_summary " + str(u) + ":")
        print(dist_summary)

        row = []
        rows = []
        for i in range(len(dist_summary["loss"])):
            for a in dist_summary.keys():
                row.append(dist_summary[a][i])
            rows.append(row)
            row = []

        header = dist_summary.keys()

        file = open(dir_t + "/summary.csv", "w")
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)
        file.close

        for i in range(len(dist_summary.keys())):
            try:
                full_summary[list(dist_summary.keys())[i]].extend(dist_summary[list(dist_summary.keys())[i]])
            except KeyError:
                full_summary[list(dist_summary.keys())[i]] = dist_summary[list(dist_summary.keys())[i]]

    print("full_summary:")
    print(full_summary)

    row = []
    rows = []
    for i in range(len(full_summary["loss"])):
        for a in full_summary.keys():
            row.append(full_summary[a][i])
        rows.append(row)
        row = []

    header = full_summary.keys()
    file = open("saved_models/rel_disturb_summary.csv", "w")
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(rows)
    file.close

#create_rel_dist_summary()


def create_abs_dist_summary_g_ext():
    n_std = 11
    n_dist = 10
    n_models = 30

    std_list = [0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 1, 2]        # for abs_disturb

    dir = "saved_models/abs_disturb_higher_g_range_"

    full_summary = {}

    for i in range(len(std_list)):
        for a in range(n_models):
            for b in range(n_dist):
                try:
                    full_summary["std"].extend([std_list[i]])
                except KeyError:
                    full_summary["std"] = [std_list[i]]

    for u in range(1, n_std+1):

        dir_t = dir + str(u)

        dist_summary = {}

        for i in range(1, n_models+1):
            for a in range(n_dist):
                try:
                    dist_summary["model"].extend([i])
                except KeyError:
                    dist_summary["model"] = [i]

        for x in range(1, n_models+1):

            dir_model = dir_t + "/model_" + str(x)
            # read history csv file

            all_eval = []
            rows = []

            for i in range(1, n_dist + 1):
                dir_this = dir_model + "/" + str(i)
                file = open(dir_this + "/evaluation.csv", "r")
                csvreader = csv.reader(file)
                header = []
                header = next(csvreader)

                for row in csvreader:
                    rows.append(row)

                all_eval.append(rows)
                file.close()

            print(all_eval)

            # extract required data
            history_dict = {}
            collumn = []

            for i in range(len(header)):
                for x in range(len(rows)):
                    collumn.append(float(rows[x][i]))  # need to add float(), otherwise reads as string
                history_dict.update({header[i]: collumn})
                collumn = []

            print("history_dict " + str(x) + ":")
            print(history_dict)


            history = history_dict
            history.update({"n_dist": list(range(1, n_dist + 1))})

            key_order = ["n_dist", "loss", "accuracy"]
            ordered_hist = {k: history[k] for k in key_order}
            history = ordered_hist

            header = key_order


            for i in range(len(history.keys())):
                try:
                    dist_summary[list(history.keys())[i]].extend(history[list(history.keys())[i]])
                except KeyError:
                    dist_summary[list(history.keys())[i]] = history[list(history.keys())[i]]


            row = []
            rows = []
            for i in range(len(history["loss"])):
                for a in history.keys():
                    row.append(history[a][i])
                rows.append(row)
                row = []

            file = open(dir_model + "/summary.csv", "w")
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)
            file.close

        print("dist_summary " + str(u) + ":")
        print(dist_summary)

        row = []
        rows = []
        for i in range(len(dist_summary["loss"])):
            for a in dist_summary.keys():
                row.append(dist_summary[a][i])
            rows.append(row)
            row = []

        header = dist_summary.keys()

        file = open(dir_t + "/summary.csv", "w")
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)
        file.close

        for i in range(len(dist_summary.keys())):
            try:
                full_summary[list(dist_summary.keys())[i]].extend(dist_summary[list(dist_summary.keys())[i]])
            except KeyError:
                full_summary[list(dist_summary.keys())[i]] = dist_summary[list(dist_summary.keys())[i]]

    print("full_summary:")
    print(full_summary)

    row = []
    rows = []
    for i in range(len(full_summary["loss"])):
        for a in full_summary.keys():
            row.append(full_summary[a][i])
        rows.append(row)
        row = []

    header = full_summary.keys()
    file = open("saved_models/abs_disturb_ext_g_summary.csv", "w")
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(rows)
    file.close

#create_abs_dist_summary_g_ext()

def create_rel_dist_summary_g_ext():
    n_std = 9
    n_dist = 10
    n_models = 30

    std_list = [0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5]        # for abs_disturb

    dir = "saved_models/rel_disturb_higher_g_range_"

    full_summary = {}

    for i in range(len(std_list)):
        for a in range(n_models):
            for b in range(n_dist):
                try:
                    full_summary["std"].extend([std_list[i]])
                except KeyError:
                    full_summary["std"] = [std_list[i]]

    for u in range(1, n_std+1):

        dir_t = dir + str(u)

        dist_summary = {}

        for i in range(1, n_models+1):
            for a in range(n_dist):
                try:
                    dist_summary["model"].extend([i])
                except KeyError:
                    dist_summary["model"] = [i]

        for x in range(1, n_models+1):

            dir_model = dir_t + "/model_" + str(x)
            # read history csv file

            all_eval = []
            rows = []

            for i in range(1, n_dist + 1):
                dir_this = dir_model + "/" + str(i)
                file = open(dir_this + "/evaluation.csv", "r")
                csvreader = csv.reader(file)
                header = []
                header = next(csvreader)

                for row in csvreader:
                    rows.append(row)

                all_eval.append(rows)
                file.close()

            print(all_eval)

            # extract required data
            history_dict = {}
            collumn = []

            for i in range(len(header)):
                for x in range(len(rows)):
                    collumn.append(float(rows[x][i]))  # need to add float(), otherwise reads as string
                history_dict.update({header[i]: collumn})
                collumn = []

            print("history_dict " + str(x) + ":")
            print(history_dict)


            history = history_dict
            history.update({"n_dist": list(range(1, n_dist + 1))})

            key_order = ["n_dist", "loss", "accuracy"]
            ordered_hist = {k: history[k] for k in key_order}
            history = ordered_hist

            header = key_order


            for i in range(len(history.keys())):
                try:
                    dist_summary[list(history.keys())[i]].extend(history[list(history.keys())[i]])
                except KeyError:
                    dist_summary[list(history.keys())[i]] = history[list(history.keys())[i]]


            row = []
            rows = []
            for i in range(len(history["loss"])):
                for a in history.keys():
                    row.append(history[a][i])
                rows.append(row)
                row = []

            file = open(dir_model + "/summary.csv", "w")
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)
            file.close

        print("dist_summary " + str(u) + ":")
        print(dist_summary)

        row = []
        rows = []
        for i in range(len(dist_summary["loss"])):
            for a in dist_summary.keys():
                row.append(dist_summary[a][i])
            rows.append(row)
            row = []

        header = dist_summary.keys()

        file = open(dir_t + "/summary.csv", "w")
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)
        file.close

        for i in range(len(dist_summary.keys())):
            try:
                full_summary[list(dist_summary.keys())[i]].extend(dist_summary[list(dist_summary.keys())[i]])
            except KeyError:
                full_summary[list(dist_summary.keys())[i]] = dist_summary[list(dist_summary.keys())[i]]

    print("full_summary:")
    print(full_summary)

    row = []
    rows = []
    for i in range(len(full_summary["loss"])):
        for a in full_summary.keys():
            row.append(full_summary[a][i])
        rows.append(row)
        row = []

    header = full_summary.keys()
    file = open("saved_models/rel_disturb_ext_g_summary.csv", "w")
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(rows)
    file.close

#create_rel_dist_summary_g_ext()

def create_L1_summary(alphas):


    dir = "saved_models/L1"
    # read history csv file

    all_eval = []
    rows = []

    for i in alphas:
        dir_model = dir + "/alpha_" + str(i)
        try:
            file = open(dir_model + "/evaluation.csv", "r")
            csvreader = csv.reader(file)
            header = []
            header = next(csvreader)

            for row in csvreader:
                rows.append(row)

            all_eval.append(rows)
            file.close()
        except NotADirectoryError:
            print("asd")

    print(all_eval)

    # extract required data
    history_dict = {}
    collumn = []

    for i in range(len(header)):
        for x in range(len(rows)):
            collumn.append(float(rows[x][i]))  # need to add float(), otherwise reads as string
        history_dict.update({header[i]: collumn})
        collumn = []

    print(history_dict)

    dir = "saved_models/L1"

    history = history_dict

    history.update({"alpha": list(alphas)})

    key_order = ["alpha", "loss", "accuracy"]
    ordered_hist = {k: history[k] for k in key_order}
    history = ordered_hist

    header = key_order

    print(history)

    row = []
    rows = []
    for i in range(len(history["loss"])):
        for a in history.keys():
                row.append(history[a][i])
        rows.append(row)
        row = []


    file = open(dir + "/summary.csv", "w")
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(rows)
    file.close


def create_noise_training_summary(stds, n_models):


    dir = "custom_training/noise2"
    # read history csv file

    all_eval = []
    rows = []

    for n in range(1, n_models+1):
        for i in stds:
            dir_model = dir + "/stddev_" + str(i) + "_n_" + str(n)
            try:
                file = open(dir_model + "/evaluation.csv", "r")
                csvreader = csv.reader(file)
                header = []
                header = next(csvreader)

                for row in csvreader:
                    rows.append(row)

                all_eval.append(rows)
                file.close()
            except NotADirectoryError:
                print("asd")

    print(all_eval)

    # extract required data
    history_dict = {}
    collumn = []



    for i in range(len(header)):
        for x in range(len(rows)):
            collumn.append(float(rows[x][i]))  # need to add float(), otherwise reads as string
        history_dict.update({header[i]: collumn})
        collumn = []

    print(history_dict)

    dir = "custom_training/noise2"

    history = history_dict

    stds = np.repeat(stds, n_models, axis=0).flatten()
    history.update({"training_stddev": list(stds)})

    key_order = ["training_stddev", "loss", "accuracy"]
    ordered_hist = {k: history[k] for k in key_order}
    history = ordered_hist

    header = key_order

    print(history)

    row = []
    rows = []
    for i in range(len(history["loss"])):
        for a in history.keys():
                row.append(history[a][i])
        rows.append(row)
        row = []


    file = open(dir + "/summary.csv", "w")
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(rows)
    file.close

