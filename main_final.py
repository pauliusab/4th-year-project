import csv
import os
import mnist
import tensorflow as tf
import crossbar_final as crossbar
import numpy as np
import custom_training_final as training


def create_dirs():
    try:
        os.mkdir("saved_models")
    except FileExistsError:
        print("dir exists")


def create_training_model(**kwargs):

    input_stddev = kwargs.pop("input_stddev", None)

    if (input_stddev != None):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28), name="input"),
            tf.keras.layers.GaussianNoise(input_stddev),
            training.Training_Layer(784, 36, activation='sigmoid', name="hidden", **kwargs),
            tf.keras.layers.GaussianNoise(input_stddev),
            training.Training_Layer(36, 10, activation='softmax', name="output", **kwargs),
        ])

    else:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28), name="input"),
            training.Training_Layer(784, 36, activation='sigmoid', name="hidden", **kwargs),
            training.Training_Layer(36, 10, activation='softmax', name="output", **kwargs),
        ])

    model.compile(optimizer='SGD',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def train_full(model, ds_train, ds_val, patience, dir):
    x_train, y_train = ds_train

    callback_stop = tf.keras.callbacks.EarlyStopping(  # stops training whenever validation loss starts increasing
        monitor='val_loss',
        min_delta=0,
        patience=patience,
        verbose=0,
        mode='min',
        restore_best_weights=True,
    )

    csv_logger = tf.keras.callbacks.CSVLogger(
        dir + "/history.csv", separator=',', append=False
    )

    try:
        os.mkdir(dir)
    except FileExistsError:
        print("dir exists")

    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=500,
        validation_data=ds_val,
        # validation_freq=10,
        callbacks=[callback_stop, csv_logger]
    )

    return history

def save_model(model, history, dir):
    model.save(dir)
    history = history.history

    row = []
    rows = []
    for i in range(len(history["loss"])):
        row.append(i + 1)
        for a in history.keys():
            row.append(history[a][i])
        rows.append(row)
        row = []

    header = list(history.keys())
    header.insert(0, "epoch")

    file = open(dir + "/history.csv", "w")
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(rows)
    file.close

    print("model saved!")

def save_eval(eval, dir):
    file = open(dir + "/evaluation.csv", "w")
    writer = csv.writer(file)
    writer.writerow(["loss", "accuracy"])
    writer.writerow(eval)
    file.close



ds_train, ds_val, ds_test = mnist.get_ds()



def make_models(dir, n, **kwargs):

    # possible kwargs:
    # ("input_stddev", None)
    # ("nonlinearity", None)
    # ("weight_stddev", None)
    # ("alpha", None)

    patience = kwargs.pop("patience", 1)

    for i in range(1, n+1):
        print("n: " + str(i))
        dir_full = dir + "/n_" + str(i)
        model = create_training_model(**kwargs)
        history = train_full(model, ds_train, ds_val, patience=patience, dir=dir_full)       # returns dict with {loss:[], accuracy:[], ...}
        eval = model.evaluate(ds_test[0], ds_test[1], verbose=0)    # returns list of [loss, accuracy]
        save_eval(eval, dir_full)
        model.save(dir_full)


def nonideality_analysis(model_dir, **kwargs):

    # possible kwargs:
    # ("k_g", None)
    # ("g_min", 0.001)
    # ("g_max", 0.002)
    #
    # ("nonlinearity", None)
    # ("read_noise_std", None)
    # ("d2d_std", None)
    # ("training_with_input_noise", False)

    model = tf.keras.models.load_model(model_dir)
    custom_model = crossbar.Memristive_Model(model, "1", **kwargs)
    custom_model.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    eval = custom_model.evaluate(ds_test[0], ds_test[1], verbose=1)

    return eval


def save_analysis(dir, parameters, evals):
    first_column = np.reshape(parameters[1], (len(parameters[1]),1))
    print(first_column)
    rows = np.concatenate((first_column, np.array(evals)), axis=1)
    header = [parameters[0], "loss", "acc"]

    file = open(dir, "w")
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(rows)
    file.close

def save_analysis2(dir, parameters1, parameters2, evals):
    first_column = np.reshape(parameters1[1], (len(parameters1[1]),1))
    print(first_column)
    second_column = np.reshape(parameters2[1], (len(parameters2[1]),1))
    print(second_column)
    rows = np.concatenate((first_column, second_column, np.array(evals)), axis=1)
    header = [parameters1[0], parameters2[0], "loss", "acc"]

    file = open(dir, "w")
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(rows)
    file.close

def save_analysis3(dir, evals):
    header = ["loss", "acc"]

    file = open(dir, "w")
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(evals)
    file.close

def input_std_analysis(train_models, min_std, max_std, n_test, n_models):

    if(train_models==True):
        for std in range(min_std ,max_std+1, 2):
            this_std = std / 100.0
            model_dir = "asd/saved_models/input_noise/stddev_" + str(this_std)

            try:
                os.mkdir(model_dir)
            except FileExistsError:
                print("dir exists")


            make_models(model_dir,
                    n_models,
                    patience=1,
                    input_stddev=this_std,
                    # nonlinearity=0.8,
                    # weight_stddev=0.2,
                    # l2_alpha=0.001,
                    )

    d2d_list = [0.1, 0.15, 0.2]


    for d2d in d2d_list:

        evals = []
        parameters = []

        for std in range(min_std, max_std+1, 2):
            this_std = std / 100.0
            model_dir = "asd/saved_models/input_noise/stddev_" + str(this_std)

            for i in range(1, n_models+1):
                this_dir = model_dir + "/n_" + str(i)
                for x in range(1, n_test + 1):
                    eval = nonideality_analysis(this_dir,
                                                # g_min=0.001,
                                                # g_max=0.002,
                                                # nonlinearity=0.8,
                                                # read_noise_std=0.2,
                                                d2d_std=d2d,
                                                training_with_input_noise=True
                                                )
                    parameters.append(this_std)
                    evals.append(eval)

        print(parameters)
        analysis_dir = "asd/nonideality_analysis/d2d/input_noise/2stddev_" + str(d2d) + ".csv"
        save_analysis(analysis_dir, ("input_stddev", parameters), evals)

    read_noise_list = [0.1, 0.15, 0.2]


    for read_noise in read_noise_list:

        evals = []
        parameters = []

        for std in range(min_std, max_std + 1, 2):
            this_std = std / 100.0
            model_dir = "asd/saved_models/input_noise/stddev_" + str(this_std)

            for i in range(1, n_models + 1):
                this_dir = model_dir + "/n_" + str(i)
                for x in range(1, n_test + 1):
                    eval = nonideality_analysis(this_dir,
                                                # g_min=0.001,
                                                # g_max=0.002,
                                                # nonlinearity=0.8,
                                                read_noise_std=read_noise,
                                                # d2d_std=0.2,
                                                training_with_input_noise=True
                                                )
                    parameters.append(this_std)
                    evals.append(eval)

        print(parameters)
        analysis_dir = "asd/nonideality_analysis/read_noise/input_noise/2stddev_" + str(read_noise) + ".csv"
        save_analysis(analysis_dir, ("input_stddev", parameters), evals)

def weights_std_analysis(train_models, min_std, max_std, step, n_test, n_models):


    if(train_models == True):
        for std in range(min_std, max_std+1, step):
            this_std = std / 100.0
            print(this_std)
            model_dir = "asd/saved_models/weights_noise/stddev_" + str(this_std)

            try:
                os.mkdir(model_dir)
            except FileExistsError:
                print("dir exists")


            make_models(model_dir,
                    n_models,
                    patience=1,
                    # input_stddev=this_std,
                    # nonlinearity=0.8,
                    weight_stddev=this_std,
                    # l2_alpha=0.001,
                    )


    d2d_list = [0.1, 0.15, 0.2]


    for d2d in d2d_list:
        evals = []
        parameters = []

        for std in range(min_std, max_std + 1, step):
            this_std = std / 100.0
            model_dir = "asd/saved_models/weights_noise/stddev_" + str(this_std)


            for i in range(1, n_models + 1):
                this_dir = model_dir + "/n_" + str(i)
                for x in range(1, n_test + 1):
                    eval = nonideality_analysis(this_dir,
                                                # g_min=0.001,
                                                # g_max=0.002,
                                                # nonlinearity=0.8,
                                                # read_noise_std=0.2,
                                                d2d_std=d2d,
                                                # training_with_input_noise=True
                                                )
                    parameters.append(this_std)
                    evals.append(eval)

        print(parameters)
        analysis_dir = "asd/nonideality_analysis/d2d/weights_noise/3stddev_" + str(d2d) + ".csv"
        save_analysis(analysis_dir, ("weights_stddev", parameters), evals)



    read_noise_list = [0.1, 0.15, 0.2]

    for read_noise in read_noise_list:
        evals = []
        parameters = []

        for std in range(min_std, max_std + 1, step):
            this_std = std / 100.0
            model_dir = "asd/saved_models/weights_noise/stddev_" + str(this_std)

            for i in range(1, n_models + 1):
                this_dir = model_dir + "/n_" + str(i)
                for x in range(1, n_test + 1):
                    eval = nonideality_analysis(this_dir,
                                                # g_min=0.001,
                                                # g_max=0.002,
                                                # nonlinearity=0.8,
                                                read_noise_std=read_noise,
                                                # d2d_std=0.2,
                                                # training_with_input_noise=True
                                                )
                    parameters.append(this_std)
                    evals.append(eval)

        print(parameters)
        analysis_dir = "asd/nonideality_analysis/read_noise/weights_noise/3stddev_" + str(read_noise) + ".csv"
        save_analysis(analysis_dir, ("weights_stddev", parameters), evals)

def nonlinearity_training_analysis(train_models, max_nonlinearity, min_nonlinearity, step, n_test, n_models):

    if(train_models == True):
        for c in range(26, max_nonlinearity+1, step):
            this_c = c / 100.0
            print(this_c)
            model_dir = "asd/saved_models/nonlinearity/2/c_" + str(this_c)

            try:
                os.mkdir(model_dir)
            except FileExistsError:
                print("dir exists")


            make_models(model_dir,
                    n_models,
                    patience=1,
                    # input_stddev=this_std,
                    nonlinearity=this_c,
                    # weight_stddev=this_c,
                    # l2_alpha=0.001,
                    )


    c_list = [0.7, 0.9, 1.0]


    for nonlinearity in c_list:
        evals = []
        parameters = []

        for c in range(min_nonlinearity, max_nonlinearity+1, step):
            this_c = c / 100.0
            model_dir = "asd/saved_models/nonlinearity/2/c_" + str(this_c)


            for i in range(1, n_models + 1):
                this_dir = model_dir + "/n_" + str(i)
                for x in range(1, n_test + 1):
                    eval = nonideality_analysis(this_dir,
                                                # g_min=0.001,
                                                # g_max=0.002,
                                                nonlinearity=nonlinearity,
                                                # read_noise_std=0.2,
                                                # d2d_std=d2d,
                                                # training_with_input_noise=True
                                                )
                    parameters.append(this_c)
                    evals.append(eval)

        print(parameters)
        analysis_dir = "asd/nonideality_analysis/nonlinearity/2/c_" + str(nonlinearity) + ".csv"
        save_analysis(analysis_dir, ("nonlinearity", parameters), evals)




def L2_input_std_analysis(train_models, min_std, max_std, n_test, n_models):

    if (train_models == True):
        for std in range(min_std, max_std + 1, 2):
            this_std = std / 100.0
            model_dir = "asd/saved_models/L2_input_noise/stddev_" + str(this_std)

            try:
                os.mkdir(model_dir)
            except FileExistsError:
                print("dir exists")

            make_models(model_dir,
                        n_models,
                        patience=1,
                        input_stddev=this_std,
                        # nonlinearity=0.8,
                        # weight_stddev=0.2,
                        l2_alpha=0.001,
                        )

    d2d_list = [0.1]

    for d2d in d2d_list:

        evals = []
        parameters = []

        for std in range(min_std, max_std + 1, 2):
            this_std = std / 100.0
            model_dir = "asd/saved_models/L2_input_noise/stddev_" + str(this_std)

            for i in range(1, n_models + 1):
                this_dir = model_dir + "/n_" + str(i)
                for x in range(1, n_test + 1):
                    eval = nonideality_analysis(this_dir,
                                                # g_min=0.001,
                                                # g_max=0.002,
                                                # nonlinearity=0.8,
                                                # read_noise_std=0.2,
                                                d2d_std=d2d,
                                                training_with_input_noise=True
                                                )
                    parameters.append(this_std)
                    evals.append(eval)

        print(parameters)
        analysis_dir = "asd/nonideality_analysis/d2d/L2_input_noise/reg_stddev_" + str(d2d) + ".csv"
        save_analysis(analysis_dir, ("input_stddev", parameters), evals)

    read_noise_list = [0.1]

    for read_noise in read_noise_list:

        evals = []
        parameters = []

        for std in range(min_std, max_std + 1, 2):
            this_std = std / 100.0
            model_dir = "asd/saved_models/L2_input_noise/stddev_" + str(this_std)

            for i in range(1, n_models + 1):
                this_dir = model_dir + "/n_" + str(i)
                for x in range(1, n_test + 1):
                    eval = nonideality_analysis(this_dir,
                                                # g_min=0.001,
                                                # g_max=0.002,
                                                # nonlinearity=0.8,
                                                read_noise_std=read_noise,
                                                # d2d_std=0.2,
                                                training_with_input_noise=True
                                                )
                    parameters.append(this_std)
                    evals.append(eval)

        print(parameters)
        analysis_dir = "asd/nonideality_analysis/read_noise/L2_input_noise/reg_stddev_" + str(read_noise) + ".csv"
        save_analysis(analysis_dir, ("input_stddev", parameters), evals)

def L2_weights_std_analysis(train_models, min_std, max_std, step, n_test, n_models):
    if (train_models == True):
        for std in range(min_std, max_std + 1, step):
            this_std = std / 100.0
            print(this_std)
            model_dir = "asd/saved_models/L2_weights_noise/stddev_" + str(this_std)

            try:
                os.mkdir(model_dir)
            except FileExistsError:
                print("dir exists")

            make_models(model_dir,
                        n_models,
                        patience=1,
                        # input_stddev=this_std,
                        # nonlinearity=0.8,
                        weight_stddev=this_std,
                        l2_alpha=0.001,
                        )

    d2d_list = [0.1]

    for d2d in d2d_list:
        evals = []
        parameters = []

        for std in range(min_std, max_std + 1, step):
            this_std = std / 100.0
            model_dir = "asd/saved_models/L2_weights_noise/stddev_" + str(this_std)

            for i in range(1, n_models + 1):
                this_dir = model_dir + "/n_" + str(i)
                for x in range(1, n_test + 1):
                    eval = nonideality_analysis(this_dir,
                                                # g_min=0.001,
                                                # g_max=0.002,
                                                # nonlinearity=0.8,
                                                # read_noise_std=0.2,
                                                d2d_std=d2d,
                                                # training_with_input_noise=True
                                                )
                    parameters.append(this_std)
                    evals.append(eval)

        print(parameters)
        analysis_dir = "asd/nonideality_analysis/d2d/L2_weights_noise/reg_stddev_" + str(d2d) + ".csv"
        save_analysis(analysis_dir, ("weights_stddev", parameters), evals)

    read_noise_list = [0.1]

    for read_noise in read_noise_list:
        evals = []
        parameters = []

        for std in range(min_std, max_std + 1, step):
            this_std = std / 100.0
            model_dir = "asd/saved_models/L2_weights_noise/stddev_" + str(this_std)

            for i in range(1, n_models + 1):
                this_dir = model_dir + "/n_" + str(i)
                for x in range(1, n_test + 1):
                    eval = nonideality_analysis(this_dir,
                                                # g_min=0.001,
                                                # g_max=0.002,
                                                # nonlinearity=0.8,
                                                read_noise_std=read_noise,
                                                # d2d_std=0.2,
                                                # training_with_input_noise=True
                                                )
                    parameters.append(this_std)
                    evals.append(eval)

        print(parameters)
        analysis_dir = "asd/nonideality_analysis/read_noise/L2_weights_noise/reg_stddev_" + str(read_noise) + ".csv"
        save_analysis(analysis_dir, ("weights_stddev", parameters), evals)

def L2_nonlinearity_training_analysis(train_models, max_nonlinearity, min_nonlinearity, step, n_test, n_models):
    if (train_models == True):
        for c in range(min_nonlinearity, max_nonlinearity + 1, step):
            this_c = c / 10.0
            print(this_c)
            model_dir = "asd/saved_models/L2_nonlinearity/c_" + str(this_c)

            try:
                os.mkdir(model_dir)
            except FileExistsError:
                print("dir exists")

            make_models(model_dir,
                        n_models,
                        patience=1,
                        # input_stddev=this_std,
                        nonlinearity=this_c,
                        # weight_stddev=this_c,
                        l2_alpha=0.001,
                        )

    c_list = [0.9]

    for nonlinearity in c_list:
        evals = []
        parameters = []

        for c in range(min_nonlinearity, max_nonlinearity + 1, step):
            this_c = c / 10.0
            model_dir = "asd/saved_models/L2_nonlinearity/c_" + str(this_c)

            for i in range(1, n_models + 1):
                this_dir = model_dir + "/n_" + str(i)
                for x in range(1, n_test + 1):
                    eval = nonideality_analysis(this_dir,
                                                # g_min=0.001,
                                                # g_max=0.002,
                                                nonlinearity=nonlinearity,
                                                # read_noise_std=0.2,
                                                # d2d_std=d2d,
                                                # training_with_input_noise=True
                                                )
                    parameters.append(this_c)
                    evals.append(eval)

        print(parameters)
        analysis_dir = "asd/nonideality_analysis/nonlinearity/L2/2/reg_c_" + str(nonlinearity) + ".csv"
        save_analysis(analysis_dir, ("nonlinearity", parameters), evals)




def mixed_training_analysis(train_models, min_std, max_std, n_test, n_models):

    if(train_models==True):
        for std in range(min_std ,max_std+1, 2):
            this_std = std / 100.0
            model_dir = "asd/saved_models/mixed_noise/w=0.7_in=_" + str(this_std)

            try:
                os.mkdir(model_dir)
            except FileExistsError:
                print("dir exists")


            make_models(model_dir,
                    n_models,
                    patience=1,
                    input_stddev=this_std,
                    # nonlinearity=0.8,
                    weight_stddev=0.7,
                    # l2_alpha=0.001,
                    )

    d2d_list = [0.1, 0.15, 0.2]


    for d2d in d2d_list:

        evals = []
        parameters = []

        for std in range(min_std, max_std+1, 2):
            this_std = std / 100.0
            model_dir = "asd/saved_models/mixed_noise/w=0.7_in=_" + str(this_std)

            for i in range(1, n_models+1):
                this_dir = model_dir + "/n_" + str(i)
                for x in range(1, n_test + 1):
                    eval = nonideality_analysis(this_dir,
                                                # g_min=0.001,
                                                # g_max=0.002,
                                                # nonlinearity=0.8,
                                                # read_noise_std=0.2,
                                                d2d_std=d2d,
                                                training_with_input_noise=True
                                                )
                    parameters.append(this_std)
                    evals.append(eval)

        print(parameters)
        analysis_dir = "asd/nonideality_analysis/d2d/mixed_noise/stddev_" + str(d2d) + ".csv"
        save_analysis(analysis_dir, ("input_stddev", parameters), evals)


    read_noise_list = [0.1, 0.15, 0.2]

    for read_noise in read_noise_list:

        evals = []
        parameters = []

        for std in range(min_std, max_std + 1, 2):
            this_std = std / 100.0
            model_dir = "asd/saved_models/mixed_noise/w=0.7_in=_" + str(this_std)

            for i in range(1, n_models + 1):
                this_dir = model_dir + "/n_" + str(i)
                for x in range(1, n_test + 1):
                    eval = nonideality_analysis(this_dir,
                                                # g_min=0.001,
                                                # g_max=0.002,
                                                # nonlinearity=0.8,
                                                read_noise_std=read_noise,
                                                # d2d_std=0.2,
                                                training_with_input_noise=True
                                                )
                    parameters.append(this_std)
                    evals.append(eval)

        print(parameters)
        analysis_dir = "asd/nonideality_analysis/read_noise/mixed_noise/stddev_" + str(read_noise) + ".csv"
        save_analysis(analysis_dir, ("input_stddev", parameters), evals)

def mixed_training_analysis2(train_models, n_test, n_models):

    if(train_models==True):
        for weights_std in range(75, 101, 5):
            this_w = weights_std / 100.0
            model_dir = "asd/saved_models/mixed_noise2/w_" + str(this_w)

            try:
                os.mkdir(model_dir)
            except FileExistsError:
                print("dir exists")

            for input_std in range(10, 51, 5):
                this_in = input_std / 100.0
                this_dir = model_dir + "/in_" + str(this_in)

                print(this_w, this_in)

                try:
                    os.mkdir(this_dir)
                except FileExistsError:
                    print("dir exists")

                print(this_dir)

                make_models(this_dir,
                        n_models,
                        patience=1,
                        input_stddev=this_in,
                        # nonlinearity=0.8,
                        weight_stddev=this_w,
                        # l2_alpha=0.001,
                        )

    d2d_list = [0.1, 0.15, 0.2]


    for d2d in d2d_list:

        evals = []
        parameters1 = []
        parameters2 = []

        for weights_std in range(10, 101, 5):
            this_w = weights_std / 100.0
            model_dir = "asd/saved_models/mixed_noise2/w_" + str(this_w)

            for input_std in range(10, 51, 5):
                this_in = input_std / 100.0
                this_dir = model_dir + "/in_" + str(this_in)

                for i in range(1, n_models+1):
                    this_n_dir = this_dir + "/n_" + str(i)

                    for x in range(1, n_test + 1):
                        eval = nonideality_analysis(this_n_dir,
                                                    # g_min=0.001,
                                                    # g_max=0.002,
                                                    # nonlinearity=0.8,
                                                    # read_noise_std=0.2,
                                                    d2d_std=d2d,
                                                    training_with_input_noise=True
                                                    )
                        parameters1.append(this_w)
                        parameters2.append(this_in)
                        evals.append(eval)

        analysis_dir = "asd/nonideality_analysis/d2d/mixed_noise2/stddev_" + str(d2d) + ".csv"
        save_analysis2(analysis_dir, ("weights_stddev", parameters1), ("input_stddev", parameters2), evals)


    read_noise_list = [0.1, 0.15, 0.2]

    for read_noise in read_noise_list:

        evals = []
        parameters1 = []
        parameters2 = []

        for weights_std in range(10, 101, 5):
            this_w = weights_std / 100.0
            model_dir = "asd/saved_models/mixed_noise2/w_" + str(this_w)

            for input_std in range(10, 51, 5):
                this_in = input_std / 100.0
                this_dir = model_dir + "/in_" + str(this_in)

                for i in range(1, n_models + 1):
                    this_n_dir = this_dir + "/n_" + str(i)

                    for x in range(1, n_test + 1):
                        eval = nonideality_analysis(this_n_dir,
                                                    # g_min=0.001,
                                                    # g_max=0.002,
                                                    # nonlinearity=0.8,
                                                    read_noise_std=read_noise,
                                                    # d2d_std=d2d,
                                                    training_with_input_noise=True
                                                    )
                        parameters1.append(this_w)
                        parameters2.append(this_in)
                        evals.append(eval)

        analysis_dir = "asd/nonideality_analysis/read_noise/mixed_noise2/stddev_" + str(read_noise) + ".csv"
        save_analysis2(analysis_dir, ("weights_stddev", parameters1), ("input_stddev", parameters2), evals)


def mixed_analysis(train_models, n_test, n_models):

    model_dir = "asd/saved_models/mixed_training"

    if(train_models==True):

        try:
            os.mkdir(model_dir)
        except FileExistsError:
            print("dir exists")

        print(model_dir)

        make_models(model_dir,
                n_models,
                patience=1,
                input_stddev=0.2,
                nonlinearity=0.4,
                weight_stddev=0.4,
                # l2_alpha=0.001,
                )



    evals = []
    for i in range(1, n_models+1):
        this_n_dir = model_dir + "/n_" + str(i)

        for x in range(1, n_test + 1):
            eval = nonideality_analysis(this_n_dir,
                                        # g_min=0.001,
                                        # g_max=0.002,
                                        nonlinearity=0.7,
                                        read_noise_std=0.1,
                                        d2d_std=0.1,
                                        training_with_input_noise=True
                                        )

            evals.append(eval)

    analysis_dir = "asd/nonideality_analysis/all_low/aware.csv"
    save_analysis3(analysis_dir, evals)




def all_nonideality_analysis():

    n_models = 15
    n_test = 3
    model_dir = "asd/saved_models/ideal"

    make_models(model_dir,
                    n_models,
                    patience=1,
                    # input_stddev=this_std,
                    # nonlinearity=0.8,
                    # weight_stddev=0.2,
                    # l2_alpha=0.001,
                    )

    n_models = 5

    analysis_dir = "asd/nonideality_analysis/read_noise/ideal_analysis.csv"
    parameter_name = "read_noise_stddev"
    std_list = np.arange(0, 40, 1) / 100.0

    read_noise_analysis(model_dir, analysis_dir, parameter_name, n_test, n_models, std_list)

    analysis_dir = "asd/nonideality_analysis/d2d/ideal_analysis.csv"
    parameter_name = "d2d_stddev"
    std_list = np.arange(0, 40, 1) / 100.0

    d2d_analysis(model_dir, analysis_dir, parameter_name, n_test, n_models, std_list)

    n_models = 15
    n_test = 1
    analysis_dir = "asd/nonideality_analysis/nonlinearity/ideal_analysis.csv"
    parameter_name = "nonlinearity"
    std_list = np.arange(0, 15, 1) / 10.0

    nonlinearity_analysis(model_dir, analysis_dir, parameter_name, n_test, n_models, std_list)


def read_noise_analysis(model_dir, analysis_dir, parameter_name, n_test, n_models, std_list):
    evals = []
    parameters = []

    for std in std_list:
        for i in range(1, n_models + 1):
            this_dir = model_dir + "/n_" + str(i)
            for x in range(1, n_test + 1):
                eval = nonideality_analysis(this_dir,
                                            # g_min=0.001,
                                            # g_max=0.002,
                                            # nonlinearity=0.8,
                                            read_noise_std=std,
                                            # d2d_std=0.2,
                                            # training_with_input_noise=True
                                            )
                parameters.append(std)
                evals.append(eval)

    save_analysis(analysis_dir, (parameter_name, parameters), evals)

def d2d_analysis(model_dir, analysis_dir, parameter_name, n_test, n_models, std_list):
    evals = []
    parameters = []

    for std in std_list:
        for i in range(1, n_models + 1):
            this_dir = model_dir + "/n_" + str(i)
            for x in range(1, n_test + 1):
                eval = nonideality_analysis(this_dir,
                                            # g_min=0.001,
                                            # g_max=0.002,
                                            # nonlinearity=0.8,
                                            # read_noise_std=0.2,
                                            d2d_std=std,
                                            # training_with_input_noise=True
                                            )
                parameters.append(std)
                evals.append(eval)

    save_analysis(analysis_dir, (parameter_name, parameters), evals)

def nonlinearity_analysis(model_dir, analysis_dir, parameter_name, n_test, n_models, std_list):
    evals = []
    parameters = []

    for std in std_list:
        for i in range(1, n_models + 1):
            this_dir = model_dir + "/n_" + str(i)
            for x in range(1, n_test + 1):
                eval = nonideality_analysis(this_dir,
                                            # g_min=0.001,
                                            # g_max=0.002,
                                            nonlinearity=std,
                                            # read_noise_std=std,
                                            # d2d_std=0.2,
                                            # training_with_input_noise=True
                                            )
                parameters.append(std)
                evals.append(eval)

    save_analysis(analysis_dir, (parameter_name, parameters), evals)

#all_nonideality_analysis()






# for std in range(9, 50 + 1, 2):
#     this_std = std / 100.0
#     model_dir = "asd/saved_models/input_noise/stddev_" + str(this_std)
#
#     try:
#         os.mkdir(model_dir)
#     except FileExistsError:
#         print("dir exists")
#
#
#     for i in range(4, 6):
#         print("n: " + str(i))
#         dir_full = model_dir + "/n_" + str(i)
#         model = create_training_model(input_stddev=this_std)
#         history = train_full(model, ds_train, ds_val, patience=1, dir=dir_full)       # returns dict with {loss:[], accuracy:[], ...}
#         eval = model.evaluate(ds_test[0], ds_test[1], verbose=0)    # returns list of [loss, accuracy]
#         save_eval(eval, dir_full)
#         model.save(dir_full)




# input_std_analysis(train_models=True,
#                    min_std=51,
#                    max_std=80,
#                    n_test=5,
#                    n_models=5
#                    )
#
#
nonlinearity_training_analysis(train_models=True,
                               max_nonlinearity=80,
                               min_nonlinearity=20,
                               step=2,
                               n_test=1,
                               n_models=10
                               )
#
#
# this_std = 0.51
# print(this_std)
# model_dir = "asd/saved_models/weights_noise/stddev_" + str(this_std)


#
# make_models(model_dir,
#         1,
#         patience=1,
#         # input_stddev=this_std,
#         # nonlinearity=0.8,
#         weight_stddev=this_std,
#         # l2_alpha=0.001,
#         )
#
#
#
#
# weights_std_analysis(train_models=False,
#                      min_std=51,
#                     max_std=80,
#                      step = 2,
#                    n_test=5,
#                    n_models=5
#                    )
#
#
# L2_input_std_analysis(train_models=True,
#                    min_std=11,
#                    max_std=40,
#                    n_test=5,
#                    n_models=5
#                    )
#
#
# L2_weights_std_analysis(train_models=True,
#                      min_std=61,
#                     max_std=80,
#                      step = 2,
#                    n_test=5,
#                    n_models=5
#                    )


# L2_nonlinearity_training_analysis(train_models=True,
#                                max_nonlinearity=12,
#                                min_nonlinearity=9,
#                                step=1,
#                                n_test=1,
#                                n_models=10
#                                )
#
# this_w = 70 / 100.0
# model_dir = "asd/saved_models/mixed_noise2/w_" + str(this_w)
#
#
# this_in = 10 / 100.0
# this_dir = model_dir + "/in_" + str(this_in)
#
# print(this_dir)
# print(this_w, this_in)
#
# make_models(this_dir,
#             1,
#             patience=1,
#             input_stddev=this_in,
#             # nonlinearity=0.8,
#             weight_stddev=this_w,
#             # l2_alpha=0.001,
#             )



# mixed_training_analysis2(train_models=True, n_test=3, n_models=3)



#mixed_analysis(train_models=False, n_test=10, n_models=5)

#
# model_dir ="asd/saved_models/ideal"
#
# evals = []
# for i in range(1, 5+1):
#     this_n_dir = model_dir + "/n_" + str(i)
#
#     for x in range(1, 10 + 1):
#         eval = nonideality_analysis(this_n_dir,
#                                     # g_min=0.001,
#                                     # g_max=0.002,
#                                     nonlinearity=0.7,
#                                     read_noise_std=0.1,
#                                     d2d_std=0.1,
#                                     training_with_input_noise=False
#                                     )
#
#         evals.append(eval)
#
# analysis_dir = "asd/nonideality_analysis/all_low/ideal.csv"
# save_analysis3(analysis_dir, evals)


