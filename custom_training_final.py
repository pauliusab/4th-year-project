import tensorflow as tf
import numpy as np
import mapping
import variance


class Training_Model(tf.keras.Model):

    def __init__(self, name, **kwargs):
        print("new")
        super(Training_Model, self).__init__(name=name)

        self.input_layer = tf.keras.layers.Flatten(input_shape=(28, 28), name="input")
        self.hidden_layer = Training_Layer(784, 36, activation="sigmoid", name="hidden", **kwargs)
        self.output_layer = Training_Layer(36, 10, activation="softmax", name="output", **kwargs)


    def call(self, input, **kwargs):

        x = self.input_layer(input)
        x = self.hidden_layer(x, **kwargs)
        x = self.output_layer(x, **kwargs)

        return x

    def test_step(self, data):
        x, y, sample_weight = data[0], data[1], data[2]

        y_pred = self(x, training=False)
        # Updates stateful loss metrics.
        self.compute_loss(x, y, y_pred, sample_weight)
        return self.compute_metrics(x, y, y_pred, sample_weight)



class Training_Layer(tf.keras.layers.Layer):

    def __init__(self, n_in, n_out, **kwargs):

        self.activation = kwargs.pop("activation", "sigmoid")

        self.nonlinearity = kwargs.pop("nonlinearity", None)
        self.weight_stddev = kwargs.pop("weight_stddev", None)
        self.l2_alpha = kwargs.pop("l2_alpha", None)

        super(Training_Layer, self).__init__(**kwargs)
        self.n_in = n_in
        self.n_out = n_out


        self.w = self.add_weight(
            shape=(n_in, n_out),
            initializer="random_normal",
            trainable=True,
            name="weights"
        )
        self.b = self.add_weight(
            shape=(n_out,),
            initializer="zeros",
            trainable=True,
            name="biases"
        )


    def noisy_w(self, w):
        noisy_w = w + w * tf.random.normal(w.shape, mean=0, stddev=self.weight_stddev)
        return noisy_w


    def call(self, input, **kwargs):

        training = kwargs.pop("training", True)

        w = self.w

        if (self.weight_stddev != None) and (training == True):
            w = self.noisy_w(w)

        if (self.nonlinearity != None):
            c_input = 0.5
            #w = self.w * (1.0 - c_input + 0.5 * c_input * self.nonlinearity * self.w)
            w = self.w * (1.0 - c_input * self.nonlinearity * (1.0 - 0.5 * self.w))


        dot = tf.tensordot(input, w, axes=1) + self.b

        if (self.activation == "sigmoid"):
            output = tf.nn.sigmoid(dot)
        elif (self.activation == "softmax"):
            output = tf.nn.softmax(dot)


        if (self.l2_alpha != None):
            self.add_loss(self.l2_alpha * tf.reduce_sum(tf.square(self.w)))


        return output