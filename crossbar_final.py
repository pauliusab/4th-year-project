import tensorflow as tf
import numpy as np
import math
import tensorflow_probability as tfp



class Memristive_Model(tf.keras.Model):

    def __init__(self, ideal_model, name, **kwargs):
        print("new")
        super(Memristive_Model, self).__init__(name=name)

        # get nonidealities frowm kwargs
        self.input_noise = kwargs.pop("training_with_input_noise", False)


        if (self.input_noise == False):
            index1, index2 = 1, 2
        else:
            index1, index2 = 2, 4       # should be 2, 4 if the model is defined as sequential layers

        # initializing layers
        self.input_layer = tf.keras.layers.Flatten(input_shape=(28, 28), name="input")
        self.hidden_layer = Memristive_Layer(ideal_model.get_layer(index=index1), name="hidden", **kwargs)
        self.output_layer = Memristive_Layer(ideal_model.get_layer(index=index2), name="output", activation="softmax", **kwargs)



    def call(self, input):

        # calls every layer sequentially (forward pass
        x = self.input_layer(input)
        x = self.hidden_layer(x)
        x = self.output_layer(x)

        # return probability vector
        return x


class Memristive_Layer(tf.keras.layers.Layer):

    def __init__(self, Layer, **kwargs):

        # need to pop kwargs before super() initialization to make sure they dont get passed on
        self.activation = kwargs.pop("activation", "sigmoid")

        self.k_g = kwargs.pop("k_g", None)
        self.g_min = kwargs.pop("g_min", 0.001)
        self.g_max = kwargs.pop("g_max", 0.002)

        self.nonlinearity = kwargs.pop("nonlinearity", None)
        self.read_noise_std = kwargs.pop("read_noise_std", None)
        self.d2d_std = kwargs.pop("d2d_std", None)


        super(Memristive_Layer, self).__init__(**kwargs)

        # get parameters from ideal model
        self.w = Layer.get_weights()[0]
        self.b = Layer.get_weights()[1]

        self.scaling_f = (self.g_max - self.g_min)


        # k_g is the inverse of the largest weight
        if (self.k_g == None):
            max_w = tf.reduce_max(tf.abs(tf.concat([self.w, tf.expand_dims(self.b, 0)], 0)))
            self.k_g = (self.g_max - self.g_min) / max_w


            # convert weights to conductances. All calculations are done using these values
        self.g_w, self.g_b = w_to_g(self)

        # need to apply d2d here to avoid passing parameter for layer initialization.
        if (self.d2d_std != None):
            self.disturb_weights()



    def disturb_weights(self):
        print("d2d")
        self.g_w = lognormal_disturbance(self.g_w, self.d2d_std)
        self.g_b = lognormal_disturbance(self.g_b, self.d2d_std)

    def apply_noise(self):
        print("noise")
        g_w_disturbed = lognormal_disturbance(self.g_w, self.read_noise_std)
        g_b_disturbed = lognormal_disturbance(self.g_b, self.read_noise_std)
        return g_w_disturbed, g_b_disturbed

    def apply_nonlinearity(self, g_w, input):
        print("nonlinearity")


        expanded_input = tf.expand_dims(input, axis=-1)

        if (self.name == "hidden"):
            extended_input = tf.repeat(expanded_input, repeats=36, axis=-1)
        elif (self.name == "output"):
            extended_input = tf.repeat(expanded_input, repeats=10, axis=-1)

        # convert input matrix to g shape (same voltage applied to positive and negative weight conductances)
        expanded_input_2 = tf.expand_dims(extended_input, axis=-1)
        extended_input_2 = tf.repeat(expanded_input_2, repeats=2, axis=-1)
        g_w = tf.convert_to_tensor(g_w, dtype="float32")

        # apply the inputs to disturb conductances
        # provides different scaling for each element in the pair by introducing nonlinearity factor g_max-g
        # the difference would be more significant if larger g range would be used
        g_w_nonlinear = tf.math.add(g_w, tf.math.multiply(extended_input_2 * tf.constant(float(self.nonlinearity)), self.scaling_f * self.scaling_f/g_w))  # g_max-g because lower g means more nonlinearity

        return g_w_nonlinear


    def call(self, input):

        dot_product_operator = tf.constant([1.0, -1.0])

        if (self.read_noise_std != None):
            g_w, g_b = self.apply_noise()
        else:
            g_w = self.g_w
            g_b = self.g_b

        if (self.nonlinearity != None):
            g_w = self.apply_nonlinearity(g_w, input)
            g_eff_w = tf.einsum("l,ijkl->ijk", dot_product_operator, g_w)
        else:
            g_eff_w = tf.einsum("k,ijk->ij", dot_product_operator, g_w)


        g_eff_b = tf.einsum("j,ij->i", dot_product_operator, g_b)
        w = g_eff_w / self.k_g
        b = g_eff_b / self.k_g
        # calculate the dot product using inputs and their corresponding nonlinear weights (for each input theres a different set of weights)
        if (self.nonlinearity != None):
            dot = tf.einsum("ikj,ik->ij", w, input) + b
        else:
            dot = tf.tensordot(input, w, axes=1) + b

        if (self.activation == "sigmoid"):
            output = tf.nn.sigmoid(dot)
        elif (self.activation == "softmax"):
            output = tf.nn.softmax(dot)

        return output


def lognormal_disturbance(g, std):
    r = 1.0 / np.asarray(g)
    r_disturbed = np.zeros(r.shape)
    for x in range(r.shape[0]):
        for y in range(r.shape[1]):
            try:
                for z in range(r.shape[2]):

                    mean = r[x, y, z]
                    this_std = std * mean

                    variance = math.pow(this_std, 2)
                    sigma_sqr = math.log((variance / math.pow(mean, 2)) + 1)
                    sigma = math.sqrt(sigma_sqr)
                    mu = math.log(mean) - (sigma_sqr) / 2

                    r_disturbed[x, y, z] = np.random.lognormal(mu, sigma)

            except IndexError:
                mean = r[x, y]
                this_std = std * mean

                variance = math.pow(this_std, 2)
                sigma_sqr = math.log((variance / math.pow(mean, 2)) + 1)
                sigma = math.sqrt(sigma_sqr)
                mu = math.log(mean) - (sigma_sqr) / 2

                r_disturbed[x, y] = np.random.lognormal(mu, sigma)

    g_disturbed = 1.0 / r_disturbed

    return g_disturbed


def w_to_g(layer):
    # converts weights to conductances at g_min
    w = layer.w
    flat_w = tf.reshape(w, -1)
    flat_w = tf.expand_dims(flat_w, 1)
    b = layer.b
    flat_b = tf.expand_dims(b, 1)

    # extend w matrix to be len(w),2
    w_map_matrix = tf.where(flat_w >= 0, flat_w * [[1.0, 0.0]], -flat_w * [[0.0, 1.0]])
    # convert w to g
    g_w = tf.where(w_map_matrix == 0, layer.g_min, layer.g_min + layer.k_g * (w_map_matrix))

    # reshape g_w to original dimensions
    g_w_shape = (w.shape[0], w.shape[1], 2)
    g_w = tf.reshape(g_w, g_w_shape)


    # extend b matrix to be len(w),2
    b_map_matrix = tf.where(flat_b >= 0, flat_b * [[1.0, 0.0]], -flat_b * [[0.0, 1.0]])
    # convert b to g
    g_b = tf.where(b_map_matrix == 0, layer.g_min, layer.g_min + layer.k_g * (b_map_matrix))


    return g_w, g_b