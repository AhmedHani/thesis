import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


class CGAN(object):

    def train(self, train_x, train_y, learning_rate=0.1, epochs=10, batch_size=128):
        with tf.variable_scope('G') as g_score:
            self.__g_input = tf.placeholder(tf.float32, [None, self.generator.get_input_size()])
            g_z = self.generator.forward(self.__g_input)

        with tf.variable_scope('D') as d_scope:
            self.__d_input = tf.placeholder(tf.float32, [None, self.discriminator.get_input_size()])
            d_real = self.discriminator.forward(self.__d_input)
            d_scope.reuse_variables()
            d_fake = self.discriminator.forward(g_z)

        eps = 1e-2
        D_loss = tf.reduce_mean(-tf.log(d_real + eps) - tf.log(1 - d_fake + eps))
        G_loss = tf.reduce_mean(-tf.log(d_fake + eps))

        # trainable variables for each network
        t_vars = tf.trainable_variables()
        D_vars = [var for var in t_vars if 'D' in var.name]
        G_vars = [var for var in t_vars if 'G' in var.name]

        #print D_vars
        #print G_vars


        # optimizer for each network
        D_optim = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list=D_vars)
        G_optim = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list=G_vars)

        # open session and initialize all variables
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        for epoch in range(epochs):
            G_losses = []
            D_losses = []

            current_batch_x, current_batch_y = [], []

            for i in range(train_x.shape[0]):
                current_batch_x.append(train_x[i])
                current_batch_y.append(train_y[i])

                if len(current_batch_x) != batch_size:
                    continue

                current_batch_x = np.asarray(current_batch_x)
                current_batch_y = np.asarray(current_batch_y)

                z_ = np.random.normal(0, 1, (batch_size, 100))
                z_ = np.concatenate((z_, current_batch_y), axis=1)

                loss_d_, _ = sess.run([D_loss, D_optim], {self.__d_input: current_batch_x, self.__g_input: z_})
                D_losses.append(loss_d_)

                z_ = np.random.normal(0, 1, (batch_size, 100))
                z_ = np.concatenate((z_, current_batch_y), axis=1)

                loss_g_, _ = sess.run([G_loss, G_optim], {self.__g_input: z_})

                G_losses.append(loss_g_)

                current_batch_x = []
                current_batch_y = []

    def build_generator(self, latent_space_size,
                        output_size,
                        hidden_layers_size,
                        hidden_layers_neurons_size,
                        hidden_layer_activation='tanh',
                        output_layer_activation='softmax',
                        conditiond_features_size=None):
        self.generator = self.DeepNeuralNetwork('G')
        self.generator.set_input_size(latent_space_size + conditiond_features_size if conditiond_features_size is not None else latent_space_size)
        self.generator.set_hidden_layers_size(hidden_layers_size)
        self.generator.set_neurons_size_per_layer(hidden_layers_neurons_size)
        self.generator.set_hidden_layers_activ_function(hidden_layer_activation)
        self.generator.set_output_size(output_size)
        self.generator.set_output_activ_function(output_layer_activation)

        self.generator.build()

        return self.generator

    def build_discriminator(self, input_size,
                            hidden_layers_size,
                            hidden_layers_neurons_size,
                            hidden_layer_activation='tanh',
                            output_layer_activation='sigmoid',
                            output_size=1):
        self.discriminator = self.DeepNeuralNetwork('D')
        self.discriminator.set_input_size(input_size)
        self.discriminator.set_hidden_layers_size(hidden_layers_size)
        self.discriminator.set_neurons_size_per_layer(hidden_layers_neurons_size)
        self.discriminator.set_hidden_layers_activ_function(hidden_layer_activation)
        self.discriminator.set_output_size(output_size)
        self.discriminator.set_output_activ_function(output_layer_activation)

        self.discriminator.build()

        return self.discriminator

    class DeepNeuralNetwork(object):

        def __init__(self, model_name):
            self.__model_name = model_name

            self.__input_size = None
            self.__input = None

            self.__hidden_layers_size = None
            self.__neurons_size_per_layer = None
            self.__hidden_layers_activ_function = None

            self.__output_size = None
            self.__output_activ_function = None

            self.__network = []

            self.__optimizer = None
            self.__learning_rate = None

            self.__model_saver = None
            self.__model_loader = None

            self.__train_features_data = None
            self.__train_labels_data = None
            self.__test_features_data = None
            self.__test_labels_data = None

            self.__train = None

        def set_input_size(self, input_size):
            self.__input_size = input_size

        def get_input_size(self):
            return self.__input_size

        def set_output_size(self, output_size):
            self.__output_size = output_size

        def get_output_size(self):
            return self.__output_size

        def set_hidden_layers_size(self, hidden_layers_size):
            self.__hidden_layers_size = hidden_layers_size

        def set_neurons_size_per_layer(self, neurons_list):
            if self.__hidden_layers_size != len(neurons_list):
                raise Exception("Miss matched sizes!")

            self.__neurons_size_per_layer = neurons_list

        def set_hidden_layers_activ_function(self, activ_func="tanh"):
            self.__hidden_layers_activ_function = activ_func

        def set_output_activ_function(self, activ_func="softmax"):
            self.__output_activ_function = activ_func

        def build(self):
            input_hidden_weights = tf.Variable(
                tf.random_uniform([self.__input_size, self.__neurons_size_per_layer[0]],
                                  -1.0 / math.sqrt(self.__input_size),
                                  1.0 / math.sqrt(self.__input_size)), name=self.__model_name + "_W" + str(0))
            input_hidden_bias = tf.Variable(tf.ones([self.__neurons_size_per_layer[0]]), name=self.__model_name + "_B" + str(0))

            self.__network.append(((input_hidden_weights, input_hidden_bias), (0, 1)))

            for i in range(1, self.__hidden_layers_size + 1):
                if i == self.__hidden_layers_size:
                    hidden_output_weights = tf.Variable(
                        tf.random_uniform(
                            [self.__neurons_size_per_layer[self.__hidden_layers_size - 1], self.__output_size],
                            -1.0 / math.sqrt(self.__neurons_size_per_layer[self.__hidden_layers_size - 1]),
                            1.0 / math.sqrt(self.__neurons_size_per_layer[self.__hidden_layers_size - 1])), name=self.__model_name + "_W" + str(i))
                    hidden_output_bias = tf.Variable(tf.ones([self.__output_size]), name=self.__model_name + "_B" + str(i))  # The bias in one

                    self.__network.append(((hidden_output_weights, hidden_output_bias),
                                           (self.__hidden_layers_size, self.__hidden_layers_size + 1)))

                    break

                else:
                    hidden_hidden_weights = tf.Variable(
                        tf.random_uniform([self.__neurons_size_per_layer[i - 1], self.__neurons_size_per_layer[i]],
                                          -1.0 / math.sqrt(self.__neurons_size_per_layer[i - 1]),
                                          1.0 / math.sqrt(self.__neurons_size_per_layer[i - 1])), name=self.__model_name + "_W" + str(i))
                    hidden_hidden_bias = tf.Variable(tf.ones([self.__neurons_size_per_layer[i]]), name=self.__model_name + "_B" + str(i))

                    self.__network.append(((hidden_hidden_weights, hidden_hidden_bias), (i - 1, i)))

        def forward(self, x):
            input_hidden_layers = self.__network[0]
            input_hidden_weights = input_hidden_layers[0][0]
            input_hidden_bias = input_hidden_layers[0][1]

            hidden_neurons_values = tf.matmul(x, input_hidden_weights) + input_hidden_bias
            hidden_activation_result = tf.nn.tanh(hidden_neurons_values)

            for i in range(1, len(self.__network) - 1):
                hidden_hidden_layers = self.__network[i]
                hidden_hidden_weights = hidden_hidden_layers[0][0]
                hidden_hidden_bias = hidden_hidden_layers[0][1]
                hidden_neurons_values = tf.matmul(hidden_activation_result, hidden_hidden_weights) + hidden_hidden_bias

                if self.__hidden_layers_activ_function is "relu":
                    hidden_activation_result = tf.nn.relu(hidden_neurons_values)
                elif self.__hidden_layers_activ_function is "tanh":
                    hidden_activation_result = tf.nn.tanh(hidden_neurons_values)
                elif self.__hidden_layers_activ_function is "sigmoid":
                    hidden_activation_result = tf.nn.sigmoid(hidden_neurons_values)

            hidden_output_layers = self.__network[len(self.__network) - 1]
            hidden_output_weights = hidden_output_layers[0][0]
            hidden_output_bias = hidden_output_layers[0][1]

            output_neurons_values = tf.matmul(hidden_activation_result, hidden_output_weights) + hidden_output_bias

            if self.__output_activ_function is "relu":
                output = tf.nn.relu(output_neurons_values)
            elif self.__output_activ_function is "tanh":
                output = tf.nn.tanh(output_neurons_values)
            elif self.__hidden_layers_activ_function is "sigmoid":
                output = tf.nn.sigmoid(output_neurons_values)
            elif self.__hidden_layers_activ_function is "softmax":
                output = tf.nn.softmax(output_neurons_values)
            else:
                output = output_neurons_values

            return output


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_set = (mnist.train.images - 0.5) / 0.5 # normalization; range: -1 ~ 1

cgan = CGAN()
cgan.build_generator(latent_space_size=100,
                     output_size=train_set.shape[1],
                     hidden_layers_size=4,
                     hidden_layers_neurons_size=[256, 512, 1024, 784],
                     hidden_layer_activation='relu',
                     output_layer_activation='tanh',
                     conditiond_features_size=10)
cgan.build_discriminator(input_size=train_set.shape[1],
                         output_size=1,
                         hidden_layers_size=3,
                         hidden_layers_neurons_size=[1024, 512, 256],
                         hidden_layer_activation='relu',
                         output_layer_activation='sigmoid')
cgan.train(train_x=train_set,
           train_y=mnist.train.labels,
           learning_rate=0.1,
           epochs=10,
           batch_size=512)