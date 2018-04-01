__author__ = 'Zumo Arthicha Sri'

'''*************************************************
*                                                  *
*                  import module                   *
*                                                  *
*************************************************'''

# system module
import os
import sys


# deep learning module
import tensorflow as tf

# matrix module
import numpy as np





'''*************************************************
*                                                  *
*                 config program                   *
*                                                  *
*************************************************'''

#disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class TenzorCNN:

    #list of CNN layers
    weigth = []
    bias = []
    conv_relu = []
    pool = []

    # create max pool layer
    def max_pool_2x2(self,x,stride=2):
        return tf.nn.max_pool(x, ksize=[1, stride, stride, 1],strides=[1, stride, stride, 1], padding='SAME')

    # create weigth matrix
    def weight_variable(self,shape):
        #random weigth as truncated_normal distribution
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # create bias matrix
    def bias_variable(self,shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # create convolution layer
    def conv2d(self,x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def CNN2(self,x,hidden_layer,kernel_size,pool_size,strides,image_size,keep_prob=1.0,input_drop=False,fully=30):
        image_batch = x
        imgZ = image_size[0]*image_size[1]
        for i in range(0,len(hidden_layer)-1):
            if i == 0:
                if input_drop:
                    input_neurons = tf.layers.dropout(inputs=image_batch, rate=keep_prob)
                else:
                    input_neurons = image_batch

            else:
                input_neurons = self.pool[-1]


            #with tf.name_scope('dropout_layer_' + str(i)):
                #input_neurons = tf.nn.dropout(input_neurons, keep_prob)

            with tf.name_scope("conv"+str(i)):
                self.conv_relu.append(tf.layers.conv2d(inputs=input_neurons, filters=hidden_layer[i], kernel_size=kernel_size[i],
                                     padding='same', activation=tf.nn.leaky_relu))
            with tf.name_scope("pool"+str(i)):
                if pool_size[i] != None:
                    self.pool.append(tf.layers.max_pooling2d(inputs=self.conv_relu[-1], pool_size=pool_size[i], strides=strides[i]))
                    imgZ = imgZ//(pool_size[i][0]*pool_size[i][1])
                else:
                    self.pool.append(self.conv_relu[-1])



        with tf.name_scope("dense"):
            # The 'images' are now 7x7 (28 / 2 / 2), and we have 64 channels per image
            pool_flat = tf.reshape(self.pool[-1], [-1, imgZ *hidden_layer[-2]])
            dense = tf.layers.dense(inputs=pool_flat, units=hidden_layer[-1], activation=tf.nn.relu)

        with tf.name_scope("dropout"):
            # Add dropout operation; 0.8 probability that a neuron will be kept
            dropout = tf.layers.dropout(
                inputs=dense, rate=keep_prob)
        logits = tf.layers.dense(inputs=dropout, units=fully)
        prob = tf.sigmoid(logits)
        return prob, [0,0] #self.conv_relu

    # create Convolutional Neural Network
    def CNN(self,x,hidden_layer,keep_prob=1.0,pool=True,stride=2):


        # check amount of hidden layer
        if len(hidden_layer) <= 1:
            sys.exit("error: amount of hidden layer error")

        # loop through each set of layer (node -> conv -> relu -> pool)
        for i in range(1,len(hidden_layer)):
            with tf.name_scope('convolutional_layer_'+str(i)):

                # create node
                self.weigth.append(self.weight_variable([5, 5, hidden_layer[i-1], hidden_layer[i]]))
                self.bias.append(self.bias_variable([hidden_layer[i]]))

                if i == 1:
                    inputMatrix = x
                else:
                    inputMatrix = self.pool[-1]

                # dropout -> overfitting
                with tf.name_scope('dropout_layer_' + str(i)):
                    inputMatrix = tf.nn.dropout(inputMatrix, keep_prob)

                # apply convolution and relu activation function

                self.conv_relu.append(tf.nn.relu(self.conv2d(inputMatrix, self.weigth[-1]) + self.bias[-1]))

            # max pool layer
            do_pu = False
            if pool:
                do_pu = True
            else:
                if pool[i]:
                    do_pu = True

            if do_pu:
                with tf.name_scope('pool_layer_' + str(i)):
                    self.pool.append(self.max_pool_2x2(self.conv_relu[-1],stride=stride))
            else:
                self.pool.append(self.conv_relu[-1])

        # return output of CNN
        return self.pool[-1]

class TenzorNN:

    weigth = []
    bias = []
    output = []

    # create weigth matrix
    def weight_variable(self,shape):
        #random weigth as truncated_normal distribution
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # create bias matrix
    def bias_variable(self,shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # create neural network layer
    # hidden_layer is percentage from the input neuron
    def NeuralNetwork(self,x,hidden_layer,imageSize,keep_prob=1.0,shape=7,fc_neu=10):

        # check amount of hidden layer
        onlyOne = 1
        if len(hidden_layer) < 1:
            sys.exit("error: amount of hidden layer error")
        elif len(hidden_layer) == 1:
            onlyOne = 0

        input_neu = shape*imageSize

        for i in range(0,len(hidden_layer)-onlyOne):

            with tf.name_scope('neural_network_layers'):

                if onlyOne == 0:
                    nextLayer = 10
                else:
                    nextLayer = int(input_neu*hidden_layer[i+1])

                if i == 0:
                    input =  tf.reshape(x, [-1, int(input_neu*hidden_layer[i])])
                    gg = input
                else:
                    input = self.output[-1]
                self.weigth.append(self.weight_variable([int(input_neu* hidden_layer[i]), nextLayer]))
                self.bias.append(self.bias_variable([nextLayer]))
                opMTX = tf.matmul(input, self.weigth[-1]) + self.bias[-1]
                # dropout -> overfitting
                with tf.name_scope('dropout_layer_' + str(i)):
                    opMTX = tf.nn.dropout(opMTX, keep_prob)
                self.output.append(tf.nn.relu(opMTX))

            #dropout layer ignor some term -> regularization
            if onlyOne:
                with tf.name_scope('last_droupout_layer'):
                    self.output[-1] = tf.nn.dropout(self.output[-1], keep_prob)

        if onlyOne:
            with tf.name_scope('fully_connected_layer'):
                W_fc = self.weight_variable([int(input_neu*hidden_layer[-1]), fc_neu])

                b_fc = self.bias_variable([fc_neu])
                y_ = tf.matmul(self.output[-1], W_fc) + b_fc
        else:
            y_ = self.output[-1]

        return y_


class TenzorAE:

    auto_layer = []

    # create convolution layer
    def conv2d(self,x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def weight_variable(self,shape):
        # From the mnist tutorial
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)


    def bias_variable(self,shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


    def fc_layer(self,previous, input_size, output_size):
        W = self.weight_variable([input_size, output_size])
        b = self.bias_variable([output_size])
        #return self.conv2d(previous,W)
        return tf.matmul(previous, W) + b

    def AE(self,x,hidden_layers,USE_RELU=True,keep_prob=1.0,acti_func='tanh'):

        if len(hidden_layers) < 1:
            sys.exit("error: amount of hidden layer error")
        elif len(hidden_layers) %2 == 0:
            sys.exit("error: amount of hidden layer is even")

        for i in range(0,len(hidden_layers)-1):
            if i == 0:
                ipt = x
            else:
                ipt = self.auto_layer[-1]

            func = self.fc_layer(ipt,hidden_layers[i],hidden_layers[i+1])
            func = tf.nn.dropout(func, keep_prob)
            if acti_func == 'tanh':
                self.auto_layer.append(tf.nn.tanh(func))


        if USE_RELU:

            fr= self.fc_layer(self.auto_layer[-1], hidden_layers[-1],  hidden_layers[-1])
            fr = tf.nn.dropout(fr, keep_prob)
            out = tf.nn.relu(fr)
        else:
            out= self.fc_layer(self.auto_layer[-1], hidden_layers[-1],  hidden_layers[-1])

        return out,self.auto_layer[(len(hidden_layers)//2)-1]