import tensorflow as tf
from utils import *

def init_weight(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01), name=name)

class MPCNN_Layer():
    def __init__(self, num_classes, embedding_size, filter_sizes, num_filters, n_hidden,
                 input_x1, input_x2, input_y, dropout_keep_prob):
        '''

        :param sequence_length:
        :param num_classes:
        :param embedding_size:
        :param filter_sizes:
        :param num_filters:
        '''
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self .num_filters = num_filters
        self.poolings = [tf.reduce_max, tf.reduce_min, tf.reduce_mean]

        self.input_x1 = input_x1
        self.input_x2 = input_x2
        self.input_y = input_y
        self.dropout_keep_prob = dropout_keep_prob

        self.W1 = [init_weight([filter_sizes[0], embedding_size, 1, num_filters[0]], "W1_0"),
                   init_weight([filter_sizes[1], embedding_size, 1, num_filters[0]], "W1_1"),
                   init_weight([filter_sizes[2], embedding_size, 1, num_filters[0]], "W1_2")]
        self.b1 = [tf.Variable(tf.constant(0.1, shape=[num_filters[0]]), "b1_0"),
                   tf.Variable(tf.constant(0.1, shape=[num_filters[0]]), "b1_1"),
                   tf.Variable(tf.constant(0.1, shape=[num_filters[0]]), "b1_2")]

        self.W2 = [init_weight([filter_sizes[0], embedding_size, 1, num_filters[1]], "W2_0"),
                   init_weight([filter_sizes[1], embedding_size, 1, num_filters[1]], "W2_1")]
        self.b2 = [tf.Variable(tf.constant(0.1, shape=[num_filters[1], embedding_size]), "b2_0"),
                   tf.Variable(tf.constant(0.1, shape=[num_filters[1], embedding_size]), "b2_1")]
        self.h = num_filters[0]*len(self.poolings)*2 + \
                 num_filters[1]*(len(self.poolings)-1)*(len(filter_sizes)-1)*3 #+ \
                 #len(self.poolings)*len(filter_sizes)*len(filter_sizes)*3
        self.Wh = tf.Variable(tf.random_normal([self.h, n_hidden], stddev=0.01), name='Wh')
        self.bh = tf.Variable(tf.constant(0.1, shape=[n_hidden]), name="bh")

        self.Wo = tf.Variable(tf.random_normal([n_hidden, num_classes], stddev=0.01), name='Wo')

    def attention(self):
        sent1_unstack = tf.unstack(self.input_x1, axis=1)
        sent2_unstack = tf.unstack(self.input_x2, axis=1)
        D = []
        for i in range(len(sent1_unstack)):
            d = []
            for j in range(len(sent2_unstack)):
                dis = compute_cosine_distance(sent1_unstack[i], sent2_unstack[j])
                #dis:[batch_size, 1(channels)]
                d.append(dis)
            D.append(d)
        D = tf.reshape(D, [-1, len(sent1_unstack), len(sent2_unstack), 1])
        A = [tf.nn.softmax(tf.expand_dims(tf.reduce_sum(D, axis=i), 2)) for i in [2, 1]]
        atten_embed = []
        atten_embed.append(tf.concat([self.input_x1, A[0] * self.input_x1], 2))
        atten_embed.append(tf.concat([self.input_x2, A[1] * self.input_x2], 2))
        return atten_embed

    def per_dim_conv_layer(self, x, w, b, pooling):
        '''

        :param input: [batch_size, sentence_length, embed_size, 1]
        :param w: [ws, embedding_size, 1, num_filters]
        :param b: [num_filters, embedding_size]
        :param pooling:
        :return:
        '''
        # unpcak the input in the dim of embed_dim
        input_unstack = tf.unstack(x, axis=2)
        w_unstack = tf.unstack(w, axis=1)
        b_unstack = tf.unstack(b, axis=1)
        convs = []
        for i in range(x.get_shape()[2]):
            conv = tf.nn.relu(tf.nn.conv1d(input_unstack[i], w_unstack[i], stride=1, padding="VALID") + b_unstack[i])
            # [batch_size, sentence_length-ws+1, num_filters_A]
            convs.append(conv)
        conv = tf.stack(convs, axis=2)  # [batch_size, sentence_length-ws+1, embed_size, num_filters_A]
        pool = pooling(conv, axis=1)  # [batch_size, embed_size, num_filters_A]

        return pool

    def bulit_block_A(self, x):
        #bulid block A and cal the similarity according to algorithm 1
        out = []
        with tf.name_scope("bulid_block_A"):
            for pooling in self.poolings:
                pools = []
                for i, ws in enumerate(self.filter_sizes):
                    #print x.get_shape(), self.W1[i].get_shape()
                    with tf.name_scope("conv-pool-%s" %ws):
                        conv = tf.nn.conv2d(x, self.W1[i], strides=[1, 1, 1, 1], padding="VALID")
                        #print conv.get_shape()
                        conv = tf.nn.relu(conv + self.b1[i])  # [batch_size, sentence_length-ws+1, 1, num_filters_A]
                        pool = pooling(conv, axis=1)
                    pools.append(pool)
                out.append(pools)
            return out

    def bulid_block_B(self, x):
        out = []
        with tf.name_scope("bulid_block_B"):
            for pooling in self.poolings[:-1]:
                pools = []
                with tf.name_scope("conv-pool"):
                    for i, ws in enumerate(self.filter_sizes[:-1]):
                        with tf.name_scope("per_conv-pool-%s" % ws):
                            pool = self.per_dim_conv_layer(x, self.W2[i], self.b2[i], pooling)
                        pools.append(pool)
                    out.append(pools)
            return out

    def similarity_sentence_layer(self):
        sent1 = self.bulit_block_A(self.input_x1)
        sent2 = self.bulit_block_A(self.input_x2)
        fea_h = []
        with tf.name_scope("cal_dis_with_alg1"):
            for i in range(3):
                regM1 = tf.concat(sent1[i], 1)
                regM2 = tf.concat(sent2[i], 1)
                for k in range(self.num_filters[0]):
                    fea_h.append(comU2(regM1[:, :, k], regM2[:, :, k]))

        #self.fea_h = fea_h

        fea_a = []
        with tf.name_scope("cal_dis_with_alg2_2-9"):
            for i in range(3):
                for j in range(len(self.filter_sizes)):
                    for k in range(len(self.filter_sizes)):
                        fea_a.append(comU1(sent1[i][j][:, 0, :], sent2[i][k][:, 0, :]))

        sent1 = self.bulid_block_B(self.input_x1)
        sent2 = self.bulid_block_B(self.input_x2)

        fea_b = []
        with tf.name_scope("cal_dis_with_alg2_last"):
            for i in range(len(self.poolings)-1):
                for j in range(len(self.filter_sizes)-1):
                    for k in range(self.num_filters[1]):
                        fea_b.append(comU1(sent1[i][j][:, :, k], sent2[i][j][:, :, k]))
        #self.fea_b = fea_b
        return tf.concat(fea_h + fea_b, 1)

    def similarity_measure_layer(self):
        fea = self.similarity_sentence_layer()
        # fea_h.extend(fea_a)
        # fea_h.extend(fea_b)
        #print len(fea_h), fea_h
        #fea = tf.concat(fea_h+fea_a+fea_b, 1)
        #print fea.get_shape()
        with tf.name_scope("full_connect_layer"):
            h = tf.nn.tanh(tf.matmul(fea, self.Wh) + self.bh)
            h = tf.nn.dropout(h, self.dropout_keep_prob)
            o = tf.matmul(h, self.Wo)
            return o
