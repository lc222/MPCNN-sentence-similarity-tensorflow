#coding=utf8
from data_helper import *
import embedding as emb
from model import *
import time
import os
import datetime
import tensorflow.python.debug as tf_debug

tf.app.flags.DEFINE_integer('embedding_dim', 50, 'The dimension of the word embedding')
tf.app.flags.DEFINE_integer('num_filters_A', 20, 'The number of filters in block A')
tf.app.flags.DEFINE_integer('num_filters_B', 20, 'The number of filters in block B')
tf.app.flags.DEFINE_integer('n_hidden', 150, 'number of hidden units in the fully connected layer')
tf.app.flags.DEFINE_integer('sentence_length', 100, 'max size of sentence')
tf.app.flags.DEFINE_integer('num_classes', 6, 'num of the labels')

tf.app.flags.DEFINE_integer('num_epochs', 10, 'Number of epochs to be trained')
tf.app.flags.DEFINE_integer('batch_size', 128, 'size of mini batch')

tf.app.flags.DEFINE_integer("display_step", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.app.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.app.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.app.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

tf.app.flags.DEFINE_float('lr', 1e-3, 'learning rate')
tf.app.flags.DEFINE_float('l2_reg_lambda', 1e-4, 'regularization parameter')

tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

filter_size = [1,2,100]
conf = tf.app.flags.FLAGS
conf._parse_flags()

#glove是载入的次向量。glove.d是单词索引字典<word, index>，glove.g是词向量矩阵<词个数,300>
glove = emb.GloVe(N=50)

#-------------------------------------Loading data----------------------------------------------#
print ("Loading data...")

Xtrain, ytrain = load_set(glove, path='./sts/semeval-sts/all')
#[22592, 句长]
Xtest, ytest = load_set(glove, path='./sts/semeval-sts/2016')
#[1186, 句长]

# max_sent_length = max([len(x) for SS in Xtrain for x in SS])
# print max_sent_length #最大的句子长度为84
#-------------------------------------Loading finished----------------------------------------------#

#-------------------------------------training the network----------------------------------------------#
with tf.Session() as sess:
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    input_1 = tf.placeholder(tf.int32, [None, conf.sentence_length], name="input_x1")
    input_2 = tf.placeholder(tf.int32, [None, conf.sentence_length], name="input_x2")
    input_3 = tf.placeholder(tf.int32, [None, conf.num_classes], name="input_y")
    dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    with tf.name_scope("embendding"):
        s0_embed = tf.nn.embedding_lookup(glove.g, input_1)
        s1_embed = tf.nn.embedding_lookup(glove.g, input_2)

    with tf.name_scope("reshape"):
        input_x1 = tf.reshape(s0_embed, [-1, conf.sentence_length, conf.embedding_dim, 1])
        input_x2 = tf.reshape(s1_embed, [-1, conf.sentence_length, conf.embedding_dim, 1])
        input_y = tf.reshape(input_3, [-1, conf.num_classes])

    # sent1_unstack = tf.unstack(input_x1, axis=1)
    # sent2_unstack = tf.unstack(input_x2, axis=1)
    # D = []
    # for i in range(len(sent1_unstack)):
    #     d = []
    #     for j in range(len(sent2_unstack)):
    #         dis = compute_cosine_distance(sent1_unstack[i], sent2_unstack[j])
    #         d.append(dis)
    #     D.append(d)
    # D = tf.reshape(D, [-1, len(sent1_unstack), len(sent2_unstack), 1])
    # A = [tf.nn.softmax(tf.expand_dims(tf.reduce_sum(D, axis=i), 2)) for i in [2, 1]]
    #
    # print A[1]
    # print A[1] * input_x2
    # atten_embed = tf.concat([input_x2, A[1] * input_x2], 2)

    setence_model = MPCNN_Layer(conf.num_classes, conf.embedding_dim, filter_size,
                                [conf.num_filters_A, conf.num_filters_B], conf.n_hidden,
                                input_x1, input_x2, input_y, dropout_keep_prob)

    out = setence_model.similarity_measure_layer()
    with tf.name_scope("cost"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=setence_model.input_y))
    train_step = tf.train.AdamOptimizer(conf.lr).minimize(cost)

    predict_op = tf.argmax(out, 1)
    with tf.name_scope("accuracy"):
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(input_y, 1), tf.argmax(out, 1)), tf.float32))

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    loss_summary = tf.summary.scalar("loss", cost)
    acc_summary = tf.summary.scalar("accuracy", acc)

    train_summary_op = tf.summary.merge([loss_summary, acc_summary])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=conf.num_checkpoints)

    init = tf.global_variables_initializer().run()

    for j in range(10):
        for i in range(0, 20000, conf.batch_size):
            x1 = Xtrain[0][i:i + conf.batch_size]
            x2 = Xtrain[1][i:i + conf.batch_size]
            y = ytrain[i:i + conf.batch_size]
            _, summaries, accc, loss = sess.run([train_step, train_summary_op, acc, cost],
                                     feed_dict={input_1: x1, input_2: x2, input_3: y, dropout_keep_prob: 1.0})
            time_str = datetime.datetime.now().isoformat()
            print("{}: loss {:g}, acc {:g}".format(time_str, loss, accc))
            train_summary_writer.add_summary(summaries)
        print("\nEvaluation:")
        accc = sess.run(acc, feed_dict={input_1: Xtest[0], input_2: Xtest[1], input_3: ytest, dropout_keep_prob: 1.0})
        print "test accuracy:", accc

    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # for i in range(conf.num_epochs):
    #     training_batch = zip(range(0, len(Xtrain[0]), conf.batch_size),
    #                          range(conf.batch_size, len(Xtrain[0]) + 1, conf.batch_size))
    #     for start, end in training_batch:
    #         feed_dict = {input_1: Xtrain[0][start:end], input_2: Xtrain[1][start:end],
    #                      dropout_keep_prob: 0.5, input_3: ytrain[start:end]}
    #         print start
    #         #assert all(x.shape == (100, 100) for x in Xtrain[0][start:end])
    #         loss, _ = sess.run(train_step, feed_dict=feed_dict)
    #         print("Epoch:", '%04d' % (i + 1), "cost=", "{:.9f}".format(loss))

    print("Optimization Finished!")