import tensorflow as tf

def compute_l1_distance(x, y):
    with tf.name_scope('l1_distance'):
        d = tf.reduce_sum(tf.abs(tf.subtract(x, y)), axis=1)
        return d

def compute_euclidean_distance(x, y):
    with tf.name_scope('euclidean_distance'):
        d = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x, y)), axis=1))
        return d

def compute_cosine_distance(x, y):
    with tf.name_scope('cosine_distance'):
        x_norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=1))
        y_norm = tf.sqrt(tf.reduce_sum(tf.square(y), axis=1))
        x_y = tf.reduce_sum(tf.multiply(x, y), axis=1)
        d = tf.divide(x_y, tf.multiply(x_norm, y_norm))
        return d

def comU1(x, y):
    result = [compute_cosine_distance(x, y), compute_euclidean_distance(x, y), compute_l1_distance(x, y)]
    return tf.stack(result, axis=1)

def comU2(x, y):
    result = [compute_cosine_distance(x, y), compute_euclidean_distance(x, y)]
    return tf.stack(result, axis=1)
