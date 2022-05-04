import logging
#from . import logging
import tensorflow as tf
import numpy as np
from itertools import combinations


def l1_norm(V, W, lambda_=0.001):
    return tf.reduce_sum(
        tf.add(tf.multiply(lambda_, tf.abs(W)), tf.multiply(lambda_, tf.abs(V)))
    )


def l2_norm(V, W, lambda_=0.001):
    return tf.reduce_sum(
        tf.add(tf.multiply(lambda_, tf.pow(W, 2)), tf.multiply(lambda_, tf.pow(V, 2)))
    )


def noop_norm(V, W, lambda_=None):
    return 0


def fm(X, w0, W, V):
    linear_terms = X * W
    interactions = tf.subtract(
        tf.pow(tf.tensordot(X, tf.transpose(V), 1), 2),
        tf.tensordot(tf.pow(X, 2), tf.transpose(tf.pow(V, 2)), 1),
    )
    V_out = 0
    for i in range(tf.shape(V)[1]-2):
        v1 = V[:,i:i+1]
        v2 = V[:,i+1:i+2]
        v3 = V[:,i+2:i+3]
        o = tf.math.multiply(tf.math.multiply(v1,v2), v3)
        V_out += np.sum(o)

    choose = 3
    data = []
    for k in range(tf.shape(X)[0]):
        y = [i for i in combinations(X[k, :].numpy(), choose)]
        for i in range(len(y)):
            xx = []
            for j in range(choose):
                print(y[i][j], end=' ')
                xx.append(y[i][j])
            print()
            out = 1
            for x in xx:
                out *= x
            out *= V_out
            #print(out)
            data.append(out)


    third = np.array(data)

    if X.ndim > 1:
        linear_terms = tf.reduce_sum(linear_terms, 1, keepdims=True)
        interactions = tf.reduce_sum(interactions, 1, keepdims=True)
        # third = tf.reduce_sum(third, keepdims=True)

    else:
        # One dimensional data: e.g. passed when we call fm() for inference
        linear_terms = tf.reduce_sum(linear_terms)
        interactions = tf.reduce_sum(interactions)
        # third = tf.reduce_sum(third)
    # third = tf.cast(third, tf.float32)
    return w0 + linear_terms + 0.5 * interactions


def train(
    train_dataset,
    num_factors=2,
    max_iter=10,
    penalty=None,
    C=1.0,
    loss=None,
    optimizer=None,
    random_state=None,
    dtype=tf.float32,
):
    """Fit a degree 2 polynomial factorization machine, implemented atop Tensorflow 2.
    This class contains the generic code to train a Factorazione Machine. Regressors and classifiers can be learnt
    by minimizing appropriate loss functions (e.g. MSE or cross entropy).

    :param train_dataset: an instance of tensorflow.data.Dataset that contains training data.
    :param num_factors: number of latent factor vectors.
    :param max_iter: iterations to convergence.
    :param penalty: regularization (l1, l2 or None). Default l2.
    :param C: inverse of regularization strength.
    :param loss: a tensorflow.keras.losses object (e.g. MSE, binary_crossentropy).
    :param optimizer: a tensorflow.keras.optimizers object (e.g. tf.keras.optimizers.Adam).
    :param random_state: int, random state.
    :param dtype: train_dataset types. Default float32.
    :returns w0, W, V: tensorflow.Variable instances for bias, weights and interaction factors.
    """
    tf.random.set_seed(random_state)
    if C < 0:
        raise ValueError(f"Inverse regularization term must be positive; got (C={C})")
    if max_iter < 1:
        raise ValueError(f"max_iter must be > zero. Got {max_iter}")
    if num_factors < 1:
        raise ValueError(f"num_factors must be >= 1. Got {num_factors}")

    # Get the number of feature columns
    p = train_dataset.shape[1]
    # bias and weights
    w0 = tf.Variable(tf.zeros([1], dtype=dtype))
    W = tf.Variable(tf.zeros([13], dtype=dtype))
    # interaction factors, randomly initialized
    V = tf.Variable(
        tf.random.normal(
            [num_factors, 13], mean=0.0, stddev=0.01, dtype=dtype, seed=random_state
        )
    )
    
    
    for epoch_count in range(max_iter):
        for batch, (x, y) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                pred = fm(x, w0, W)
                loss_ = loss(y, pred) + penalty(V, W, lambda_=1.0 / C)
            grads = tape.gradient(loss_, [w0, W, V]) #GradientTape.gradient(target, sources)
            optimizer.apply_gradients(zip(grads, [w0, W, V]))
            logging.debug(f"Epoch: {epoch_count}, batch: {batch} loss:, {loss_.numpy()}")
    return w0, W, V
