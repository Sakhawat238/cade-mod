"""
autoencoder.py
~~~~~~~

Functions for training a unified autoencoder or individual autoencoders for each family.

"""

import os, sys
os.environ['PYTHONHASHSEED'] = '0'
from numpy.random import seed
import random
random.seed(1)
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5

import math
import time
import warnings

from sklearn.cluster import KMeans

from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

import numpy as np


def get_cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy.
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def epoch_batches(X_train, y_train, batch_size, similar_samples_ratio):
    '''
        used for contrastive autoencoder split data into pairs of same label and different labels
        code was adapted from https://github.com/mmasana/OoD_Mining.
    '''
    if batch_size % 4 == 0:
        half_size = int(batch_size / 2) # the really used batch_size for each batch. Another half data is filled by similar and dissimilar samples.
    else:
        print('batch_size should be a multiple of 4.')
        sys.exit(-1)

    # Divide data into batches. # TODO: ignore the last batch for now, maybe there is a better way to address this.
    batch_count = int(X_train.shape[0] / half_size)
    print(f'batch_count: {batch_count}')  # -> 118
    num_sim = int(batch_size * similar_samples_ratio)  # 64 * 0.25 = 16
    b_out_x = np.zeros([batch_count, batch_size, X_train.shape[1]])
    b_out_y = np.zeros([batch_count, batch_size], dtype=int)
    print(f'b_out_x: {b_out_x.shape}, b_out_y: {b_out_y.shape}')

    random_idx = np.random.permutation(X_train.shape[0]) # random shuffle the batches
    # split the random shuffled X_train and y_train to batch_count shares
    b_out_x[:, :half_size] = np.split(X_train[random_idx[: batch_count * half_size]], batch_count)
    b_out_y[:, :half_size] = np.split(y_train[random_idx[: batch_count * half_size]], batch_count)

    tmp = random_idx[half_size]

    # NOTE: if error here, it's because we didn't convert X_train and X_test as np.float32 when generating the npz file.
    assert np.all(X_train[tmp] == b_out_x[1, 0])  # to check if the split is correct

    # Sort data by label
    index_cls, index_no_cls = [], []
    ''' NOTE: if we want to adapt to training label non-continuing, e.g., [0,1,2,3,4,5,7], but this would cause
    b_out_y[b, m] list index out of range. So we should convert [0,1,2,3,4,5,7] to [0,1,2,3,4,5,6] in the training set.'''
    for label in range(len(np.unique(y_train))):
        index_cls.append(np.where(y_train == label)[0]) # each row shows the index of y_train where y_train == label
        index_no_cls.append(np.where(y_train != label)[0])

    index_cls_len = [len(e) for e in index_cls]
    print(f'index_cls len: {index_cls_len}')
    index_no_cls_len = [len(e) for e in index_no_cls]
    print(f'index_no_cls len: {index_no_cls_len}')

    print('generating the batches and pairs...')
    # Generate the pairs
    print(f'num_sim: {num_sim}')
    print(f'half_size: {half_size}')

    for b in range(batch_count):
        # Get similar samples
        for m in range(0, num_sim):
            # random sampling without replacement, randomly pick an index from y_train
            # where y_train[index] = b_out_y[b, m]
            # NOTE: list() operation is very slow, random.sample is also slower than np.random.choice()
            # ## pair = random.sample(list(index_cls[b_out_y[b, m]]), 1) would take 80s for each b,
            # np.random.choice() and list() would lead to 130s for each b
            # using only np.random.choice() would be 0.06s for each b
            pair = np.random.choice(index_cls[b_out_y[b, m]], 1)
            b_out_x[b, m + half_size] = X_train[pair[0]] # pick num_sim samples with the same label
            b_out_y[b, m + half_size] = y_train[pair[0]]
        # pick (half_size - num_sim) dissimilar samples
        for m in range(num_sim, half_size):
            # randomly pick an index from y_train where y_train[index] != b_out_y[b, m]
            pair = np.random.choice(index_no_cls[b_out_y[b, m]], 1)
            b_out_x[b, m + half_size] = X_train[pair[0]]
            b_out_y[b, m + half_size] = y_train[pair[0]]
        # DEBUG
        # if b == 1:
            # b_out_y[0] should looks like this (for simplicity assuming batch_size = 32, half_size = 16)
            # The first half is similar, the second half is dissimilar
            # 1, 2, 4, 8 | 2, 3, 5, 6
            # 1, 2, 4, 8 | 3, 4, 1, 7
            # print(f'b_out_x[1, 0, :20]: {b_out_x[b, 0, :20]}')
            # print(f'b_out_y[1]: {b_out_y[b]}')

    print(f'split batch finished') 
    return batch_count, b_out_x, b_out_y


class Autoencoder(object):
    def __init__(self,
                 dims,
                 activation='relu',
                 init='glorot_uniform',
                 verbose=1):
        '''
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
        The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        activation: activation, not applied to Input, last layer of the encoder, and Output layers

        '''
        self.dims = dims
        self.act = activation
        self.init = init
        self.verbose = verbose

    def build(self):
        """Fully connected auto-encoder model, symmetric.

        return:
            (ae_model, encoder_model), Model of autoencoder and model of encoder
        """
        dims = self.dims
        act = self.act
        init = self.init

        n_stacks = len(dims) - 1
        # input
        input_img = Input(shape=(dims[0],), name='input')
        x = input_img
        # internal layers in encoder
        for i in range(n_stacks-1):
            x = Dense(dims[i + 1], activation=act,
                      kernel_initializer=init, name='encoder_%d' % i)(x)
            # kernel_initializer is a fancy term for which statistical distribution or function to use for initializing the weights. 
            # Neural network needs to start with some weights and then iteratively update them

        # hidden layer, features are extracted from here, no activation is applied here, i.e., "linear" activation: a(x) = x
        encoded = Dense(dims[-1], kernel_initializer=init,
                        name='encoder_%d' % (n_stacks - 1))(x)
        self.encoded = encoded

        x = encoded
        # internal layers in decoder
        for i in range(n_stacks-1, 0, -1):
            x = Dense(dims[i], activation=act,
                      kernel_initializer=init, name='decoder_%d' % i)(x)

        # output
        x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)
        decoded = x
        self.out = decoded

        ae = Model(inputs=input_img, outputs=decoded, name='AE')
        encoder = Model(inputs=input_img, outputs=encoded, name='encoder')
        return ae, encoder

    def train_and_save(self, X,
                       weights_save_name,
                       lr=0.001,
                       batch_size=32,
                       epochs=250,
                       loss='mse'):
        if os.path.exists(weights_save_name):
            print('weights file exists, no need to train pure AE')
        else:
            print(f'AE train_and_save lr: {lr}')
            print(f'AE train_and_save batch_size: {batch_size}')
            print(f'AE train_and_save epochs: {epochs}')

            verbose = self.verbose

            autoencoder, encoder = self.build()

            pretrain_optimizer = Adam(lr=lr)

            autoencoder.compile(optimizer=pretrain_optimizer, loss='mse')

            if not os.path.exists(os.path.dirname(weights_save_name)):
                os.makedirs(os.path.dirname(weights_save_name))

            mcp_save = ModelCheckpoint(weights_save_name,
                                    monitor='loss',
                                    save_best_only=True,
                                    save_weights_only=True,
                                    verbose=verbose,
                                    mode='min')

            hist = autoencoder.fit(X, X,
                                epochs=epochs,
                                batch_size=batch_size,
                                verbose=1,
                                callbacks=[mcp_save])

    def evaluate_quality(self, X_old, y_old, model_save_name):
        if not os.path.exists(model_save_name):
            self.train_and_save(X_old, model_save_name)

        K.clear_session()
        autoencoder, encoder = self.build()
        encoder.load_weights(model_save_name, by_name=True)
        print(f'Load weights from {model_save_name}')
        latent = encoder.predict(X_old)

        best_acc = 0
        best_n_init = 10
        num_classes = len(np.unique(y_old))
        print(f'KMeans k = {num_classes}')

        warnings.filterwarnings('ignore')

        for n_init in range(10, 110, 10):
            kmeans = KMeans(n_clusters=num_classes, n_init=n_init,
                            random_state=42, n_jobs=-1)
            y_pred = kmeans.fit_predict(latent)
            acc = get_cluster_acc(y_old, y_pred)
            print(f'KMeans n_init: {n_init}, acc: {acc}')
            if acc > best_acc:
                best_n_init = best_n_init
                best_acc = acc
        print(f'best accuracy of KMeans on latent data: {best_acc} with n_init {best_n_init}')
        return best_acc


class ContrastiveAE(object):
    def __init__(self, dims, optimizer, lr, verbose=1):
        self.dims = dims
        self.optimizer = optimizer(lr)
        self.verbose = verbose

    def train(self, X_train, y_train,
              lambda_1, batch_size, epochs, similar_ratio, margin,
              weights_save_name, display_interval):
        """Train an autoencoder with standard mse loss + contrastive loss.

        Arguments:
            X_train {numpy.ndarray} -- feature vectors of the training data
            y_train {numpy.ndarray} -- ground-truth labels of the training data
            lambda_1 {float} -- balance factor for the autoencoder reconstruction loss and contrastive loss
            batch_size {int} -- number of samples in each batch (note we only use **half of batch_size**
                                from the training data).
            epochs {int} -- No. of maximum epochs.
            similar_ratio {float} -- ratio of similar samples, use 0.25 for now.
            margin {float} -- the hyper-parameter m.
            weights_save_name {str} -- file path to save the best weights files.
            display_interval {int} -- print traning logs per {display_interval} epoches
        """
        if os.path.exists(weights_save_name):
            print('weights file exists, no need to train contrastive AE')
        else:
            tf.reset_default_graph()

            labels = tf.placeholder(tf.float32, [None])
            lambda_1_tensor = tf.placeholder(tf.float32)
            ae = Autoencoder(self.dims)
            ae_model, encoder_model = ae.build()

            input_ = ae_model.get_input_at(0)

            # add loss function -- for efficiency and not doubling the network's weights, we pass a batch of samples and
            # make the pairs from it at the loss level.
            left_p = tf.convert_to_tensor(list(range(0, int(batch_size / 2))), np.int32)
            right_p = tf.convert_to_tensor(list(range(int(batch_size / 2), batch_size)), np.int32)

            # left_p: indices with all the data in this batch, right_p: half with similar data compared to left_p, half with dissimilar data compared to left_p
            # if batch_size = 16 (but only using 8 samples in this batch):
            # e.g., left_p labels: 1, 2, 4, 8 | 2, 3, 5, 6
            #      right_p labels: 1, 2, 4, 8 | 3, 4, 1, 7
            # check whether labels[left_p] == labels[right_p] for each element
            is_same = tf.cast(tf.equal(tf.gather(labels, left_p), tf.gather(labels, right_p)), tf.float32)
            # NOTE: add a small number like 1e-10 would prevent tf.sqrt() to have 0 values, further leading gradients and loss all NaN.
            # check: https://stackoverflow.com/questions/33712178/tensorflow-nan-bug
            dist = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tf.gather(ae.encoded, left_p), tf.gather(ae.encoded, right_p))), 1) + 1e-10) # ||zi - zj||_2
            contrastive_loss = tf.multiply(is_same, dist) # y_ij = 1 means the same class.
            contrastive_loss = contrastive_loss + tf.multiply((tf.constant(1.0) - is_same), tf.nn.relu(margin - dist))  # as relu(z) = max(0, z)
            contrastive_loss = tf.reduce_mean(contrastive_loss)

            ae_loss = tf.keras.losses.MSE(input_, ae.out) # ae.out equals ae_model(input_)
            ae_loss = tf.reduce_mean(ae_loss)

            # Final loss
            loss = lambda_1 * contrastive_loss + ae_loss

            train_op = self.optimizer.minimize(loss, var_list=tf.trainable_variables())

            # Start training
            with tf.Session(config=config) as sess:
                loss_batch, aux_batch = [], []
                contrastive_loss_batch, ae_loss_batch = [], []

                sess.run(tf.global_variables_initializer())

                min_loss = np.inf

                # epoch training loop
                for epoch in range(epochs):
                    epoch_time = time.time()
                    # split data into batches
                    batch_count, batch_x, batch_y = epoch_batches(X_train, y_train,
                                                                    batch_size,
                                                                    similar_ratio)
                    # batch training loop
                    for b in range(batch_count):
                        print(f'b: {b}')
                        feed_dict = {
                            input_: batch_x[b],
                            labels: batch_y[b],
                            lambda_1_tensor: lambda_1
                        }
                        loss1, _, aux1, contrastive_loss1, ae_loss1, \
                            dist1, encoded1 = sess.run([loss, train_op, is_same, contrastive_loss, ae_loss,
                                                        dist, ae.encoded], feed_dict=feed_dict)

                        print(f'loss1: {loss1},  aux1: {aux1}')
                        print(f'contrastive: {contrastive_loss1}, ae: {ae_loss1}')
                        print(f'epoch-{epoch} dist1[left]: {dist1[0:batch_size // 4]}')
                        print(f'epoch-{epoch} dist1[right]: {dist1[batch_size // 4:]}')

                        loss_batch.append(loss1)
                        aux_batch.append(aux1)
                        contrastive_loss_batch.append(contrastive_loss1)
                        ae_loss_batch.append(ae_loss1)

                    if math.isnan(np.mean(loss_batch)):
                        print('NaN value in loss')

                    # print logs each xxx epoch
                    if epoch % display_interval == 0:
                        current_loss = np.mean(loss_batch)
                        print(f'Epoch {epoch}: loss {current_loss} -- ' + \
                                    f'contrastive {np.mean(contrastive_loss_batch)} -- ' + \
                                    f'ae {np.mean(ae_loss_batch)} -- ' + \
                                    f'pairs {np.mean(np.sum(np.mean(aux_batch)))} : ' + \
                                    f'{np.mean(np.sum(1-np.mean(aux_batch)))} -- ' + \
                                    f'time {time.time() - epoch_time}')
                        loss_batch, aux_batch = [], []
                        contrastive_loss_batch, ae_loss_batch = [], []

                        # save best weights
                        if current_loss < min_loss:
                            print(f'updating best loss from {min_loss} to {current_loss}')
                            min_loss = current_loss
                            ae_model.save_weights(weights_save_name)
