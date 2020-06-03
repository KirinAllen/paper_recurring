import tensorflow as tf
import input_data_Tri_training
import os
import numpy as np
from utils import flat, max_pool_2x2, run_layer, run_transpose_layer
import csv
import matplotlib.pyplot as plt


layers = [{"type": "dense", "kernel_shape": [784, 1000]},
          {"type": "dense", "kernel_shape": [1000, 500]},
          {"type": "dense", "kernel_shape": [500, 250]},
          {"type": "dense", "kernel_shape": [250, 250]},
          {"type": "dense", "kernel_shape": [250, 250]},
          {"type": "dense", "kernel_shape": [250, 10]}]

denoising_cost = np.ones(len(layers) + 1) * 0.1
denoising_cost[0] = 100
denoising_cost[1] = 10

checkpoint_dir = "checkpoints/Ladder_tri_training/checkpoints_900/"

image_shape = [784]

num_epochs = 150
starter_learning_rate = 0.00001
decay_after = 10  # epoch after which to begin learning rate decay
batch_size = 100
num_labeled = 900

noise_std = 0.03  # scaling factor for noise used in corrupted encoder

# ==================================================================================================
feedforward_inputs = {}
L = len(layers)  # number of layers
tf.reset_default_graph()

feedforward_inputs[0] = tf.placeholder(tf.float32, shape=(None, np.prod(image_shape)), name="FFI_0")
feedforward_inputs[1] = tf.placeholder(tf.float32, shape=(None, np.prod(image_shape)), name="FFI_1")
feedforward_inputs[2] = tf.placeholder(tf.float32, shape=(None, np.prod(image_shape)), name="FFI_2")

autoencoder_inputs = tf.placeholder(tf.float32, shape=(None, np.prod(image_shape)), name="AEI")

outputs = tf.placeholder(tf.float32)  # labels for the labeled images
training = tf.placeholder(tf.bool)  # If training or not

FFI_0 = tf.reshape(feedforward_inputs[0], [-1] + image_shape)
FFI_1 = tf.reshape(feedforward_inputs[1], [-1] + image_shape)
FFI_2 = tf.reshape(feedforward_inputs[2], [-1] + image_shape)

AEI = tf.reshape(autoencoder_inputs, [-1] + image_shape)

# ==================================================================================================
# Batch normalization functions
ewma = tf.train.ExponentialMovingAverage(decay=0.99)  # to calculate the moving averages of mean and variance
bn_assigns = []  # this list stores the updates to be made to average mean and variance


def update_batch_normalization(batch, output_name="bn", scope_name="BN"):
    dim = len(batch.get_shape().as_list())
    mean, var = tf.nn.moments(batch, axes=list(range(0, dim - 1)))
    # Function to be used during the learning phase. Normalize the batch and update running mean and variance.
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        running_mean = tf.get_variable("running_mean", mean.shape, initializer=tf.constant_initializer(0))
        running_var = tf.get_variable("running_var", mean.shape, initializer=tf.constant_initializer(1))

    assign_mean = running_mean.assign(mean)
    assign_var = running_var.assign(var)
    bn_assigns.append(ewma.apply([running_mean, running_var]))

    with tf.control_dependencies([assign_mean, assign_var]):
        z = (batch - mean) / tf.sqrt(var + 1e-10)
        return tf.identity(z, name=output_name)


def batch_normalization(batch, mean=None, var=None, output_name="bn"):
    dim = len(batch.get_shape().as_list())
    mean, var = tf.nn.moments(batch, axes=list(range(0, dim - 1)))
    if mean is None or var is None:
        mean, var = tf.nn.moments(batch, axes=[0])
    z = (batch - mean) / tf.sqrt(var + tf.constant(1e-10))
    return tf.identity(z, name=output_name)


# ______________________________________________________________________________________
# Encoder

def encoder_bloc(h, layer_spec, noise_std, update_BN, activation):
    # Run the layer
    z_pre = run_layer(h, layer_spec, output_name="z_pre")

    # Compute mean and variance of z_pre (to be used in the decoder)
    dim = len(z_pre.get_shape().as_list())
    mean, var = tf.nn.moments(z_pre, axes=list(range(0, dim - 1)))
    mean = tf.identity(mean, name="mean")
    var = tf.identity(var, name="var")

    # tf.identity在计算图内部创建了两个节点，send / recv节点，用来发送和接受两个变量，
    # 如果两个变量在不同的设备上，比如 CPU 和 GPU，那么将会复制变量，如果在一个设备上，将会只是一个引用
    # 返回一个和输入的 tensor 大小和数值都一样的 tensor ,类似于 y=x 操作

    # Batch normalization
    def training_batch_norm():
        if update_BN:
            z = update_batch_normalization(z_pre)
        else:
            z = batch_normalization(z_pre)
        return z

    def eval_batch_norm():
        with tf.variable_scope("BN", reuse=tf.AUTO_REUSE):
            mean = ewma.average(tf.get_variable("running_mean", shape=z_pre.shape[-1]))
            var = ewma.average(tf.get_variable("running_var", shape=z_pre.shape[-1]))
        z = batch_normalization(z_pre, mean, var)
        return z

    # Perform batch norm depending to the phase (training or testing)
    z = tf.cond(training, training_batch_norm, eval_batch_norm)
    z += tf.random_normal(tf.shape(z)) * noise_std
    z = tf.identity(z, name="z")

    # Center and scale plus activation
    size = z.get_shape().as_list()[-1]
    beta = tf.get_variable("beta", [size], initializer=tf.constant_initializer(0))
    gamma = tf.get_variable("gamma", [size], initializer=tf.constant_initializer(1))

    h = activation(z * gamma + beta)
    return tf.identity(h, name="h")


def encoder(h, noise_std, update_BN):
    # Perform encoding for each layer
    h += tf.random_normal(tf.shape(h)) * noise_std
    h = tf.identity(h, "h0")

    for i, layer_spec in enumerate(layers):
        with tf.variable_scope("encoder_bloc_" + str(i + 1), reuse=tf.AUTO_REUSE):
            # Create an encoder bloc if the layer type is dense or conv2d
            if layer_spec["type"] == "flat":
                h = flat(h, output_name="h")
            elif layer_spec["type"] == "max_pool_2x2":
                h = max_pool_2x2(h, output_name="h")
            else:
                if i == L - 1:
                    activation = tf.nn.softmax  # Only for the last layer
                else:
                    activation = tf.nn.relu
                h = encoder_bloc(h, layer_spec, noise_std, update_BN=update_BN, activation=activation)

    y = tf.identity(h, name="y")
    return y


# __________________________________________________________________________________________________________
# Decoder

def g_gauss(z_c, u, output_name="z_est", scope_name="denoising_func"):
    #  gaussian denoising function proposed in the original paper
    size = u.get_shape().as_list()[-1]
    wi = lambda inits, name: tf.Variable(inits * tf.ones([size]), name=name)
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        a1 = wi(0., 'a1')
        a2 = wi(1., 'a2')
        a3 = wi(0., 'a3')
        a4 = wi(0., 'a4')
        a5 = wi(0., 'a5')

        a6 = wi(0., 'a6')
        a7 = wi(1., 'a7')
        a8 = wi(0., 'a8')
        a9 = wi(0., 'a9')
        a10 = wi(0., 'a10')

        mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
        v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10

        z_est = (z_c - mu) * v + mu
    return tf.identity(z_est, name=output_name)


def decoder_bloc(u, z_corr, mean, var, layer_spec=None):
    z_est = g_gauss(z_corr, u)
    z_est_BN = (z_est - mean) / tf.sqrt(var + tf.constant(1e-10))
    z_est_BN = tf.identity(z_est_BN, name="z_est_BN")

    if layer_spec is not None:
        u = run_transpose_layer(z_est, layer_spec)
        u = batch_normalization(u, output_name="u")

    return u, z_est_BN


# ========================================================================================================
# Graph building
print("===  Building graph ===")

# Encoder
FF_y = {}
FF_y_corr = {}
with tf.name_scope("FF_clean"):
    FF_y[0] = encoder(FFI_0, 0, update_BN=False)  # output of the clean encoder. Used for prediction
    FF_y[1] = encoder(FFI_1, 0, update_BN=False)
    FF_y[2] = encoder(FFI_2, 0, update_BN=False)
with tf.name_scope("FF_corrupted"):
    FF_y_corr[0] = encoder(FFI_0, noise_std, update_BN=False)  # output of the corrupted encoder. Used for training.
    FF_y_corr[1] = encoder(FFI_1, noise_std, update_BN=False)
    FF_y_corr[2] = encoder(FFI_2, noise_std, update_BN=False)

with tf.name_scope("AE_clean"):
    AE_y = encoder(AEI, 0, update_BN=True)  # Clean encoding of unlabeled images
with tf.name_scope("AE_corrupted"):
    AE_y_corr = encoder(AEI, noise_std, update_BN=False)  # corrupted encoding of unlabeled images

# __________________________________________________________________________________________________________
# Decoder
# Function used to get a tensor from encoder
get_tensor = lambda input_name, num_encoder_bloc, name_tensor: tf.get_default_graph().get_tensor_by_name(
        input_name + "/encoder_bloc_" + str(num_encoder_bloc) + "/" + name_tensor + ":0")

d_cost = []
u = batch_normalization(AE_y_corr, output_name="u_L")
for i in range(L, 0, -1):
    layer_spec = layers[i - 1]

    with tf.variable_scope("decoder_bloc_" + str(i), reuse=tf.AUTO_REUSE):
        # if the layer is max pooling or "flat", the transposed layer is run without creating a decoder bloc.
        if layer_spec["type"] in ["max_pool_2x2", "flat"]:
            h = get_tensor("AE_corrupted", i - 1, "h")
            output_shape = tf.shape(h)
            u = run_transpose_layer(u, layer_spec, output_shape=output_shape)
        else:
            z_corr, z = [get_tensor("AE_corrupted", i, "z"), get_tensor("AE_clean", i, "z")]
            mean, var = [get_tensor("AE_clean", i, "mean"), get_tensor("AE_clean", i, "var")]

            u, z_est_BN = decoder_bloc(u, z_corr, mean, var, layer_spec=layer_spec)
            d_cost.append((tf.reduce_mean(tf.square(z_est_BN - z))) * denoising_cost[i])

# last decoding step
with tf.variable_scope("decoder_bloc_0", reuse=tf.AUTO_REUSE):
    z_corr = tf.get_default_graph().get_tensor_by_name("AE_corrupted/h0:0")
    z = tf.get_default_graph().get_tensor_by_name("AE_clean/h0:0")
    mean, var = tf.constant(0.0), tf.constant(1.0)

    u, z_est_BN = decoder_bloc(u, z_corr, mean, var)
    d_cost.append((tf.reduce_mean(tf.square(z_est_BN - z))) * denoising_cost[0])

corr_cost = {}
F_acc = {}
F_correct_prediction = {}
F_op = {}
S_loss = {}
u_cost = tf.add_n(d_cost)  # decoding cost

corr_cost[0] = -tf.reduce_mean(tf.reduce_sum(outputs * tf.log(FF_y_corr[0]), 1))  # supervised cost
corr_cost[1] = -tf.reduce_mean(tf.reduce_sum(outputs * tf.log(FF_y_corr[1]), 1))
corr_cost[2] = -tf.reduce_mean(tf.reduce_sum(outputs * tf.log(FF_y_corr[2]), 1))

corr_pred_cost = corr_cost[0] + corr_cost[1] + corr_cost[2]

# Classifier
F_correct_prediction[0] = tf.equal(tf.argmax(FF_y[0], 1), tf.argmax(outputs, 1))  # number of correct predictions
F_acc[0] = tf.reduce_mean(tf.cast(F_correct_prediction[0], "float")) * tf.constant(100.0)


F_correct_prediction[1] = tf.equal(tf.argmax(FF_y[1], 1), tf.argmax(outputs, 1))
F_acc[1] = tf.reduce_mean(tf.cast(F_correct_prediction[1], "float")) * tf.constant(100.0)

F_correct_prediction[2] = tf.equal(tf.argmax(FF_y[2], 1), tf.argmax(outputs, 1))
F_acc[2] = tf.reduce_mean(tf.cast(F_correct_prediction[2], "float")) * tf.constant(100.0)

# Optimization setting
learning_rate = tf.Variable(starter_learning_rate, trainable=False)
loss = corr_pred_cost + u_cost  # total cost
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

S_loss[0] = corr_cost[0] + u_cost
S_loss[1] = corr_cost[1] + u_cost
S_loss[2] = corr_cost[2] + u_cost
F_op[0] = tf.train.AdamOptimizer(learning_rate).minimize(corr_cost[0])
F_op[1] = tf.train.AdamOptimizer(learning_rate).minimize(corr_cost[1])
F_op[2] = tf.train.AdamOptimizer(learning_rate).minimize(corr_cost[2])

bn_updates = tf.group(*bn_assigns)
with tf.control_dependencies([train_step]):
    train_step = tf.group(bn_updates)

with tf.control_dependencies([F_op[0]]):
    F_op[0] = tf.group(bn_updates)

with tf.control_dependencies([F_op[1]]):
    F_op[1] = tf.group(bn_updates)

with tf.control_dependencies([F_op[2]]):
    F_op[2] = tf.group(bn_updates)

n = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
print(str(n) + " trainable parameters")

# ========================================================================================================================
# Learning phase
print("===  Loading Data ===")
data = input_data_Tri_training.read_data_sets("MNIST_data", n_labeled=num_labeled, one_hot=True)
num_examples = data.train.unlabeled_ds.images.shape[0]
num_iter = (num_examples // batch_size) * num_epochs  # number of loop iterations

print("===  Starting Session ===")
sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

print("=== Training ===")
initial_accuracy = sess.run(F_acc[0], feed_dict={feedforward_inputs[0]: data.test.images,
                                                 outputs: data.test.labels,
                                                 training: False})
print("Initial Accuracy: ", initial_accuracy, "%")

i_iter = 0
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    epoch_n = int(ckpt.model_checkpoint_path.split('-')[1])
    i_iter = (epoch_n + 1) * (num_examples // batch_size)
    print("Restored Epoch ", epoch_n)
else:
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    init = tf.global_variables_initializer()
    sess.run(init)

for i in range(i_iter, num_iter):

    labeled_images_0, labels_0 = data.train.S0.next_batch(batch_size)
    sess.run(F_op[0], feed_dict={feedforward_inputs[0]: labeled_images_0,
                                 outputs: labels_0,
                                 training: True})
    labeled_images_1, labels_1,  = data.train.S1.next_batch(batch_size)
    sess.run(F_op[1], feed_dict={feedforward_inputs[1]: labeled_images_1,
                                 outputs: labels_1,
                                 training: True})
    labeled_images_2, labels_2,  = data.train.S2.next_batch(batch_size)
    sess.run(F_op[2], feed_dict={feedforward_inputs[2]: labeled_images_2,
                                 outputs: labels_2,
                                 training: True})

    if (i > 1) and ((i + 1) % (num_iter / num_epochs) == 0):
        epoch_n = i // (num_examples // batch_size)

        train_loss_0 = sess.run(corr_cost[0], feed_dict={feedforward_inputs[0]: labeled_images_0,
                                                         outputs: labels_0,
                                                         training: False})
        train_loss_1 = sess.run(corr_cost[1], feed_dict={feedforward_inputs[1]: labeled_images_1,
                                                         outputs: labels_1,
                                                         training: False})
        train_loss_2 = sess.run(corr_cost[2], feed_dict={feedforward_inputs[2]: labeled_images_2,
                                                         outputs: labels_2,
                                                         training: False})

        test_acc = sess.run(F_acc[0], feed_dict={feedforward_inputs[0]: data.test.images,
                                                 outputs: data.test.labels,
                                                 training: False})

        print("\nEpoch ", str(int(epoch_n + 0.1)),
              ": loss_0 = {:.5f}, loss_1 = {:.3}, loss_2 = {:.3}, test accuracy = {:.3}".format(train_loss_0,
                                                                                                train_loss_1,
                                                                                                train_loss_2,
                                                                                                test_acc))
        if (epoch_n + 1) >= decay_after:
            ratio = 1.0 * (num_epochs - (epoch_n + 1))
            ratio = max(0, ratio / (num_epochs - decay_after))
            sess.run(learning_rate.assign(starter_learning_rate * ratio))
        saver.save(sess, checkpoint_dir + 'model.ckpt', epoch_n)

final_accuracy = sess.run(F_acc[0], feed_dict={feedforward_inputs[0]: data.test.images,
                                               outputs: data.test.labels,
                                               training: False})
print("Accuracy_0: ", final_accuracy, "%")

a0 = sess.run((tf.argmax(FF_y[0], 1)),
              feed_dict={feedforward_inputs[0]: data.train.S0.images[15:30], training: False})
a1 = sess.run((tf.argmax(FF_y[1], 1)),
              feed_dict={feedforward_inputs[1]: data.train.S0.images[15:30], training: False})
a2 = sess.run((tf.argmax(FF_y[2], 1)),
              feed_dict={feedforward_inputs[2]: data.train.S0.images[15:30], training: False})
print(a0, a1, a2)


def measure_error(x, y, j, k):
    a = np.argwhere(y == 1)
    list = [x[1] for x in a]
    j_pred = sess.run((tf.argmax(FF_y_corr[j], 1)), feed_dict={feedforward_inputs[j]: x, training: False})
    k_pred = sess.run((tf.argmax(FF_y_corr[k], 1)), feed_dict={feedforward_inputs[k]: x, training: False})
    wrong_index = np.logical_and(j_pred != list, k_pred == j_pred)
    a = sum(wrong_index)
    b = sum(j_pred == k_pred)
    sum(wrong_index) / sum(j_pred == k_pred)
    return a / b

train_data_labels = {}
train_data_images = {}
e_prime = [0.05] * 3
l_prime = [0] * 3
e = [0] * 3
update = [False] * 3
Li_X, Li_y = [[]] * 3, [[]] * 3
# 保存代理标记的数据
U_X = data.train.unlabeled_ds.images
L_X = data.train.labeled_ds.images
L_y = data.train.labeled_ds.labels
num_examples = data.train.labeled_ds.num_examples
indices = np.arange(num_examples)
shuffled_indices = np.random.permutation(indices)

L_X = L_X[shuffled_indices]
L_y = L_y[shuffled_indices]

train_data_images[0] = data.train.S0.images
train_data_images[1] = data.train.S1.images
train_data_images[2] = data.train.S2.images

train_data_labels[0] = data.train.S0.labels
train_data_labels[1] = data.train.S1.labels
train_data_labels[2] = data.train.S2.labels
improve = True
n = 0

while improve:
    n += 1
    # count iterations
    for i in range(3):
        j, k = np.delete(np.array([0, 1, 2]), i)
        update[i] = False
        e[i] = measure_error(data.test.images, data.test.labels, j, k)
        print(e[i])
        if e[i] < e_prime[i]:
            U_y_j = sess.run((tf.argmax(FF_y_corr[j], 1)), feed_dict={feedforward_inputs[j]: U_X,
                                                                      training: False})
            U_y_k = sess.run((tf.argmax(FF_y_corr[k], 1)), feed_dict={feedforward_inputs[k]: U_X,
                                                                      training: False})
            num = sum(U_y_k == U_y_j)
            indices = np.arange(num)
            Li_y[i] = U_y_j[U_y_j == U_y_k]
            Li_X[i] = U_X[U_y_j == U_y_k]

            if l_prime[i] == 0:
                # no updated before
                l_prime[i] = int(e[i] / (e_prime[i] - e[i]) + 1)
            if l_prime[i] < len(Li_y[i]):
                if e[i] * len(Li_y[i]) < e_prime[i] * l_prime[i]:
                    update[i] = True
                elif l_prime[i] > e[i] / (e_prime[i] - e[i]):
                    L_index = np.random.choice(len(Li_y[i]), int(e_prime[i] * l_prime[i] / e[i] - 1))
                    # int(e_prime[i] * l_prime[i] / e[i] - 1)
                    # 从代理标记数据中重采样
                    Li_X[i], Li_y[i] = Li_X[i][L_index], Li_y[i][L_index]
                    update[i] = True

    for i in range(3):
        if update[i]:
            print(Li_X[i].shape)
            L_x = np.append(train_data_images[i], Li_X[i], axis=0)
            print("New_x: ", L_x.shape)
            c = np.argwhere(train_data_labels[i] == 1)
            d = [x[1] for x in c]
            L_y = np.append(d, Li_y[i], axis=0).tolist()
            batch_size_ = tf.size(L_y)
            labels = tf.to_int32(tf.expand_dims(L_y, 1), name='ToInt32')
            indices = tf.expand_dims(tf.range(0, batch_size_, 1), 1)
            concated = tf.concat([indices, labels], 1)
            L_Y = tf.sparse_to_dense(concated, tf.stack([batch_size_, 10]), 1.0, 0.0)
            L_Y = sess.run(L_Y)

            print("New_y: ", L_Y.shape)
            # train the classifier on integrated dataset
            num_examples = L_x.shape[0]
            m = np.arange(num_examples)
            shuffled_m = np.random.permutation(m)
            L_x = L_x[shuffled_m]
            L_Y = L_Y[shuffled_m]

            for i_ in range(num_iter):
                labeled_images, labels, unlabeled_images = data.train.next_batch(batch_size)
                sess.run([F_op[i]], feed_dict={feedforward_inputs[i]: L_x, outputs: L_Y,
                                               autoencoder_inputs: unlabeled_images,
                                               training: True})
            print(sess.run(F_acc[i], feed_dict={feedforward_inputs[i]: data.test.images,
                                                outputs: data.test.labels,
                                                training: False}))

            e_prime[i] = e[i]
            l_prime[i] = len(Li_y[i])

    if update == [False] * 3:
        improve = False
        # if no classifier was updated, no improvement
        for _i_ in range(i_iter, num_iter):
            labeled_images, labels, unlabeled_images = data.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={feedforward_inputs[0]: data.train.S0.images,
                                            outputs: data.train.S0.labels,
                                            feedforward_inputs[1]: data.train.S1.images,
                                            outputs: data.train.S1.labels,
                                            feedforward_inputs[2]: data.train.S2.images,
                                            outputs: data.train.S2.labels,
                                            autoencoder_inputs: unlabeled_images,
                                            training: True})
            if (_i_ > 1) and ((_i_ + 1) % (num_iter / num_epochs) == 0):
                epoch_n_ = _i_ // (num_examples // batch_size)
                train_loss = sess.run(loss, feed_dict={feedforward_inputs[0]: data.train.S0.images,
                                                       outputs: data.train.S0.labels,
                                                       feedforward_inputs[1]: data.train.S1.images,
                                                       outputs: data.train.S1.labels,
                                                       feedforward_inputs[2]: data.train.S2.images,
                                                       outputs: data.train.S2.labels,
                                                       autoencoder_inputs: unlabeled_images,
                                                       training: False})

                test_acc_ = sess.run(F_acc[0], feed_dict={feedforward_inputs[0]: data.test.images,
                                                          outputs: data.test.labels,
                                                          training: False})

                print("\nEpoch ", str(int(epoch_n_ + 0.1)),
                      ": train loss = {:.3}, test acc = {:.3}".format(train_loss, test_acc_))

                if (epoch_n_ + 1) >= decay_after:
                    ratio_ = 1.0 * (num_epochs - (epoch_n_ + 1))
                    ratio_ = max(0, ratio_ / (num_epochs - decay_after))
                    sess.run(learning_rate.assign(starter_learning_rate * ratio_))
                with open('Tri_acc_900', 'a') as Tri_acc_900:
                    train_log_w = csv.writer(Tri_acc_900)
                    acc = sess.run(F_acc[0], feed_dict={feedforward_inputs[0]: data.test.images,
                                                        outputs: data.test.labels,
                                                        training: False})
                    log_i = [epoch_n_, acc]
                    train_log_w.writerow(log_i)

        final_accuracy = sess.run(F_acc[0], feed_dict={feedforward_inputs[0]: data.test.images,
                                                       outputs: data.test.labels,
                                                       training: False})
        print("Final Accuracy: ", final_accuracy, "%")

