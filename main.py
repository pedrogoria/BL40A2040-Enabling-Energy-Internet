from abc import ABC
from Functions import *
from tensorflow.keras.layers import Conv2D, LSTM, Dense, Flatten
from tensorflow.keras import Model
import os as os

# solution for tensorflow error (cuda's version < 11.0)
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# definition of constants
delta_t_max = 10  # prediction maximum time (in channel step_time)
batch_train = 64  # batch length
n_sim_channel = 1  # number of simulated channels
data_path = 'data/power_1_1C_4096train_1024test/'
n_slices = 16  #
epochs = 3
model_path = 'model_0/'
power = True  # use False to complex inputs
frame_to_train = [4096, None, 64, 1]  # [frames number, None, temporal samples, 1]
frame_to_test = [512, None, 64, 1]  # [frames number, None, temporal samples, 1]

if not os.path.exists(data_path):
    os.makedirs(data_path)
    print('\n \n running Simulation Channel')

    channel = sc.Channel(scatterers=3)

    generating_samples_for_the_training(channel=channel, file_path=data_path,
                                        n_sim_channel=n_sim_channel,
                                        frame_form_train=frame_to_train,
                                        frame_form_test=frame_to_test,
                                        delta_t_max=delta_t_max,
                                        dtype_save='float32', dtype='float32')

    t = np.arange(0, 100 * channel.step_time, channel.step_time)
    x_train = np.load(data_path + 'x_train.npy', mmap_mode='r')
    x_test = np.load(data_path + 'x_test.npy', mmap_mode='r')
    y_train = np.load(data_path + 'y_train.npy', mmap_mode='r')
    y_test = np.load(data_path + 'y_test.npy', mmap_mode='r')
else:
    print('\n ---->  downloading...')
    x_train = np.load(data_path + 'x_train.npy', mmap_mode='r')
    x_test = np.load(data_path + 'x_test.npy', mmap_mode='r')
    y_train = np.load(data_path + 'y_train.npy', mmap_mode='r')
    y_test = np.load(data_path + 'y_test.npy', mmap_mode='r')
    print(" ---->    end \n")

if not os.path.exists(model_path):
    os.makedirs(model_path)

assert delta_t_max == y_train.shape[2], 'error in parameter value: delta_t_max '


# train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train[:, :, 0])).batch(batch_train)
# # shuffle(10000).
# test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test[:, :, 0])).batch(batch_train)


# Create an instance of the model
# definition Class
class MyModel(Model):
    def __init__(self, power_input=False):
        super(MyModel, self).__init__()
        self.power = power_input
        self.conv0 = Conv2D(6, kernel_size=(5, 4), dilation_rate=(1, 1), padding="same", activation='tanh')
        self.conv1 = Conv2D(12, kernel_size=(5, 4), dilation_rate=(1, 4), padding="same", activation='tanh')
        self.conv2 = Conv2D(12, kernel_size=(5, 4), dilation_rate=(1, 16), padding="same", activation='tanh')
        self.conv3 = Conv2D(6, kernel_size=(5, 4), dilation_rate=(1, 64), padding="same", activation='tanh')
        self.conv4 = Conv2D(delta_t_max, kernel_size=(1, 64), dilation_rate=(1, 1), activation='tanh')
        # self.flatten = Flatten()
        self.d1 = Dense(delta_t_max)
        # self.d1 = Dense(x_train.shape[1])

    # @tf.function
    def call(self, x, training=False, **ops):
        if training:
            if self.power:
                x = self.conv0(x) + tf.concat([x, x, x, x, x, x], axis=3)
            else:
                x = self.conv0(x) + tf.concat([x, x, x], axis=3)
            x = self.conv1(x) + tf.concat([x, x], axis=3)
            x = self.conv2(x) + x
            x = self.conv3(x)
            x = self.conv4(x)
            return self.d1(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            return self.d1(x)
        # x = self.flatten(x)
        # x = self.d1(x)
        # return self.d1(x)


# class MyModel(Model):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.d0 = Dense(1)
#         self.LSTM0 = LSTM(64)
#         self.d1 = Dense(delta_t_max*256)
#
#     # @tf.function
#     def call(self, x, training=None, mask=None, **ops):
#         x = tf.reshape(self.d0(x), shape=(-1, 256, 64))
#         x = self.LSTM0(x)
#         return tf.reshape(self.d1(x), shape=(-1, 256, 1, delta_t_max))


model = MyModel(power_input=power)
loss_object = tf.keras.losses.MeanSquaredError()
# loss_object = tf.keras.losses.KLDivergence()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.03)
ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
manager = tf.train.CheckpointManager(ckpt, model_path + '/tf_ckpts', max_to_keep=30)

train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.MeanSquaredError(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')


# test_accuracy = tf.keras.metrics.MeanSquaredError(name='test_accuracy')


@tf.function
def train_step(frames=0, label=0):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(frames, training=True)
        loss = loss_object(label, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    # train_accuracy(label, predictions)
    return


@tf.function
def test_step(frames, label):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(frames, training=False)
    t_loss = loss_object(label, predictions)

    test_loss(t_loss)
    # test_accuracy(label, predictions)
    return


ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
else:
    print("Initializing from scratch.")
slice_train = int(x_train.shape[0] / n_slices)
slice_test = int(x_test.shape[0] / n_slices)

hist_loss_train = []
hist_loss_test = []

# int(ckpt.step.numpy())
for epoch in range(epochs):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    # train_accuracy.reset_states()
    test_loss.reset_states()
    # test_accuracy.reset_states()
    print("--> %d " % 0, end='')
    for index in range(n_slices):
        train_ds = tf.data.Dataset.from_tensor_slices((x_train[index * slice_train:(index + 1) * slice_train],
                                                       y_train[index * slice_train:(index + 1) * slice_train]
                                                       .reshape(slice_train, -1, 1, delta_t_max))).batch(batch_train)
        # shuffle(10000).
        test_ds = tf.data.Dataset.from_tensor_slices((x_test[index * slice_test:(index + 1) * slice_test],
                                                      y_test[index * slice_test:(index + 1) * slice_test]
                                                      .reshape(slice_test, -1, 1, delta_t_max))).batch(batch_train)

        for frame, labels in train_ds:
            train_step(frame, labels)

        for frame, labels in test_ds:
            test_step(frame, labels)

        print("--> %d " % (index + 1), end='')
        hist_loss_train.append(train_loss.result().numpy())
        hist_loss_test.append(test_loss.result().numpy())
    # template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    template = 'Epoch {}, Loss: {}, Test Loss: {}'
    print('\n', template.format(epoch + 1,
                                train_loss.result(),
                                # train_accuracy.result() * 100,
                                test_loss.result()))
    # test_accuracy.result() * 100))
    ckpt.step.assign_add(1)
    save_path = manager.save()

tf.saved_model.save(model, model_path)
np.save(model_path + 'hist_loss_train.npy', np.array(hist_loss_train))
np.save(model_path + 'hist_loss_test.npy', np.array(hist_loss_test))
# imported = tf.saved_model.load(model_path)
