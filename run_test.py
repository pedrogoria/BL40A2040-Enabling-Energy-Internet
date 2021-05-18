import tensorflow as tf
import numpy as np

# definition of constants
delta_t_max = 10  # prediction maximum time (in channel step_time)
batch_train = 32  # batch length
data_path = 'data/complex_0_2C_4096train_512test/'
validation_path = 'data/complex_1_1C_4096train_512test/'
delta_t = -1
n_slices = 32  #
model_path = 'model_residual 2C 4096trains 30epochs 32batch/'

# print('\n ---->  downloading...')
# x_train = np.load(data_path + 'x_train.npy', mmap_mode='r')
# x_test = np.load(data_path + 'x_test.npy', mmap_mode='r')
# y_train = np.load(data_path + 'y_train.npy', mmap_mode='r')
# y_test = np.load(data_path + 'y_test.npy', mmap_mode='r')
# print(" ---->    end \n")

model = tf.saved_model.load(model_path)
loss_object = tf.keras.losses.MeanSquaredError()
test_loss = tf.keras.metrics.Mean(name='test_loss')


@tf.function
def test_step(frames, label):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(frames, training=False)
    if delta_t == -1:
        t_loss = loss_object(label, predictions)
    else:
        t_loss = loss_object(label[:, :, :, delta_t], predictions[:, :, :, delta_t])

    test_loss(t_loss)
    # test_accuracy(label, predictions)
    return


print("--> %d " % 0, end='')
test_loss.reset_states()
x_test = np.load(data_path + '/x_train.npy', mmap_mode='r')
y_test = np.load(data_path + '/y_train.npy', mmap_mode='r')
slice_test = int(x_test.shape[0] / n_slices)

for index in range(n_slices):
    test_ds = tf.data.Dataset.from_tensor_slices((x_test[index * slice_test:(index + 1) * slice_test],
                                                  y_test[index * slice_test:(index + 1) * slice_test]
                                                  .reshape(slice_test, -1, 1, delta_t_max))).batch(batch_train)

    for frame, labels in test_ds:
        test_step(frame, labels)

    print("--> %d " % (index + 1), end='')
train_loss = test_loss.result().numpy()

print("\n\n--> %d " % 0, end='')
test_loss.reset_states()
x_test = np.load(data_path + '/x_test.npy', mmap_mode='r')
y_test = np.load(data_path + '/y_test.npy', mmap_mode='r')
slice_test = int(x_test.shape[0] / n_slices)

for index in range(n_slices):
    test_ds = tf.data.Dataset.from_tensor_slices((x_test[index * slice_test:(index + 1) * slice_test],
                                                  y_test[index * slice_test:(index + 1) * slice_test]
                                                  .reshape(slice_test, -1, 1, delta_t_max))).batch(batch_train)

    for frame, labels in test_ds:
        test_step(frame, labels)

    print("--> %d " % (index + 1), end='')

test_loss_v = test_loss.result().numpy()

print("\n\n--> %d " % 0, end='')
test_loss.reset_states()
x_test = np.load(validation_path + '/x_train.npy', mmap_mode='r')
y_test = np.load(validation_path + '/y_train.npy', mmap_mode='r')
slice_test = int(x_test.shape[0] / n_slices)

for index in range(n_slices):
    test_ds = tf.data.Dataset.from_tensor_slices((x_test[index * slice_test:(index + 1) * slice_test],
                                                  y_test[index * slice_test:(index + 1) * slice_test]
                                                  .reshape(slice_test, -1, 1, delta_t_max))).batch(batch_train)

    for frame, labels in test_ds:
        test_step(frame, labels)

    print("--> %d " % (index + 1), end='')

template = ' For delta_t = {} -> \n Loss: {}, Test Loss: {}, Validation Loss: {}'
print('\n', template.format((delta_t + 1), train_loss, test_loss_v, test_loss.result()))
# Loss: 1.09846830368042, Test Loss: 0.12582343816757202, Validation Loss: 0.2662394046783447
