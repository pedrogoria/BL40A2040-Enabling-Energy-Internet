import simulation_channel as sc
import tensorflow as tf
import numpy as np


# import matplotlib.pyplot as plt
# from scipy.fftpack import fft, fftfreq
# from pandas import DataFrame
# from pandas import concat


# creating functions
def generate_frames(c=sc.Channel(), update_channel=True, frame_form=(1, None, 64, 1), f_pair=True, delta=0, **ops):
    if update_channel:
        p, a, f = c(n_steps=frame_form[2], f_pair=f_pair, **ops)
        p = p[tf.newaxis, ..., tf.newaxis]
        a = a[tf.newaxis, ..., tf.newaxis]
        for k in range(frame_form[0] - 1 + delta):
            p_aux, a_aux, _ = c(n_steps=1, f_pair=f_pair, **ops)
            p_aux = p_aux[tf.newaxis, ..., tf.newaxis]
            a_aux = a_aux[tf.newaxis, ..., tf.newaxis]
            p = np.concatenate((p, np.concatenate((p[k:k + 1, :, 1:, :], p_aux), axis=2)))
            a = np.concatenate((a, np.concatenate((a[k:k + 1, :, 1:, :], a_aux), axis=2)))
    else:
        p, a, f = c(n_steps=0, f_pair=f_pair, **ops)
        p = p[tf.newaxis, ..., tf.newaxis]
        a = a[tf.newaxis, ..., tf.newaxis]

    return p, a, f


def convert_p2c(modulo, angle):
    x = modulo * np.exp(1j * angle)
    x_r = np.real(x)
    x_i = np.imag(x)
    return np.concatenate((x_r, x_i), axis=3)


def generating_samples_for_the_training(channel=sc.Channel(), n_sim_channel=1, frame_form_train=(512, None, 64, 1),
                                        frame_form_test=(128, None, 64, 1), file_path='/data/',
                                        delta_t_max=10, dtype_save='float64', dtype='float64'):
    x_train1, a_train1, f = generate_frames(c=channel, frame_form=frame_form_train, delta=delta_t_max, period=20e-6,
                                            dtype=dtype)
    x_test1, a_test1, _ = generate_frames(c=channel, frame_form=frame_form_test, delta=delta_t_max, period=20e-6,
                                          dtype=dtype)

    y_train1 = np.array(x_train1[delta_t_max:, :, -delta_t_max:, 0], dtype=dtype_save)
    y_test1 = np.array(x_test1[delta_t_max:, :, -delta_t_max:, 0], dtype=dtype_save)

    # convert from power (modulo) to complex
    # x_train1 = convert_p2c(x_train1, a_train1)
    # x_test1 = convert_p2c(x_test1, a_test1)

    x_train1 = np.array(x_train1[:-delta_t_max], dtype=dtype_save)
    x_test1 = np.array(x_test1[:-delta_t_max], dtype=dtype_save)

    np.save(file_path + 'x_test.npy',
            np.zeros((int(n_sim_channel * x_test1.shape[0]), x_test1.shape[1], x_test1.shape[2],
                      x_test1.shape[3]), dtype=dtype_save))
    np.save(file_path + 'x_train.npy', np.zeros((int(n_sim_channel * x_train1.shape[0]), x_train1.shape[1],
                                                 x_train1.shape[2], x_train1.shape[3]), dtype=dtype_save))
    np.save(file_path + 'f.npy', f)
    np.save(file_path + 'y_train.npy', np.zeros((int(n_sim_channel * y_train1.shape[0]), y_train1.shape[1],
                                                 y_train1.shape[2]), dtype=dtype_save))
    np.save(file_path + 'y_test.npy', np.zeros((int(n_sim_channel * y_test1.shape[0]), y_test1.shape[1],
                                                y_test1.shape[2]), dtype=dtype_save))

    x_train = np.load(file_path + 'x_train.npy', mmap_mode='r+')
    x_test = np.load(file_path + 'x_test.npy', mmap_mode='r+')
    y_train = np.load(file_path + 'y_train.npy', mmap_mode='r+')
    y_test = np.load(file_path + 'y_test.npy', mmap_mode='r+')

    x_train[0:frame_form_train[0]] = x_train1
    x_test[0:frame_form_test[0]] = x_test1
    y_train[0:frame_form_train[0]] = y_train1
    y_test[0:frame_form_test[0]] = y_test1

    print("--> %d " % 0, end='')

    # generating samples for the training
    for k in range(1, n_sim_channel):
        channel = sc.Channel()

        x_train1, a_train1, _ = generate_frames(c=channel, frame_form=frame_form_train, delta=delta_t_max,
                                                period=20e-6, dtype=dtype)
        x_test1, a_test1, _ = generate_frames(c=channel, frame_form=frame_form_test, delta=delta_t_max, period=20e-6,
                                              dtype=dtype)

        y_train1 = np.array(x_train1[delta_t_max:, :, -delta_t_max:, 0], dtype=dtype_save)
        y_test1 = np.array(x_test1[delta_t_max:, :, -delta_t_max:, 0], dtype=dtype_save)

        # convert from power (modulo) to complex
        # x_train1 = convert_p2c(x_train1, a_train1)
        # x_test1 = convert_p2c(x_test1, a_test1)

        x_train1 = np.array(x_train1[:-delta_t_max], dtype=dtype_save)
        x_test1 = np.array(x_test1[:-delta_t_max], dtype=dtype_save)

        x_train[int(k * frame_form_train[0]):int((k + 1) * frame_form_train[0])] = x_train1
        x_test[int(k * frame_form_test[0]):int((k + 1) * frame_form_test[0])] = x_test1
        y_train[int(k * frame_form_train[0]):int((k + 1) * frame_form_train[0])] = y_train1
        y_test[int(k * frame_form_test[0]):int((k + 1) * frame_form_test[0])] = y_test1

        # x_train = np.concatenate((x_train, x_train1))
        # x_test = np.concatenate((x_test, x_test1))
        # y_train = np.concatenate((y_train, y_train1))
        # y_test = np.concatenate((y_test, y_test1))

        # np.save('data/complex/x_test.npy', x_test)
        # np.save('data/complex/x_train.npy', x_train)
        # np.save('data/complex/y_train.npy', y_train)
        # np.save('data/complex/y_test.npy', y_test)

        print("--> %d " % k, end='')

# end functions
