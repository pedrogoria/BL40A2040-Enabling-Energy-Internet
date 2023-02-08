import simulation_channel as sc
# import tensorflow as tf
import os as os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# sitting constants
scatterers0 = 300
scatterers1 = 2
steps = 10 ** 5
sampling_frequency = 51.2e6
step_time = 500 * 10 ** -6
lags_max = 50 * step_time

data_path = 'data/cross_corr_0/'

if not os.path.exists(data_path):
    os.makedirs(data_path)

# simulating
channel0 = sc.Channel(scatterers=scatterers0, sampling_frequency=sampling_frequency, step_time=step_time, n_moves=200)
channel1 = sc.Channel(scatterers=scatterers1, sampling_frequency=sampling_frequency, step_time=step_time)

print("getting channel 0 samples")
p0, a0, f = channel0(n_steps=steps, only_bw=False)
print("getting channel 1 samples")
p1, a1, _ = channel0(n_steps=steps, only_bw=False)

# normalised cross-covariance
lags = signal.correlation_lags(len(p0[0, :]), len(p0[0, :]), 'same') * step_time
valid_lags = np.where(np.abs(lags) <= lags_max)
lags = lags[valid_lags]

cross_correlation = np.zeros((len(f), len(lags)))
for indx in range(len(f)):
    print("->%d" % int(100 * indx / (len(f) - 1)), end='')
    cross_correlation[indx, :] = np.array(signal.correlate(p0[indx, :] - p0[indx, :].mean(), p1[indx, :] - p1[indx, :].mean(),
                                                           mode='same') / (np.std(p0[indx, :]) * np.std(p1[indx, :]) * steps))[valid_lags]

# normalised auto-covariance
print("-> 100% \n ")
auto_correlation = np.zeros((len(f), len(lags)))
for indx in range(len(f)):
    print("->%d" % int(100 * indx / (len(f) - 1)), end='')
    auto_correlation[indx, :] = np.array(signal.correlate(p0[indx, :] - p0[indx, :].mean(), p0[indx, :] - p0[indx, :].mean(),
                                                          mode='same') / (np.std(p0[indx, :]) * np.std(p0[indx, :]) * steps))[valid_lags]
print("-> 100% \n ")

# printing
fig = plt.figure(figsize=plt.figaspect(0.5))
ax0 = fig.add_subplot(1, 2, 1, projection='3d')
ax1 = fig.add_subplot(1, 2, 2, projection='3d')

# print normalised cross-covariance
x, y = np.meshgrid(lags, f / sampling_frequency)
# ax = plt.axes(projection='3d')
ax0.plot_surface(y, x, cross_correlation, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax0.set_title('power cross-correlation')
ax0.set_xlabel(r'$\frac{f}{f_s}$', fontsize=15)
ax0.set_ylabel(r'Lag [s]', fontsize=15)
ax0.set_zlabel(r'$R(t)$', fontsize=15)

# print normalised auto-covariance
x, y = np.meshgrid(lags, f / sampling_frequency)
# ax = plt.axes(projection='3d')
ax1.plot_surface(y, x, auto_correlation, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax1.set_title('power auto-correlation')
ax1.set_xlabel(r'$\frac{f}{f_s}$', fontsize=15)
ax1.set_ylabel(r'Lag [s]', fontsize=15)
ax1.set_zlabel(r'$R(t)$', fontsize=15)

np.save(data_path + 'lags.npy', lags)
np.save(data_path + 'f.npy', f)
np.save(data_path + 'auto_correlation.npy', auto_correlation)
np.save(data_path + 'cross_correlation.npy', cross_correlation)
np.save(data_path + 'scatterers0.npy', scatterers0)
np.save(data_path + 'scatterers1.npy', scatterers1)
np.save(data_path + 'steps.npy', steps)
np.save(data_path + 'sampling_frequency.npy', sampling_frequency)
np.save(data_path + 'step_time.npy', step_time)
np.save(data_path + 'lags_max.npy', lags_max)
np.save(data_path + 'p0.npy', p0)
np.save(data_path + 'p1.npy', p1)
np.save(data_path + 'a0.npy', a0)
np.save(data_path + 'a1.npy', a1)
