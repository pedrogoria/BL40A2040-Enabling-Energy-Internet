import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq


class PositionScatter:

    def __init__(self, scatterers=256, n_moves=63, random_method='gaussian', random_state=None,
                 uniform_low=-300, uniform_high=300, standard_deviation=70, step_time=500 * 10 ** -6, **ops):
        np.random.seed(random_state)
        self.step_time = step_time
        self.scatterers = scatterers
        # self.random_method = random_method
        self.n_moves = n_moves if n_moves <= scatterers else scatterers
        if random_method is 'uniform':
            self.sc_positions = np.random.uniform(low=uniform_low, high=uniform_high, size=[scatterers, 2])
        else:
            self.sc_positions = np.random.normal(loc=0, scale=standard_deviation, size=[scatterers, 2])

        self.sc_velocity = ops.pop('sc_velocity', np.random.normal(loc=0, scale=ops.pop('sc_velocity_scale', 5.5), size=[self.n_moves, 2]))

    def move_scatterers(self, **options):
        movement_type = options.pop('movement_type', 'not random')
        velocity = options.pop('velocity', self.sc_velocity)

        if movement_type == 'direction':
            if len(velocity) > self.n_moves:
                velocity = velocity[:self.n_moves]
            if len(velocity) < self.n_moves:
                velocity = np.concatenate((velocity, np.random.random((self.n_moves - len(velocity), 2))))
            module = np.sqrt(np.diag(np.dot(velocity, velocity.T)))
            angle = np.angle(velocity[:, 0] + velocity[:, 1] * 1j)
            velocity = module * np.random.random((self.n_moves,)) * np.array([[np.cos(angle), np.sin(angle)]])
            velocity = velocity.reshape((2, self.n_moves)).transpose()
            velocity = np.concatenate((velocity, np.zeros((self.scatterers - self.n_moves, 2))))
        elif movement_type == 'not random':
            if len(velocity) > self.n_moves:
                velocity = velocity[:self.n_moves]
            if len(velocity) < self.n_moves:
                velocity = np.concatenate((velocity, np.random.random((self.n_moves - len(velocity), 2))))
            velocity = np.concatenate((velocity, np.zeros((self.scatterers - self.n_moves, 2))))
        elif movement_type == 'module':
            angle = 2 * np.pi * np.random.random((self.n_moves, 1))
            module = np.sqrt(np.diag(np.dot(velocity, velocity.T)))
            velocity = module * np.concatenate([np.cos(angle), np.sin(angle)], axis=1)
            velocity = velocity.reshape((2, self.n_moves)).transpose()
            velocity = np.concatenate((velocity, np.zeros((self.scatterers - self.n_moves, 2))))
        elif movement_type == 'gaussian':
            velocity = np.random.normal(loc=0, scale=options.pop('sc_mov_std', 10), size=[self.n_moves, 2])
            velocity = np.concatenate((velocity, np.zeros((self.scatterers - self.n_moves, 2))))
        else:
            velocity = 0

        self.sc_positions = self.sc_positions + self.step_time * velocity


class Channel(PositionScatter):
    def __init__(self, transmitter_position=np.array([[-200, 0]]), receiver_position=np.array([[200, 0]]),
                 scatterers=256, **options):
        self.name = options.pop('name', 'Generic')
        self.transmitter_position = transmitter_position
        self.receiver_position = receiver_position
        self.step_time = options.pop('step_time', 500 * 10 ** -6)
        self.rx_movement_type = options.pop('rx_movement_type', 'not random')
        self.tx_movement_type = options.pop('tx_movement_type', 'not move')
        self.rx_velocity = options.pop('rx_velocity', np.random.normal(1, 2, (1, 2)))
        self.tx_velocity = options.pop('tx_velocity', np.array([[0, 0]]))
        self.sc_movement_type = options.pop('sc_movement_type', 'not random')
        self.LOS = options.pop('LOS', not options.pop('NLOS', True))
        if self.sc_movement_type == 'gaussian':
            self.sc_mov_std = options.pop('sc_mov_std', 10)

        # The doppler frequency is not correct. The receiver and transmitter moving must be considered, and the
        # relative displacement of the scatters is done only by the module, lacks direction!!
        self.sc_doppler = options.pop('frequency_doppler', np.zeros((scatterers, 1)))
        self.sc_doppler = self.sc_doppler if self.sc_doppler.shape == (scatterers, 1) else np.zeros((scatterers, 1))

        super().__init__(scatterers=scatterers, step_time=self.step_time,
                         **options)

        if self.transmitter_position.shape != (1, 2):
            self.transmitter_position = np.array([[0, 0]])

        if self.receiver_position.shape != (1, 2):
            self.receiver_position = np.array([[400, 0]])
        self.update()

    def __call__(self, n_steps=1, sampling_frequency=51.2e6, bandwidth=12.8e6, period=10e-6, log_power=False,
                 only_bw=True, **options):
        p, a, f = self.get_sinc_r_frequency(sampling_frequency=sampling_frequency, bandwidth=bandwidth,
                                            period=period, log_power=log_power,
                                            only_bw=only_bw, **options)
        p = p.reshape((-1, 1))
        a = a.reshape((-1, 1))
        f = f.reshape((-1, 1))
        self.update(**options)
        for k in range(1, n_steps):
            p1, a1, f1 = self.get_sinc_r_frequency(sampling_frequency=sampling_frequency, bandwidth=bandwidth,
                                                   period=period, log_power=log_power,
                                                   only_bw=only_bw, **options)
            p = np.concatenate((p, p1.reshape((-1, 1))), axis=1)
            a = np.concatenate((a, a1.reshape((-1, 1))), axis=1)
            self.update(**options)

        return p, a, f

    def move_tx_rx(self, position, **options):
        movement_type = options.pop('movement_type', 'not move')
        velocity = options.pop('velocity', np.array([[0, 0]]))
        if movement_type == 'direction':
            angle = np.angle(velocity[0] + velocity[1] * 1j)
            module = np.sqrt(np.diag(np.dot(velocity, velocity.T)))
            velocity = module * np.random.random() * np.array([[np.cos(angle), np.sin(angle)]])
        elif movement_type == 'module':
            angle = 2 * np.pi * np.random.random()
            module = np.sqrt(np.diag(np.dot(velocity, velocity.T)))
            velocity = module * np.array([[np.cos(angle), np.sin(angle)]])
        elif movement_type == 'gaussian':
            velocity = np.random.normal(loc=0, scale=options.pop('standard_deviation', 10), size=(1, 2))
        elif movement_type == 'not move':
            velocity = 0

        return position + velocity * self.step_time

    def update(self, carrier_frequency=900e6, c0=3e8, n_steps=1, **ops):
        for k in range(n_steps):
            pos = self.sc_positions
            if self.LOS:
                pos = np.concatenate([pos, self.receiver_position])
            self.receiver_position = self.move_tx_rx(self.receiver_position,
                                                     movement_type=self.rx_movement_type, velocity=self.rx_velocity,
                                                     **ops)
            self.transmitter_position = self.move_tx_rx(self.transmitter_position,
                                                        movement_type=self.tx_movement_type, velocity=self.tx_velocity,
                                                        **ops)
            self.move_scatterers(movement_type=self.sc_movement_type, velocity=self.sc_velocity,
                                 sc_mov_std=self.sc_mov_std if self.sc_movement_type == 'gaussian' else 0, **ops)
            if self.LOS:
                pos = pos - np.concatenate([self.sc_positions, self.receiver_position])
            else:
                pos = pos - self.sc_positions
            pos = np.sqrt(np.diag(np.dot(pos, pos.T)))
            self.sc_doppler = pos.reshape((len(pos), 1)) * carrier_frequency / (self.step_time * c0)

    def get_impulse(self, carrier_frequency=900e6, c0=3e8, normalized_time=True, **ops):
        a1 = (self.sc_positions - self.transmitter_position)
        path_length = np.reshape(np.sqrt(np.diag(a1.dot(a1.T))), (len(a1), 1))
        a1 = (self.sc_positions - self.receiver_position)
        path_length += np.reshape(np.sqrt(np.diag(a1.dot(a1.T))), (len(a1), 1))

        if self.LOS:
            a1 = (self.receiver_position - self.transmitter_position)
            path_length = np.concatenate([path_length, np.reshape(np.sqrt(np.diag(a1.dot(a1.T))), (1, 1))])

        path_delay = path_length / c0
        path_phase = np.mod(-path_length * carrier_frequency / c0, 2 * np.pi)
        amplitude = c0 / (4 * np.pi * carrier_frequency * path_length)
        envelope = amplitude * np.exp(1j * (path_phase + 2 * np.pi * self.sc_doppler * path_delay))
        envelope = envelope / np.sqrt(np.sum(envelope ** 2))
        if normalized_time:
            path_delay = path_delay - min(path_delay)

        return envelope, path_delay

    def get_sinc_response(self, sampling_frequency=51.2e6, bandwidth=12.8e6, period=10e-6, noise=True, **options):
        # dt = options.pop('dtype', 'float64')
        t = np.arange(-period / 2, period / 2, 1 / sampling_frequency)  # , dtype=dt)
        amp_impulse, delay = self.get_impulse(**options)
        out = 0 * t
        for k in range(len(amp_impulse)):
            out = out + amp_impulse[k] * np.sinc(bandwidth * (delay[k] - t))
        energy = np.sum(out ** 2) / sampling_frequency
        out = out / np.sqrt(energy)
        if noise:
            snr_db = options.pop('snr_db', 12)
            snr = options.pop('snr', None)
            if snr is None:
                snr = 10 ** (snr_db / 10)
            out = out + np.random.normal(loc=0, scale=1 / np.sqrt(snr), size=(len(out, )))
            out = out + 1j * np.random.normal(loc=0, scale=1 / np.sqrt(snr), size=(len(out, )))
        out = out * bandwidth * np.sqrt(energy)
        return out, t

    def get_sinc_r_frequency(self, sampling_frequency=51.2e6, bandwidth=12.8e6, period=10e-6, log_power=False,
                             only_bw=False, **options):
        x, t = self.get_sinc_response(sampling_frequency=sampling_frequency, bandwidth=bandwidth, period=period,
                                      **options)
        xf = fft(x)
        power = 10 * np.log10(np.abs(xf) / sampling_frequency) if log_power else np.abs(xf) / sampling_frequency
        angle = np.angle(xf)
        sample_freq = fftfreq(x.size, d=1 / sampling_frequency)
        dt = options.pop('dtype', 'float64')
        p = np.zeros((len(x),), dtype=dt)
        a = np.zeros((len(x),), dtype=dt)
        f = np.zeros((len(x),), dtype=dt)
        pos_mask = np.where(sample_freq < 0)
        l = len(pos_mask[0])
        p[:l] = power[pos_mask]
        a[:l] = angle[pos_mask]
        f[:l] = sample_freq[pos_mask]
        pos_mask = np.where(sample_freq == 0)
        p[l] = power[pos_mask]
        a[l] = angle[pos_mask]
        f[l] = sample_freq[pos_mask]
        pos_mask = np.where(sample_freq > 0)
        p[l + 1:l + len(pos_mask[0]) + 1] = power[pos_mask]
        a[l + 1:l + len(pos_mask[0]) + 1] = angle[pos_mask]
        f[l + 1:l + len(pos_mask[0]) + 1] = sample_freq[pos_mask]

        if only_bw:
            pos_mask = np.where(np.abs(f) <= bandwidth / 2)
            p = p[pos_mask]
            a = a[pos_mask]
            f = f[pos_mask]

        if options.pop('f_pair', False):
            if len(f) % 2 != 0:
                p = p[1:]
                a = a[1:]
                f = f[1:]

        return p, a, f

    def plot_scatterers(self):
        fig = plt.figure(figsize=(7, 7))
        plt.scatter(self.sc_positions[:, 0], self.sc_positions[:, 1])
        plt.scatter(self.receiver_position[0, 0], self.receiver_position[0, 1], s=100)
        plt.scatter(self.transmitter_position[0, 0], self.transmitter_position[0, 1], s=100)
        plt.xlabel('$x$', fontsize=14)
        plt.ylabel('$y$', fontsize=14)
        # plt.grid()
        plt.show()

    def calc_and_plot_fft(self, sampling_frequency=51.2e6, fc=9e6, bandwidth=12.8e6, period=10e-6,
                          new_figure=True, log_power=True, plot_only_bw=True, **options):
        p, a, f = self.get_sinc_r_frequency(sampling_frequency=sampling_frequency, bandwidth=bandwidth, period=period,
                                            log_power=log_power, **options)
        f = f + fc

        if new_figure:
            plt.figure(figsize=(7, 7))
        if plot_only_bw:
            pos_mask = np.where(np.abs(f - fc) <= bandwidth / 2)
            plt.plot(f[pos_mask], p[pos_mask])
        else:
            plt.plot(f, p)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('power')
        plt.show()

        return p, a, f

    def update_and_plot_3dfft(self, n_steps=100, sampling_frequency=51.2e6, bandwidth=12.8e6, period=10e-6, log_power=True,
                              only_bw=False, **options):
        p, a, f = self.__call__(n_steps=n_steps, sampling_frequency=sampling_frequency, bandwidth=bandwidth,
                                period=period, log_power=log_power, only_bw=only_bw, **options)
        if options.pop('time_x_freq', False):
            x, y = np.meshgrid(np.arange(n_steps) * self.step_time, f / sampling_frequency)
            if options.pop('plot_phase', False):
                z = a
            else:
                z = p
            ax = plt.axes(projection='3d')
            ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
            ax.set_title('Frequency Response')
            ax.set_ylabel(r'$\frac{f}{f_s}$ [Hz]', fontsize=15)
            ax.set_xlabel(r'$t$ [s]', fontsize=15)
            ax.set_zlabel('Pot [dB]', fontsize=15)
        else:
            x, y = np.meshgrid(f / sampling_frequency, np.arange(n_steps) * self.step_time)
            if options.pop('plot_phase', False):
                z = a.T
            else:
                z = p.T
            ax = plt.axes(projection='3d')
            ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
            ax.set_title('Frequency Response')
            ax.set_xlabel(r'$\frac{f}{f_s}$ [Hz]', fontsize=15)
            ax.set_ylabel(r'$t$ [s]', fontsize=15)
            ax.set_zlabel('Pot [dB]', fontsize=15)

    def update_and_plot_2dfft(self, n_steps=100, sampling_frequency=51.2e6, bandwidth=12.8e6, period=10e-6, log_power=True,
                              only_bw=True, **options):
        p, a, f = self.__call__(n_steps=n_steps, sampling_frequency=sampling_frequency, bandwidth=bandwidth, period=period, log_power=log_power,
                                only_bw=only_bw, **options)
        plt.figure(figsize=(20, 10))
        plt.pcolormesh(p)
        colb = plt.colorbar()
        plt.xlabel('Time', fontsize=30)
        plt.ylabel('Frequency [MHz]', fontsize=30)
        colb.set_label('Power [dB]', fontsize=30)
        plt.yticks([])
        plt.xticks([])

        plt.show()
