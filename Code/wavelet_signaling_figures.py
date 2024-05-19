import os
import time
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
import scipy.stats

from collections import defaultdict, Counter


# Varioius Wavelet Types

discrete_wavelets = ['db5', 'sym5', 'coif5', 'bior2.4']
continuous_wavelets = ['mexh', 'morl', 'cgau5', 'gaus5']
list_list_wavelets = [discrete_wavelets, continuous_wavelets]
list_funcs = [pywt.Wavelet, pywt.ContinuousWavelet]

fig, axarr = plt.subplots(nrows=2, ncols=4, figsize=(16,8))
for ii, list_wavelets in enumerate(list_list_wavelets):
    func = list_funcs[ii]
    row_no = ii
    for col_no, waveletname in enumerate(list_wavelets):
        wavelet = func(waveletname)
        family_name = wavelet.family_name
        biorthogonal = wavelet.biorthogonal
        orthogonal = wavelet.orthogonal
        symmetry = wavelet.symmetry
        if ii == 0:
            _ = wavelet.wavefun()
            wavelet_function = _[0]
            x_values = _[-1]
        else:
            wavelet_function, x_values = wavelet.wavefun()
        if col_no == 0 and ii == 0:
            axarr[row_no, col_no].set_ylabel("Discrete Wavelets", fontsize=16)
        if col_no == 0 and ii == 1:
            axarr[row_no, col_no].set_ylabel("Continuous Wavelets", fontsize=16)
        axarr[row_no, col_no].set_title("{}".format(family_name), fontsize=16)
        axarr[row_no, col_no].plot(x_values, wavelet_function)
        axarr[row_no, col_no].set_yticks([])
        axarr[row_no, col_no].set_yticklabels([])

plt.tight_layout()
plt.savefig('Figures/wavelet.pdf', format='pdf', dpi=1200)
plt.show()



# Original Chirp Signal

x = np.linspace(0, 1, num=1024)
chirp_signal = np.sin(240 * np.pi * x**2)

fig, ax = plt.subplots(figsize=(6,1))
ax.set_title("Original Chirp Signal: ")
ax.plot(chirp_signal, color = 'teal')
ax.set_yticks([])
ax.set_yticklabels([])
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
plt.savefig('Figures/chirp_signal_reformatted.pdf', format='pdf', dpi=1200)

plt.show()


# Schema

data = chirp_signal
waveletname = 'sym5'

fig, axarr = plt.subplots(nrows=5, ncols=2, figsize=(6,6))
for ii in range(5):
    (data, coeff_d) = pywt.dwt(data, waveletname)
    axarr[ii, 0].plot(data, color = 'darkblue')
    axarr[ii, 1].plot(coeff_d, color = 'orange')
    axarr[ii, 0].set_ylabel("Level {}".format(ii + 1), fontsize=14, rotation=90)
    axarr[ii, 0].set_yticklabels([])
    if ii == 0:
        axarr[ii, 0].set_title("Approximation coefficients", fontsize=14)
        axarr[ii, 1].set_title("Detail coefficients", fontsize=14)
    axarr[ii, 1].set_yticklabels([])
plt.tight_layout()
plt.savefig('Figures/illustration_reformatted.pdf', format='pdf', dpi=1200)
plt.show()



# Bior figures

import pywt
import matplotlib.pyplot as plt

# Define the wavelets
wavelets = ['bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8']
fig, axes = plt.subplots(5, 3, figsize=(15, 10))  # This creates 15 subplots
axes = axes.flatten()

# Generate and plot each wavelet
for i, wavelet in enumerate(wavelets):
    wavelet_function = pywt.Wavelet(wavelet)
    outputs = wavelet_function.wavefun(level=5)

    # Assuming psi is always the second-to-last output and x_psi is the last
    psi = outputs[-2]
    x_psi = outputs[-1]

    axes[i].plot(x_psi, psi, color = 'teal')
    axes[i].set_title(wavelet)
    axes[i].grid(True)

# Hide the last two axes which aren't needed
for j in range(i + 1, 15):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('Figures/bior_waves.pdf', format='pdf', dpi=1200)
plt.show()



# Sine Wave vs Wavelet

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

# Parameters for the sine wave
frequency = 1  # Frequency in Hz
sampling_rate = 100  # Sampling rate in Hz
t = np.linspace(0, 2, int(2 * sampling_rate), endpoint=False)  # 2 seconds duration

# Generate the sine wave
sine_wave = np.sin(2 * np.pi * frequency * t)

# Parameters for the wavelet (using a Morlet wavelet for illustration)
widths = np.array([7])
wavelet = signal.morlet(100, w=5.0, s=widths, complete=True)

# Plot both the sine wave and the wavelet
fig, ax = plt.subplots(2, 1, figsize=(10, 6))

# Plotting the sine wave
ax[0].plot(t, sine_wave, label='Sine Wave', color='teal')
ax[0].set_title('Sine Wave')
ax[0].set_ylabel('Amplitude')
ax[0].set_xlabel('Time (seconds)')

# Plotting the wavelet
x_wavelet = np.linspace(-1, 1, len(wavelet))
ax[1].plot(x_wavelet, wavelet.real, label='Real Part of Wavelet', color='teal')  # Plotting the real part
ax[1].set_title('Wavelet')
ax[1].set_ylabel('Amplitude')
ax[1].set_xlabel('Time (seconds)')

plt.tight_layout()
plt.savefig('Figures/sine_vs_wavelet.pdf', format='pdf', dpi=1200)
plt.show()


# Sym5 Wavelet

import numpy as np
import matplotlib.pyplot as plt
import pywt

# Generate the sym5 wavelet
wavelet = 'sym5'
sym5_wavelet = pywt.Wavelet(wavelet)
phi, psi, x = sym5_wavelet.wavefun(level=5)  # Correctly unpack all returned values

# Plot the sym5 wavelet
plt.figure(figsize=(10, 4))
plt.plot(x, psi, 'k', color = 'teal')  # Plotting the wavelet function (psi)
plt.title(f'{wavelet} Wavelet')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.savefig('Figures/sym5.pdf', format='pdf', dpi=1200)
plt.show()


# Wavelet Signaling Figures
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def generate_normal_beat():
    t = np.linspace(0, 1, 180)
    p_wave = 0.5 * np.exp(-((t - 0.2) ** 2) / 0.01)
    q_wave = -1.5 * np.exp(-((t - 0.45) ** 2) / 0.002)
    r_wave = 3.0 * np.exp(-((t - 0.5) ** 2) / 0.0005)
    s_wave = -1.0 * np.exp(-((t - 0.55) ** 2) / 0.002)
    t_wave = 1.0 * np.exp(-((t - 0.75) ** 2) / 0.02)
    return t, p_wave + q_wave + r_wave + s_wave + t_wave

def generate_sveb_beat():
    t = np.linspace(0, 1, 180)
    p_wave = 0.25 * np.exp(-((t - 0.15) ** 2) / 0.01)
    q_wave = -1.0 * np.exp(-((t - 0.4) ** 2) / 0.002)
    r_wave = 2.5 * np.exp(-((t - 0.45) ** 2) / 0.0005)
    s_wave = -1.0 * np.exp(-((t - 0.5) ** 2) / 0.002)
    t_wave = 0.5 * np.exp(-((t - 0.65) ** 2) / 0.02)
    return t, p_wave + q_wave + r_wave + s_wave + t_wave

def generate_veb_beat():
    t = np.linspace(0, 1, 180)
    q_wave = -2.0 * np.exp(-((t - 0.45) ** 2) / 0.003)
    r_wave = 5.0 * np.exp(-((t - 0.5) ** 2) / 0.0005)
    s_wave = -2.0 * np.exp(-((t - 0.55) ** 2) / 0.003)
    t_wave = 1.0 * np.exp(-((t - 0.75) ** 2) / 0.02)
    return t, q_wave + r_wave + s_wave + t_wave

def generate_fusion_beat():
    t = np.linspace(0, 1, 180)
    p_wave = 0.3 * np.exp(-((t - 0.2) ** 2) / 0.01)
    q_wave = -1.0 * np.exp(-((t - 0.45) ** 2) / 0.002)
    r_wave = 3.0 * np.exp(-((t - 0.5) ** 2) / 0.0005)
    s_wave = -1.0 * np.exp(-((t - 0.55) ** 2) / 0.002)
    t_wave = 0.5 * np.exp(-((t - 0.75) ** 2) / 0.02)
    q_wave2 = -1.5 * np.exp(-((t - 0.65) ** 2) / 0.003)
    r_wave2 = 2.0 * np.exp(-((t - 0.7) ** 2) / 0.0005)
    s_wave2 = -1.5 * np.exp(-((t - 0.75) ** 2) / 0.003)
    return t, p_wave + q_wave + r_wave + s_wave + t_wave + q_wave2 + r_wave2 + s_wave2

def generate_unknown_beat():
    t = np.linspace(0, 1, 180)
    noise = 1.0 * np.random.randn(180)
    return t, noise

# Generate the beats
t, normal_beat = generate_normal_beat()
t, sveb_beat = generate_sveb_beat()
t, veb_beat = generate_veb_beat()
t, fusion_beat = generate_fusion_beat()
t, unknown_beat = generate_unknown_beat()

# Extend beats to simulate multiple heartbeats in sequence
normal_beat = np.tile(normal_beat, 5)
sveb_beat = np.tile(sveb_beat, 5)
veb_beat = np.tile(veb_beat, 5)
fusion_beat = np.tile(fusion_beat, 5)
unknown_beat = np.tile(unknown_beat, 5)

# Generate time vector for extended signals
t = np.linspace(0, 5, len(normal_beat))

# Plot the beats
fig = plt.figure(figsize=(10, 8))
gs = GridSpec(3, 2, height_ratios=[1, 1, 1.5])

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(t, normal_beat)
ax1.set_title("Class 1 (Normal)")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude")
ax1.grid(True)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(t, sveb_beat)
ax2.set_title("Class 2 (SVEB)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Amplitude")
ax2.grid(True)

ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(t, veb_beat)
ax3.set_title("Class 3 (VEB)")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Amplitude")
ax3.grid(True)

ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(t, fusion_beat)
ax4.set_title("Class 5 (Fusion)")
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Amplitude")
ax4.grid(True)

ax5 = fig.add_subplot(gs[2, :])
ax5.plot(t, unknown_beat)
ax5.set_title("Class 12 (Unknown)")
ax5.set_xlabel("Time (s)")
ax5.set_ylabel("Amplitude")
ax5.grid(True)

plt.tight_layout()
plt.savefig('Figures/beats.pdf', format='pdf', dpi=1200)
plt.show()


# Decompositions Waves

 import numpy as np
import matplotlib.pyplot as plt

# Generate the original complex wave with higher frequencies and multiple wavelengths
t = np.linspace(0, 4 * np.pi, 500)
original_wave = np.sin(4 * t) + 0.5 * np.sin(10 * t) + 0.25 * np.sin(20 * t)

# Generate decomposed waves with different wavelengths and positions
wave1 = 0.5 * np.sin(8 * t)      # Much shorter wavelength
wave2 = 0.3 * np.sin(2 * t)      # Shorter wavelength
wave3 = 0.2 * np.sin(0.5 * t)    # Longer wavelength
wave4 = 0.25 * np.sin(10 * t + np.pi / 4)  # Phase shifted
wave5 = 0.15 * np.sin(5 * t + np.pi / 2)   # Different wavelength and phase shift

# Create the plot
fig, axs = plt.subplots(6, 1, figsize=(15, 10))

# Plot the original wave on the left
axs[0].plot(t, original_wave, color='blue')
axs[0].set_title("Original Complex Wave")
axs[0].axis('off')

# Plot the decomposed waves on the right
axs[1].plot(t, wave1, color='red')
axs[1].set_title("Decomposed Wave 1")
axs[1].axis('off')

axs[2].plot(t, wave2, color='green')
axs[2].set_title("Decomposed Wave 2")
axs[2].axis('off')

axs[3].plot(t, wave3, color='blue')
axs[3].set_title("Decomposed Wave 3")
axs[3].axis('off')

axs[4].plot(t, wave4, color='orange')
axs[4].set_title("Decomposed Wave 4")
axs[4].axis('off')

axs[5].plot(t, wave5, color='purple')
axs[5].set_title("Decomposed Wave 5")
axs[5].axis('off')

plt.tight_layout()

plt.savefig('Figures/decomposed.pdf', format='pdf', dpi=1200)
plt.show()