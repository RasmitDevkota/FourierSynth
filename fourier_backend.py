import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq, rfft, irfft, rfftfreq

import streamlit as st

import io
import soundfile as sf

# Filters
# - Average noise
# - Specific frequencies
# - Male voices (85-155 Hz)
# - Female voices (165-255 Hz)
# - Fans
# - Wind??? (20-250 Hz)
# - Music (A4 @ 440 Hz)
# - Birds (100-10K Hz)
# - Feet (???)
# - Noise calibration (record background audio)

equal_gain_plot = np.array((int(20E3), 1))/20E3 # 0 Hz - 20 KHz

preset_gain_plots = {
    "emale": {
        # "85-155": 0
        "0-5000": 0
    },
    "efemale": {
        "160-255": 0
    },
    "ebird": {
        "2000-3000": 0
    }
}

def process_audio(signal, gain_plot, sample_rate, bg_noise_ref=None):
    """
    Filters the input signal to retain only the frequencies within the specified range.

    Parameters:
    - signal: numpy array of the time-domain signal.
    - gain_plot: numpy array of gains for each frequency.
    - sample_rate: Sampling rate of the signal in Hz.
    - bg_noise_ref: (optional) numpy array of Fourier components from a background noise reference scan.

    Returns:
    - processed_signal: Time-domain signal with only the desired frequency range.
    - freqs: Frequencies present in the transformed signal.
    - processed_fft: Fourier components after gains are applied.
    - original_fft: Magnitude of the FFT before processing.
    """

    # Perform FFT on the signal
    n = len(signal)
    fft_values = rfft(signal)
    freqs = rfftfreq(n, d=1/sample_rate)

    # Calculate magnitude of FFT for visualization (before process)
    original_fft = np.abs(fft_values)

    # Copy the FFT values to process frequencies
    processed_fft = np.copy(fft_values)
    # print(processed_fft)

    # Subtract the Fourier components from the background noise reference
    if bg_noise_ref is not None:
        processed_fft -= bg_noise_ref

    # Multiply the FFT values by the gain plot
    for f, freq in enumerate(freqs):
        if freq <= 1:
            processed_fft[f] = 0

        for freq_range, gain in gain_plot.items():
            min_freq = int(freq_range.split("-")[0])
            max_freq = int(freq_range.split("-")[1])
            if freq >= min_freq and freq <= max_freq:
                processed_fft[f] *= gain
                break

    # Delete noisy artifacts of the discrete FFT
    for f, component in enumerate(processed_fft):
        if abs(component) <= 1E-11:
            processed_fft[f] = 0

    # Perform inverse FFT to get back the processed time-domain signal
    processed_signal = irfft(processed_fft)

    return processed_signal, freqs, processed_fft, original_fft

def plot(t, signal, sample_rate, processed_signal, freqs, processed_fft, original_fft, outcon):
    # Plot the results
    figure = plt.figure(figsize=(12, 24))

    # Plot original signal
    plt.subplot(4, 1, 1)
    plt.plot(t, signal, label='Original Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Original Signal')
    plt.legend()

    # Plot FFT magnitude before filtering
    plt.subplot(4, 1, 2)
    plt.plot(freqs, original_fft, label='FFT of Original Signal')
    # plt.xlim(0, sample_rate / 2)
    plt.xlim(0, 1000)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title('FFT of Original Signal')
    plt.legend()

    # Plot FFT magnitude after filtering
    plt.subplot(4, 1, 3)
    plt.plot(freqs, np.abs(processed_fft), label='FFT of Processed Signal')
    # plt.xlim(0, sample_rate / 2)
    plt.xlim(0, 1000)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title('FFT of Processed Signal')
    plt.legend()

    # Plot processed signal
    plt.subplot(4, 1, 4)
    plt.plot(t, processed_signal, label='Processed Signal', color='orange')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Processed Signal in Time Domain')
    plt.legend()

    # plt.tight_layout()

    return figure

def fourier(audio_obj=None, presets=None, outcon=None):
    # Validate inputs
    if audio_obj == None or presets == None or outcon == None:
        # st.write("Error occurred!")
        return

    # Label output container
    outcon.write("Output:")

    # Process audio object input
    audio_str = audio_obj.read()

    audio = np.fromstring(audio_str, np.int16)

    # Convert any stereo input to mono
    if len(audio.shape) > 1:
        audio = audio[:,0]

    # Get sample rate
    audio_array, sample_rate = sf.read(io.BytesIO(audio_str))

    # Plot original audio waveform
    audio = audio[:len(audio_array)]
    audio_length = np.size(audio)
    t = np.array(list(range(len(audio_array))))/sample_rate
    print("time goes between:", min(t), max(t), len(t))

    # Get list of active presets
    all_presets = list(presets.keys())

    active_presets = []
    for preset, switch in presets.items():
        if switch and preset not in active_presets:
            active_presets.append(preset)

    # @TODO - figure out a way to "combine" multiple presets

    # Run Fourier transform and processor
    processed_audio, freqs, processed_fft, original_fft = process_audio(audio, preset_gain_plots[active_presets[0]], sample_rate, bg_noise_ref=None)

    output_fig = plot(t, audio, sample_rate, processed_audio, freqs, processed_fft, original_fft, outcon)
    outcon.pyplot(output_fig)

    # Add processed audio playback
    outcon.write("Listen to processed audio")
    outcon.audio(processed_audio, sample_rate=sample_rate)

    return

