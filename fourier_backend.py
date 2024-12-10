import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq

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
        "85-155": 0
    },
    "efemale": {
        "165-255": 0
    },
    "ebird": {
        "2000-3000": 0
    }
}

def filter_frequency_range(signal, gain_plot, sample_rate, bg_noise_ref=None):
    """
    Filters the input signal to retain only the frequencies within the specified range.

    Parameters:
    - signal: numpy array of the time-domain signal.
    - gain_plot: numpy array of gains for each frequency.
    - sample_rate: Sampling rate of the signal in Hz.
    - bg_noise_ref: (optional) numpy array of Fourier components from a background noise reference scan.

    Returns:
    - filtered_signal: Time-domain signal with only the desired frequency range.
    - freqs: Frequencies present in the transformed signal.
    - filtered_fft: Fourier components after gains are applied
    - original_fft: Magnitude of the FFT before filtering.
    """

    # Perform FFT on the signal
    n = len(signal)
    fft_values = fft(signal)
    freqs = fftfreq(n, d=1/sample_rate)

    # Copy the FFT values to filter frequencies
    filtered_fft = np.copy(fft_values)
    # print(filtered_fft)

    # Subtract the Fourier components from the background noise reference
    if bg_noise_ref is not None:
        filtered_fft -= bg_noise_ref

    # Multiply the FFT values by the gain plot
    for f, freq in enumerate(freqs):
        for freq_range, gain in gain_plot.items():
            min_freq = int(freq_range.split("-")[0])
            max_freq = int(freq_range.split("-")[1])
            if freq >= min_freq and freq <= max_freq:
                filtered_fft[f] *= gain
                break
            else:
                pass
                # center_freq = (min_freq + max_freq)/2
                # quotient = freq/center_freq
                # if abs(int(quotient) - quotient)/quotient <= 0.25:
                #     filtered_fft[f] *= gain
                # break

    # Delete noisy artifacts of the discrete FFT
    for f, component in enumerate(filtered_fft):
        if abs(component) <= 1E-11:
            filtered_fft[f] = 0

    # filtered_fft[1] = 200
    # print(filtered_fft[115:135])

    # Calculate magnitude of FFT for visualization (before filtering)
    original_fft = np.abs(fft_values)

    # Perform inverse FFT to get back the filtered time-domain signal
    filtered_signal = np.real(ifft(filtered_fft))

    # print(filtered_signal)

    return filtered_signal, freqs, filtered_fft, original_fft

def plot(t, signal, filtered_signal, freqs, filtered_fft, original_fft):
    # Plot the results
    # plt.figure(figsize=(12, 8))
    fig, ax = plt.subplots(figsize=(12, 8))

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
    plt.xlim(0, sample_rate / 2)  # Limit to positive frequencies
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title('FFT of Original Signal')
    plt.legend()

    # Plot FFT magnitude after filtering
    plt.subplot(4, 1, 3)
    plt.plot(freqs, filtered_fft, label='FFT of Filtered Signal')
    plt.xlim(0, sample_rate / 2)  # Limit to positive frequencies
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title('FFT of Filtered Signal')
    plt.legend()

    # Plot filtered signal
    plt.subplot(4, 1, 4)
    plt.plot(t, filtered_signal, label='Filtered Signal (0-20,000 Hz)', color='orange')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Filtered Signal in Time Domain')
    plt.legend()

    plt.tight_layout()

    return fig, ax

def fourier(audio_obj=None, presets=None, outcon=None):
    # Validate inputs
    if audio_obj == None or presets == None or outcon == None:
        # st.write("Error occurred!")
        return

    # Process audio object input
    audio_str = audio_obj.read()
    # outcon.write(audio_str)

    audio = np.fromstring(audio_str, np.int16)
    # outcon.write(np.shape(audio))

    # Plot original audio waveform
    audio_length = np.size(audio)
    t = np.array(list(range(audio_length)))

    input_fig, _ = plt.subplots()
    plt.plot(t, audio)
    outcon.pyplot(input_fig)

    # Get list of active presets
    all_presets = list(presets.keys())
    outcon.write("presets:", presets)

    active_presets = []
    for preset, switch in presets.items():
        if switch and preset not in active_presets:
            active_presets.append(preset)
    outcon.write("active_presets:", active_presets)

    # @TODO - figure out a way to "combine" multiple presets

    # Get sample rate
    audio_array, sample_rate = sf.read(io.BytesIO(audio_str))
    outcon.write(sample_rate)

    # Run Fourier transform and equalizer
    filtered_audio, freqs, filtered_fft, original_fft = filter_frequency_range(audio,
                                                                                preset_gain_plots[active_presets[0]],
                                                                                sample_rate, bg_noise_ref=None)

    output_fig, _ = plot(t, audio, filtered_audio, freqs, filtered_fft, original_fft)

    outcon.pyplot(output_fig)

    return

