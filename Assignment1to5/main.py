import numpy as np
import scipy.io.wavfile
import scipy.signal.windows
import scipy.fftpack
import time
from tqdm import tqdm

###############
# Features
###############
def calculate_rms(arr):
    """Calculates root-mean-squared of an audio array.

    A unique way to characterize the average of continuous
    varying signals such as audio.
    This makes sense because signals cross negative fairly often.
    E.g. It's not very meaningful to say that the average of a sine wave is 0.
    """
    return np.sqrt(np.mean(arr * 2))


def calculate_par(arr):
    """Calculates peak-to-average-ratio of an audio array.

    Quite literally is the ratio from the highest signal peak in the array
    to the average of the array.
    """
    return np.max(np.abs(arr)) / calculate_rms(arr)


def calculate_zcr(arr):
    """Calculates zero crossings of an audio array.

    If element 1 is negative and element 2 is positive (or vice versa),
    when they are multiplied together, they will give a negative number.
    When their product is negative, it means it has 'crossed zero'.
    """
    arr_multiplied = arr[:-1] * arr[1:]
    arr_binary = np.where(arr_multiplied < 0, 1, 0)
    return np.mean(arr_binary)


def calculate_median_ad(arr):
    """Calculates median absolute deviation of an audio array.

    A robust measure of the variability of a univariate sample of
    quanitative data; a measure of statistical disperson (like std dev).
    It is robust in the sense that it is more resilient to utliers than std dev.
    """
    return np.median(np.abs(arr - np.median(arr)))


def calculate_mean_ad(arr):
    return np.mean(np.abs(arr - np.mean(arr)))


def calculate_spectral_centroid(mat):
    """
    mat -> an array of spectrums

    A measure used in DSP to characterize a spectrum.

    Indicates where the centre of mass of the spectrum is located.

    Has a robust connection with the impression of brightness of a sound.
    'Brightness' is usually an indication of the amount of high-frequency content in a sound.

    Calculated as the weighted mean of frequencies present in the signal (determined using a Fourier transform)
    with magnitudes as the weights.

    'Weighted mean' is similar to an arithmetic mean except some data points contribute more than others.
    If all weights are equal, then the weighted mean is the same as the arithmetic mean.
    """
    numer = np.sum(mat * range(mat.shape[1]), axis=1)
    denom = np.sum(mat, axis=1)
    return numer / denom


def calculate_spectral_roll_off(mat):
    pass


def calculate_spectral_flatness_measure(mat):
    pass


def calculate_spectral_flux(mat):
    pass


###############
# Utils
###############
def _read_ground_truth_file():
    """
    Dataset contains 64 music & speech 30s files stored as 16-bit signed integers, 22050 Hz.
    lines = [[filename, music/speech], ..., [filename, music/speech]]
    """
    with open("./music_speech.mf", "r") as fin:
        music_speech_data = [line.strip().split("\t") for line in fin.readlines()]
    return music_speech_data


def _read_audio_file(fpath):
    _, audio_array = scipy.io.wavfile.read(fpath)
    return audio_array / 32758


###############
# Main
###############
def assignment1():
    music_speech_data = _read_ground_truth_file()
    with open("results.csv", "w") as fout:
        for fpath, label in music_speech_data:
            audio_array = _read_audio_file(fpath)
            rms = calculate_rms(audio_array)
            par = calculate_par(audio_array)
            zcr = calculate_zcr(audio_array)
            mad = calculate_median_ad(audio_array)
            fout.write(f"{filename},{rms:0.6f},{par:0.6f},{zcr:0.6f},{mad:0.6f}\n")


def assignment2():
    music_speech_data = _read_ground_truth_file()
    with open("assignment2.arff", "w") as fout:
        fout.write("@RELATION music_speech\n")
        fout.write("@ATTRIBUTE RMS NUMERIC\n")
        fout.write("@ATTRIBUTE PAR NUMERIC\n")
        fout.write("@ATTRIBUTE ZCR NUMERIC\n")
        fout.write("@ATTRIBUTE MAD NUMERIC\n")
        fout.write("@ATTRIBUTE class {music,speech}\n")
        fout.write("\n")
        fout.write("@DATA\n")

        for fpath, label in music_speech_data:
            audio_array = _read_audio_file(fpath)
            rms = calculate_rms(audio_array)
            par = calculate_par(audio_array)
            zcr = calculate_zcr(audio_array)
            mad = calculate_median_ad(audio_array)
            features = np.array([rms, par, zcr, mad])
            fout.write(f"{rms:0.6f},{par:0.6f},{zcr:0.6f},{mad:0.6f},{label}\n")
        # I'm too lazy to plot the plots


def assignment3():
    music_speech_data = _read_ground_truth_file()

    with open("assignment3.arff", "w") as fout:
        fout.write("@RELATION music_speech\n")
        fout.write("@ATTRIBUTE RMS_MEAN NUMERIC\n")
        fout.write("@ATTRIBUTE PAR_MEAN NUMERIC\n")
        fout.write("@ATTRIBUTE ZCR_MEAN NUMERIC\n")
        fout.write("@ATTRIBUTE MAD_MEAN NUMERIC\n")
        fout.write("@ATTRIBUTE MEAN_AD_MEAN NUMERIC\n")
        fout.write("@ATTRIBUTE RMS_STD NUMERIC\n")
        fout.write("@ATTRIBUTE PAR_STD NUMERIC\n")
        fout.write("@ATTRIBUTE ZCR_STD NUMERIC\n")
        fout.write("@ATTRIBUTE MAD_STD NUMERIC\n")
        fout.write("@ATTRIBUTE MEAN_AD_STD NUMERIC\n")
        fout.write("@ATTRIBUTE class {music,speech}\n")
        fout.write("\n")
        fout.write("@DATA\n")

        for fpath, label in tqdm(music_speech_data):
            audio_array = _read_audio_file(fpath)
            buffer_features = []
            # split data into buffers of length 1024 with 50% overlap (hopsize of 512)
            # only include complete buffers. if buffer length < 1024, omit it
            # should result in a (1290, 5) matrix per wav file
            for i in range(1024, len(audio_array), 512):
                buffer = audio_array[i - 1024 : i]
                rms = calculate_rms(buffer)
                par = calculate_par(buffer)
                zcr = calculate_zcr(buffer)
                median_ad = calculate_median_ad(buffer)
                mean_ad = calculate_mean_ad(buffer)
                buffer_features.append([rms, par, zcr, median_ad, mean_ad])
            buffer_features = np.array(buffer_features)
            rms_mean, par_mean, zcr_mean, median_ad_mean, mean_ad_mean = np.mean(
                buffer_features, axis=0
            ).T
            rms_std, par_std, zcr_std, median_ad_std, mean_ad_std = np.std(
                buffer_features, axis=0
            ).T
            fout.write(
                f"{rms_mean:0.6f},{par_mean:0.6f},{zcr_mean:0.6f},{median_ad_mean:0.6f},{mean_ad_mean:0.6f},{rms_std:0.6f},{par_std:0.6f},{zcr_std:0.6f},{median_ad_std:0.6f},{mean_ad_std:0.6f},{label}\n"
            )


def assignment4():
    # a windowing function for smoothing values
    # M represents the number of points in the output window
    hamming_window = scipy.signal.windows.hamming(1024)

    music_speech_data = _read_ground_truth_file()
    fpath, label = music_speech_data[0]
    audio_array = _read_audio_file(fpath)

    dft_matrix = np.zeros((1290, 513))

    j = 0
    for i in range(1024, len(audio_array), 512):
        buffer = audio_array[i - 1024 : i]
        buffer *= hamming_window
        # perform discrete fourier transform
        buffer_dft = scipy.fftpack.fft(buffer)
        buffer_dft = buffer_dft[:513]
        buffer_dft = np.abs(buffer_dft)
        dft_matrix[j] = buffer_dft
        j += 1

    # print(dft_matrix)
    # print(dft_matrix.shape)
    calculate_spectral_centroid(dft_matrix)


if __name__ == "__main__":
    # start_time = time.time()
    # assignment1()
    # print(f"Time taken for Assignment 1: {time.time() - start_time:0.2f}")
    # start_time = time.time()
    # assignment2()
    # print(f"Time taken for Assignment 2: {time.time() - start_time:0.2f}")
    # start_time = time.time()
    # assignment3()
    # print(f"Time taken for Assignment 3: {time.time() - start_time:0.2f}")
    start_time = time.time()
    assignment4()
    print(f"Time taken for Assignment 4: {time.time() - start_time:0.2f}")
