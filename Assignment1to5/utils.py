import numpy as np
import scipy.io.wavfile
import time
from tqdm import tqdm

###############
# Features
###############


def calculate_rms(arr):
    """Calculates root-mean-squared.

    A unique way to characterize the average of continuous
    varying signals such as audio.
    This makes sense because signals cross negative fairly often.
    E.g. It's not very meaningful to say that the average of a sine wave is 0.
    """
    return np.sqrt(np.mean(np.square(arr)))


def calculate_par(arr):
    """Calculates peak-to-average-ratio.

    Quite literally is the ratio from the highest signal peak in the array
    to the average of the array.
    """
    return np.max(np.abs(arr)) / calculate_rms(arr)


def calculate_zcr(arr):
    """Calculates zero crossings.

    If element 1 is negative and element 2 is positive (or vice versa),
    when they are multiplied together, they will give a negative number.
    When their product is negative, it means it has 'crossed zero'.
    """
    arr_multiplied = arr[:-1] * arr[1:]
    arr_binary = np.where(arr_multiplied < 0, 1, 0)
    return np.mean(arr_binary)


def calculate_median_ad(arr):
    """Calculates median absolute deviation.

    A robust measure of the variability of a univariate sample of
    quanitative data; a measure of statistical disperson (like std dev).
    It is robust in the sense that it is more resilient to utliers than std dev.
    """
    return np.median(np.abs(arr - np.median(arr)))


def calculate_mean_ad(arr):
    return np.mean(np.abs(arr - np.mean(arr)))


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
