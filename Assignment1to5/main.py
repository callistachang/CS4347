import numpy as np
import scipy.io.wavfile
import time
import pylab

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

def calculate_mad(arr):
    """Calculates median absolute deviation.
    
    A robust measure of the variability of a univariate sample of
    quanitative data; a measure of statistical disperson (like std dev).
    It is robust in the sense that it is more resilient to utliers than std dev. 
    """
    return np.median(np.abs(arr - np.median(arr)))

def read_ground_truth_file():
    """
    Dataset contains 64 music & speech 30s files stored as 16-bit signed integers, 22050 Hz.
    lines = [[filename, music/speech], ..., [filename, music/speech]]
    """
    with open("./music_speech.mf", "r") as fin:
        music_speech_data = [line.strip().split('\t') for line in fin.readlines()]
        return music_speech_data

def assignment1():
    music_speech_data = read_ground_truth_file()
    with open("results.csv", "w") as fout:
        for filename, label in music_speech_data:
            sample_rate, audio_array = scipy.io.wavfile.read(filename)
            # convert data to floats
            audio_array = audio_array / 32768
            rms = calculate_rms(audio_array)
            par = calculate_par(audio_array)
            zcr = calculate_zcr(audio_array)
            mad = calculate_mad(audio_array)
            fout.write(f"{filename},{rms:0.6f},{par:0.6f},{zcr:0.6f},{mad:0.6f}\n")

def assignment2():
    music_speech_data = read_ground_truth_file()
    with open("results.arff", "w") as fout:
        fout.write("@RELATION music_speech\n")
        fout.write("@ATTRIBUTE RMS NUMERIC\n")
        fout.write("@ATTRIBUTE PAR NUMERIC\n")
        fout.write("@ATTRIBUTE ZCR NUMERIC\n")
        fout.write("@ATTRIBUTE MAD NUMERIC\n")
        fout.write("@ATTRIBUTE class {music,speech}\n")
        fout.write("\n")
        fout.write("@DATA\n")

        music_features = np.zeros((len(music_speech_data), 4))
        speech_features = np.zeros((len(music_speech_data), 4))
        for i, (filename, label) in enumerate(music_speech_data):
            sample_rate, audio_array = scipy.io.wavfile.read(filename)
            audio_array = audio_array / 32768
            rms = calculate_rms(audio_array)
            par = calculate_par(audio_array)
            zcr = calculate_zcr(audio_array)
            mad = calculate_mad(audio_array)
            features = [rms, par, zcr, mad]
            if label == "music":
                music_features[i] = features
            elif label == "speech":
                speech_features[i] = features
            fout.write(f"{rms:0.6f},{par:0.6f},{zcr:0.6f},{mad:0.6f},{label}\n")
        pylab.plot(music_features[:,2], music_features[:,1])
        pylab.plot(speech_features[:,2], speech_features[:,1])
        # pylab.show()


if __name__ == "__main__":
    start_time = time.time()
    assignment1()
    print(f"Time taken for Assignment 1: {time.time() - start_time:0.2f}")
    start_time = time.time()
    assignment2()
    print(f"Time taken for Assignment 2: {time.time() - start_time:0.2f}")