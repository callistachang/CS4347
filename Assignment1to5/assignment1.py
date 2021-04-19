from utils import *


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


start_time = time.time()
assignment1()
print(f"Time taken for Assignment 1: {time.time() - start_time:0.2f}")
