from utils import _read_ground_truth_file, _read_audio_file, calculate_rms, calculate_par, calculate_zcr, calculate_median_ad 
import time
import numpy as np

header = """@RELATION music_speech
@ATTRIBUTE RMS NUMERIC
@ATTRIBUTE PAR NUMERIC
@ATTRIBUTE ZCR NUMERIC
@ATTRIBUTE MAD NUMERIC
@ATTRIBUTE class {music,speech}

@DATA
"""

def assignment2():
    music_speech_data = _read_ground_truth_file()
    with open("out/assignment2.arff", "w") as fout:
        fout.write(header)

        music_features = np.zeros((len(music_speech_data), 4))
        speech_features = np.zeros((len(music_speech_data), 4))
        for fpath, label in music_speech_data:
            audio_array = _read_audio_file(fpath)
            rms = calculate_rms(audio_array)
            par = calculate_par(audio_array)
            zcr = calculate_zcr(audio_array)
            mad = calculate_median_ad(audio_array)
            features = np.array([rms, par, zcr, mad])
            fout.write(f"{rms:0.6f},{par:0.6f},{zcr:0.6f},{mad:0.6f},{label}\n")
        # I'm too lazy to plot the plots


start_time = time.time()
assignment2()
print(f"Time taken for Assignment 2: {time.time() - start_time:0.2f}")
