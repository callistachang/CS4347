from utils import *


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
