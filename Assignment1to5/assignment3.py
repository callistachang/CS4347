from utils import *


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
            # should result in a (1920, 5) matrix per wav file
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


start_time = time.time()
assignment3()
print(f"Time taken for Assignment 3: {time.time() - start_time:0.2f}")
