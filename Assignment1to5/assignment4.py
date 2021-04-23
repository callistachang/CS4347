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

start_time = time.time()
assignment4()
print(f"Time taken for Assignment 4: {time.time() - start_time:0.2f}")
