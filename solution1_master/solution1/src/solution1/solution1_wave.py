




def wave_algorithm(
    t, eeg_data, fs, signals, window_size=2, overlap=0.5, background_size=16, transition_size=12
):
    """
    wavelet based algorithm for seizure detection
    Args:
    t: array of time points
    eeg_data: EEG data, shape (n_samples, n_channels)
    fs: sampling frequency
    signals

    Returns:
    seizure_detections: boolean array of seizure detections, shape (n_samples,)
    """
    window_size = int(window_size * fs)
    step = int(window_size * (1 - overlap))
    background_size = int(background_size * fs)
    transition_size = int(transition_size * fs)
    channel_detection = np.zeros(eeg_data.shape)
    idx_detection = [[] for _ in range(eeg_data.shape[0])]

    for k, eeg_channel in enumerate(eeg_data):
        X_data = eeg_channel
        # preprocessing
        X_filtered = lowpass_filter(X_data, fs)

        # conversion to half waves according to Gotman algorithm
        X_halfwave = half_wave(t, X_filtered)

        # channel detection
        for i in range(background_size + transition_size, X_halfwave.shape[0], step):
            background_amp = np.mean(
                np.abs(
                    X_halfwave[
                        i - (background_size + transition_size) : i - transition_size
                    ]
                )
            )
            window_amp = np.mean(np.abs(X_halfwave[i : i + window_size]))
            period_halfwaves = calculate_period(
                t[i : i + window_size], X_halfwave[i : i + window_size]
            )
            mean_period = np.mean(period_halfwaves)

            mean_period_bool = (mean_period > 0.025) & (
                mean_period < 0.150
            )  # between 3 and 20 Hz
            variation_coeff = np.std(period_halfwaves) / mean_period
            variation_coeff_bool = variation_coeff < 0.6
            if (
                (window_amp > 3 * background_amp)
                and mean_period_bool
                and variation_coeff_bool
            ):
                channel_detection[k, i : min(channel_detection.shape[1], i + step)] = 1
                idx_detection[k].append(i)

    # seizure detection criteria
    seizure_detections = np.zeros(eeg_data.shape[1])
    for channel in range(len(idx_detection)):
        for i in range(len(idx_detection[channel]) - 1):
            # check adjacent epochs within same channel
            if (
                np.abs(idx_detection[channel][i + 1] - idx_detection[channel][i])
                == step
            ):
                seizure_detections[
                    idx_detection[channel][i] : idx_detection[channel][i + 1]
                ] = 1
            for channel2 in range(len(idx_detection)):
                if channel == channel2:
                    continue
                for j in range(len(idx_detection[channel2])):
                    # check adjacent epochs within different channels
                    if idx_detection[channel2][j] - idx_detection[channel][i] == step:
                        seizure_detections[
                            idx_detection[channel][i] : idx_detection[channel2][j]
                        ] = 1
                    # check same epoch within different channels
                    if idx_detection[channel2][j] == idx_detection[channel][i]:
                        seizure_detections[
                            idx_detection[channel][i] : idx_detection[channel][i] + step
                        ] = 1

    return seizure_detections