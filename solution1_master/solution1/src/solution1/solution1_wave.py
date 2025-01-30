import numpy as np
import pywt



def wave_algorithm(
    # t, eeg_data, fs, signals, XGB_model, window_size=2, overlap=0.5, background_size=16, transition_size=12, wavelet_level_4 = True
    fs, signals, XGB_model, wavelet_level_4 = True
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
    # window_size = int(window_size * fs)
    # step = int(window_size * (1 - overlap))
    # background_size = int(background_size * fs)
    # transition_size = int(transition_size * fs)
    # channel_detection = np.zeros(eeg_data.shape)
    # idx_detection = [[] for _ in range(eeg_data.shape[0])]

    # for i in np.arange(0, signals.shape[1]):
    num_of_sec = int(signals.shape[1]/fs)
    numChan = signals.shape[0]
    features = np.zeros([num_of_sec, numChan, (fs) * 5])
    offset = signals.shape[1]
    signals_add = np.concatenate((signals[:,0:2*(fs)],signals,signals[:,-2*(fs):]),axis=1)   # add 2 sec left and right
    to_pred = []
    for i in range(0, num_of_sec):  # number of indexes increase on  4
        features[i, :] = signals_add[:,(i)*fs:(i+5)*fs]               # get 5 sec interval signals[-2 : +3] or signals_add[0:5]

        file = features[i]
        if wavelet_level_4:
            coefficients = pywt.wavedec(file, wavelet='haar', level=4)
            X = coefficients[0][0:8]
            to_pred.append(X.reshape(504))
        else:
            coefficients = pywt.dwt(file, 'haar')  # Perform discrete Haar wavelet transform
            X = coefficients[0][0:8]
            to_pred.append(X.reshape(4000))

    answer = XGB_model.predict_proba(to_pred)
    # output = torch.tensor(answer.argmax(1) < 3).float()
    output = (answer.argmax(1) < 3)

    return output