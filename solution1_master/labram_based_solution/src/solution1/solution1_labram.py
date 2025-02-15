import numpy as np
import pywt
from PyQt5.QtCore import qQNaN
import solution1.utils as utils
import torch
from einops import rearrange

@torch.no_grad()
# def labram_algorithm(model, device, fs, signals, ch_names=None):
#     """
#     Labram-based algorithm for seizure detection.
#     Args:
#         model: The trained PyTorch model.
#         device: Device to run the model on (e.g., CPU or GPU).
#         fs: Sampling frequency.
#         signals: EEG signals with shape (n_channels, n_samples).
#         ch_names: List of channel names (optional).
# 
#     Returns:
#         output: Boolean array indicating seizure detections, shape (n_samples,).
#     """
#     input_chans = None
#     if ch_names is not None:
#         input_chans = utils.get_input_chans(ch_names)
# 
#     num_of_sec = int(signals.shape[1] // fs)
#     num_chan = signals.shape[0]
#     features = np.zeros([num_of_sec, num_chan, fs * 5])
#     offset = signals.shape[1]
# 
#     if signals[:, -2 * (fs):].shape[1] < 400 or signals[:, 0:2 * (fs)].shape[1] < 400:
#         signals_add = np.concatenate((np.zeros([19, 400]), signals, np.zeros([19, 400])), axis=1)
#     else:
#         signals_add = np.concatenate((signals[:, 0:2 * (fs)], signals, signals[:, -2 * (fs):]), axis=1)
# 
#     for i in range(num_of_sec):
#         features[i, :, :] = signals_add[:, i * fs:(i + 5) * fs]
# 
#     batch_size = 2000
#     n_batches = (features.shape[0] + batch_size - 1) // batch_size
#     all_answers = np.zeros([num_of_sec, 6])
# 
#     for i in range(n_batches):
#         start_idx = i * batch_size
#         end_idx = min(start_idx + batch_size, features.shape[0])
# 
#         batch_features = torch.from_numpy(features[start_idx:end_idx]).float().to(device) / 100
#         batch_features = rearrange(batch_features, 'B N (A T) -> B N A T', T=200)
# 
#         answers = model(batch_features, input_chans=input_chans)
#         all_answers[start_idx:end_idx] = answers.cpu().detach().numpy()
# 
#     output = (all_answers.argmax(1) < 3)
#     return output



def labram_algorithm(
    # t, eeg_data, fs, signals, XGB_model, window_size=2, overlap=0.5, background_size=16, transition_size=12, wavelet_level_4 = True
    model, device, fs, signals, ch_names=None
):
    """
    labram based algorithm for seizure detection
    Args:
    t: array of time points
    eeg_data: EEG data, shape (n_samples, n_channels)
    fs: sampling frequency
    signals

    Returns:
    seizure_detections: boolean array of seizure detections, shape (n_samples,)
    """

    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)

    num_of_sec = int(signals.shape[1]/fs)
    numChan = signals.shape[0]
    features = np.zeros([num_of_sec, numChan, (fs) * 5])
    offset = signals.shape[1]
    if signals[:,-2*(fs):].shape[1]<400 or signals[:,0:2*(fs)].shape[1]<400:
        signals_add = np.concatenate((np.zeros([19, 400]),signals,np.zeros([19, 400])),axis=1)
    else:
        signals_add = np.concatenate((signals[:,0:2*(fs)],signals,signals[:,-2*(fs):]),axis=1)   # add 2 sec left and right
    to_pred = []
    for i in range(0, num_of_sec):  # number of indexes increase on  4
        features[i, :] = signals_add[:,(i)*fs:(i+5)*fs]               # get 5 sec interval signals[-2 : +3] or signals_add[0:5]

    EEG=torch.tensor(features)
    EEG = EEG.float().to(device, non_blocking=True) / 100
    EEG = rearrange(EEG, 'B N (A T) -> B N A T', T=200)
    model.to(device)
    # batch 350 maximum
    batch_size = 2000
    n_batches = EEG.shape[0]//batch_size
    all_answer = np.zeros([num_of_sec, 6])
    for i in range(n_batches+1):
        answer = model(EEG[i*batch_size:(i+1)*batch_size,:], input_chans=input_chans)
        all_answer[i*batch_size:(i+1)*batch_size,:] = answer.cpu().detach().numpy()



    # output = torch.tensor(answer.argmax(1) < 3).float()
    output = (all_answer.argmax(1) < 3)

    return output