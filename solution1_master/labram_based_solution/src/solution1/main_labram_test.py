import numpy as np
from epilepsy2bids.annotations import Annotations
from epilepsy2bids.eeg import Eeg
from mne.io import RawArray
import mne

from solution1.solution1_labram import labram_algorithm

standard_1020_subset = ['FP1', 'F3', 'C3', 'P3',
                        'O1', 'F7', 'T3', 'T5',
                        'FZ', 'CZ', 'PZ', 'FP2',
                        'F4', 'C4', 'P4', 'O2',
                        'F8', 'T4', 'T6']

def main(edf_file, outFile, model, device):
    eeg = Eeg.loadEdfAutoDetectMontage(edfFile=edf_file)
    if eeg.montage is Eeg.Montage.BIPOLAR:
        print("Model needs unipolar. Eeg Montage BIPOLAR: ", edf_file)  # cant convert to unipolar
        exit(0)
    eeg_data = eeg.data
    channel_names = standard_1020_subset  # replace eeg.channels to remove  "-Avg"  here could be additional check that FP1-Avg => FP1 etc.. but not nessesary for chellenge conditions
    sfreq = eeg.fs
    info = mne.create_info(channel_names, sfreq, ch_types="eeg")
    Rawdata = RawArray(data=eeg_data, info=info)

    Rawdata.notch_filter(50.0)
    Rawdata.resample(200, n_jobs=5)

    signals = Rawdata.get_data()

    # t = np.arange(0, eeg.data.shape[1]) / eeg.fs
    hypMask = labram_algorithm(model, device, 200, signals, ch_names=standard_1020_subset)
    hypMask = np.repeat(hypMask,200)
    hyp = Annotations.loadMask(hypMask, 200)
    hyp.saveTsv(outFile)

