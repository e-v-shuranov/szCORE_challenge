import numpy as np
from epilepsy2bids.annotations import Annotations
from epilepsy2bids.eeg import Eeg
from mne.io import RawArray
import mne

from solution1.solution1_wave import wave_algorithm


def main(edf_file, outFile, XGB_mod=None):
    eeg = Eeg.loadEdfAutoDetectMontage(edfFile=edf_file)
    if eeg.montage is Eeg.Montage.UNIPOLAR:
        eeg.reReferenceToBipolar()

    eeg_data = eeg.data
    channel_names = eeg.channels
    sfreq = eeg.fs

    info = mne.create_info(channel_names, sfreq, ch_types="eeg")
    Rawdata = RawArray(data=eeg_data, info=info)  

    Rawdata.notch_filter(50.0)
    Rawdata.resample(200, n_jobs=5)
    # signals = Rawdata.get_data(units='uV')
    signals = Rawdata.get_data()

    # t = np.arange(0, eeg.data.shape[1]) / eeg.fs
    hypMask = wave_algorithm(200, signals, XGB_model = XGB_mod)
    hypMask = np.repeat(hypMask,200)
    hyp = Annotations.loadMask(hypMask, 200)
    hyp.saveTsv(outFile)

