import numpy as np
from epilepsy2bids.annotations import Annotations
from epilepsy2bids.eeg import Eeg

from solution1.solution1_wave import wave_algorithm


def main(edf_file, outFile):
    eeg = Eeg.loadEdfAutoDetectMontage(edfFile=edf_file)
    if eeg.montage is Eeg.Montage.UNIPOLAR:
        eeg.reReferenceToBipolar()

    Rawdata = eeg.to_mne()
    Rawdata.notch_filter(50.0)
    Rawdata.resample(200, n_jobs=5)
    signals = Rawdata.get_data(units='uV')

    t = np.arange(0, eeg.data.shape[1]) / eeg.fs
    hypMask = wave_algorithm(t, eeg.data, eeg.fs, signals)
    hyp = Annotations.loadMask(hypMask, eeg.fs)
    hyp.saveTsv(outFile)

