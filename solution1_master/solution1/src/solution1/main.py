import numpy as np
from epilepsy2bids.annotations import Annotations
from epilepsy2bids.eeg import Eeg



# from solution1_alg import gotman_algorithm
from solution1.solution1_alg import gotman_algorithm    # for Docker!!!

def main(edf_file, outFile):
    
    eeg = Eeg.loadEdfAutoDetectMontage(edfFile=edf_file)
    if eeg.montage is Eeg.Montage.UNIPOLAR:
        eeg.reReferenceToBipolar()

    t = np.arange(0, eeg.data.shape[1]) / eeg.fs
    hypMask = gotman_algorithm(t, eeg.data, eeg.fs)
    hyp = Annotations.loadMask(hypMask, eeg.fs)
    hyp.saveTsv(outFile)


#
# if __name__ == '__main__':
#     edf_path = "/media/public/Datasets/epilepsybenchmarks_chellenge/BIDS_Siena/sub-00/ses-01/eeg/sub-00_ses-01_task-szMonitoring_run-04_eeg.edf"
#     out_tsv= "/media/public/Datasets/epilepsybenchmarks_chellenge/BIDS_Siena_result_test/sub-00/ses-01/eeg/1.tsv"
#
#     main(edf_path, out_tsv)