# import numpy as np
# from epilepsy2bids.annotations import Annotations
# from epilepsy2bids.eeg import Eeg

from solution1.main_wave import main
from pathlib import Path
import os
import pickle
# from epilepsy2bids.annotations import Annotations
# from timescoring.annotations import Annotation
# import numpy as np
# from timescoring import scoring

# from solution1.solution1_alg import gotman_algorithm    # for Docker!!!


def process_dataset(
    input: Path, result: Path, is_wave_model=False
) -> bool:
    """
    run siezors detection from main for dataset input

    Returns:
        True if number of results file equal number of input edf and tsv
    """

    edf_count = 0
    tsv_count = 0
    result_count = 0
    if is_wave_model:
        XGB_mod = pickle.load(open('xgb_model_wav4.pkl', 'rb'))
    else:
        XGB_mod = None
        
    for subject in Path(input).glob("sub-*"):
        for ref_tsv in subject.glob("**/*.tsv"):  # use tsv for loop to be sure that we will have
            # test_edf="/media/public/Datasets/epilepsybenchmarks_chellenge/tuh_train_preprocess/sub-592/ses-07/eeg/sub-592_ses-07_task-szMonitoring_run-07_eeg.edf"
            # test_tsv = "/media/public/Datasets/epilepsybenchmarks_chellenge/tuh_train_preprocess/sub-592/ses-07/eeg/sub-592_ses-07_task-szMonitoring_run-07_events.tsv"
            # main(str(test_edf), test_tsv, XGB_mod=XGB_mod)
            # exit(-555)

            print(ref_tsv)
            last_parts = list(ref_tsv.parts)[-4:]
            last_path = Path(*last_parts)
            res_tsv_name = os.path.join(result,last_path)
            os.makedirs(os.path.dirname(res_tsv_name), exist_ok=True)

            edf_path = ref_tsv
            edf_path = str(edf_path)[:-10]+'eeg.edf'   #  replace "events.tsv" to "eeg.edf"
            main(str(edf_path), res_tsv_name, XGB_mod=XGB_mod)

            if os.path.exists(ref_tsv):
                tsv_count += 1
            if os.path.exists(edf_path):
                edf_count += 1
            if os.path.exists(res_tsv_name):
                result_count += 1

    if tsv_count == edf_count == result_count:
        return True
    else:
        return False


if __name__ == '__main__':
    # input = "/media/public/Datasets/epilepsybenchmarks_chellenge/BIDS_Siena_test"
    # result = "/media/public/Datasets/epilepsybenchmarks_chellenge/BIDS_Siena_result_wave"
    # input = "/media/public/Datasets/epilepsybenchmarks_chellenge/BIDS_CHB-MIT"
    # result = "/media/public/Datasets/epilepsybenchmarks_chellenge/BIDS_CHB-MIT_result"
    # input = "/media/public/Datasets/epilepsybenchmarks_chellenge/tuh_train_preprocess"
    # result = "/media/public/Datasets/epilepsybenchmarks_chellenge/tuh_train_preprocess_result_baseline"


    input = "/media/public/Datasets/epilepsybenchmarks_chellenge/tuh_train_preprocess"
    result = "/media/public/Datasets/epilepsybenchmarks_chellenge/tuh_train_preprocess_result_wave"


    process_dataset(input, result, is_wave_model=True)