# import numpy as np
# from epilepsy2bids.annotations import Annotations
# from epilepsy2bids.eeg import Eeg

from solution1.main import main
from pathlib import Path
import os
# from epilepsy2bids.annotations import Annotations
# from timescoring.annotations import Annotation
# import numpy as np
# from timescoring import scoring

# from solution1.solution1_alg import gotman_algorithm    # for Docker!!!


def process_dataset(
    input: Path, result: Path
) -> bool:
    """
    run siezors detection from main for dataset input

    Returns:
        True if number of results file equal number of input edf and tsv
    """

    edf_count = 0
    tsv_count = 0
    result_count = 0

    for subject in Path(input).glob("sub-*"):
        for ref_tsv in subject.glob("**/*.tsv"):  # use tsv for loop to be sure that we will have
            print(ref_tsv)
            last_parts = list(ref_tsv.parts)[-4:]
            last_path = Path(*last_parts)
            res_tsv_name = os.path.join(result,last_path)
            os.makedirs(os.path.dirname(res_tsv_name), exist_ok=True)

            edf_path = ref_tsv
            edf_path = str(edf_path)[:-10]+'eeg.edf'   #  replace "events.tsv" to "eeg.edf"
            main(str(edf_path), res_tsv_name)

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
    input = "/media/public/Datasets/epilepsybenchmarks_chellenge/BIDS_Siena"
    result = "/media/public/Datasets/epilepsybenchmarks_chellenge/BIDS_Siena_result"

    process_dataset(input, result)