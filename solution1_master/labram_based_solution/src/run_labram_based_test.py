# import numpy as np
# from epilepsy2bids.annotations import Annotations
# from epilepsy2bids.eeg import Eeg

from solution1.main_labram_test import main
import solution1.utils as utils
import solution1.run_class_finetuning_sz_chlng_2025 as run_class_finetuning_sz_chlng_2025

from collections import OrderedDict
import torch
from pathlib import Path
import os
import pickle
# from epilepsy2bids.annotations import Annotations
# from timescoring.annotations import Annotation
# import numpy as np
# from timescoring import scoring

# from solution1.solution1_alg import gotman_algorithm    # for Docker!!!

def labram_model_load():
    # model loading. steps::
    # 1) get model
    # 2) load_state_dict
    # 3) load best finetuned



    args, ds_init = run_class_finetuning_sz_chlng_2025.get_args()
    utils.init_distributed_mode(args)
    args.nb_classes = 6
    model = run_class_finetuning_sz_chlng_2025.get_models(args)
    patch_size = model.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (1, args.input_size // patch_size)
    args.patch_size = patch_size
    
    if args.finetune:  # --finetune ./checkpoints/labram-base.pth \
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        if (checkpoint_model is not None) and (args.model_filter_name != ''):
            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('student.'):
                    new_dict[key[8:]] = checkpoint_model[key]
                else:
                    pass
            checkpoint_model = new_dict

        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    # load best finetuned
    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model,
        optimizer=None, loss_scaler=None, model_ema=None)
    # utils.auto_load_model(
    #     args=args, model=model, model_without_ddp=model_without_ddp,
    #     optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    # [end] model loading
    device = torch.device(args.device)

    return model, device

def labram_model_load_short():
    # 1) get model
    # 2) load best finetuned
    args, ds_init = run_class_finetuning_sz_chlng_2025.get_args()
    utils.init_distributed_mode(args)
    model = run_class_finetuning_sz_chlng_2025.get_models(args)
    # utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
    # load best finetuned
    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model,
        optimizer=None, loss_scaler=None, model_ema=None)

    return model

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

    model, device = labram_model_load()

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
            main(str(edf_path), res_tsv_name, model, device)

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
    result = "/media/public/Datasets/epilepsybenchmarks_chellenge/tuh_train_preprocess_labram"


    process_dataset(input, result)