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

# from epilepsy2bids.eeg import Eeg
# import numpy as np
# import mne


def process_chunk(model, device, chunk, sfreq, channel_names):
    """
    Обрабатывает один чанк данных.
    :param model: модель для обработки
    :param device: устройство для выполнения модели
    :param chunk: ndarray с данными одного чанка
    :param sfreq: частота дискретизации
    :param channel_names: имена каналов
    :return: гипноз-маска для данного чанка
    """
    info = mne.create_info(channel_names, sfreq, ch_types="eeg")
    raw_data = mne.io.RawArray(chunk, info)

    raw_data.notch_filter(50.0)
    raw_data.resample(200, n_jobs=5)

    signals = raw_data.get_data()
    hyp_mask_chunk = labram_algorithm(model, device, 200, signals, ch_names=channel_names)
    hyp_mask_chunk = np.repeat(hyp_mask_chunk, 200)

    return hyp_mask_chunk


def merge_results(current_result, new_result, overlap_secs, fs):
    """
    Объединение текущих результатов с новым результатом с учётом перекрытия.
    :param current_result: текущие результаты
    :param new_result: новый результат
    :param overlap_secs: перекрытие в секундах
    :param fs: частота дискретизации
    :return: объединённые результаты
    """
    half_overlap_size = int(overlap_secs/2 * fs)

    # Последние 5 секунд текущего чанка заменяем на соответствующие 5 секунд из нового чанка
    current_result_without_half_overlap = current_result[:-half_overlap_size]
    remaining_part = new_result[half_overlap_size:]

    current_result = np.concatenate([current_result_without_half_overlap, remaining_part])

    return current_result


def main(edf_file, outFile, model, device):
    model.eval()
    eeg = Eeg.loadEdfAutoDetectMontage(edfFile=edf_file)
    if eeg.montage is Eeg.Montage.BIPOLAR:
        print("Model needs unipolar. Eeg Montage BIPOLAR: ", edf_file)  # can't convert to unipolar
        exit(0)

    channel_names = standard_1020_subset
    sfreq = eeg.fs
    total_seconds = int(eeg.data.shape[1] / sfreq)
    info = mne.create_info(channel_names, sfreq, ch_types="eeg")

    chunk_duration_hrs = 0.5
    overlap_secs = 10

    chunk_size = int(chunk_duration_hrs * 60 * 60 * sfreq)  # Размер чанка в образцах
    overlap_size = int(overlap_secs * sfreq)  # Размер перекрытия в образцах

    start_idx = 0
    final_hyp_mask = np.array([])  # Начальный пустой массив для хранения результатов

    while start_idx + chunk_size <= total_seconds * sfreq:
        end_idx = start_idx + chunk_size
        eeg_data =  eeg.data[:, start_idx:end_idx]
        Rawdata = RawArray(data=eeg_data, info=info)
        Rawdata.notch_filter(50.0)
        new_fs = 200
        Rawdata.resample(new_fs, n_jobs=5)
        chunk = Rawdata.get_data()
        
        hyp_mask_chunk = process_chunk(model, device, chunk, new_fs, channel_names)

        if final_hyp_mask.size == 0:
            final_hyp_mask = hyp_mask_chunk
        else:
            final_hyp_mask = merge_results(final_hyp_mask, hyp_mask_chunk, overlap_secs, new_fs)

        # Сдвигаем начало следующего чанка с учётом перекрытия
        start_idx += chunk_size - overlap_size

    # Последний кусок, если остался
    if start_idx < total_seconds * sfreq:
        eeg_data =  eeg.data[:, start_idx:eeg.data.shape[1]]
        Rawdata = RawArray(data=eeg_data, info=info)
        Rawdata.notch_filter(50.0)
        new_fs = 200
        Rawdata.resample(new_fs, n_jobs=5)
        last_chunk = Rawdata.get_data()

        last_hyp_mask_chunk = process_chunk(model, device, last_chunk, new_fs, channel_names)
        if final_hyp_mask.size == 0:
            final_hyp_mask = last_hyp_mask_chunk
        else:
            final_hyp_mask = merge_results(final_hyp_mask, last_hyp_mask_chunk, overlap_secs, new_fs)

    # Сохраняем результат
    hyp = Annotations.loadMask(final_hyp_mask, 200)
    hyp.saveTsv(outFile)

# def main(edf_file, outFile, model, device):
#     model.eval()
#     eeg = Eeg.loadEdfAutoDetectMontage(edfFile=edf_file)
#     if eeg.montage is Eeg.Montage.BIPOLAR:
#         print("Model needs unipolar. Eeg Montage BIPOLAR: ", edf_file)  # cant convert to unipolar
#         exit(0)
#     eeg_data = eeg.data
#     channel_names = standard_1020_subset  #  replace eeg.channels to remove  "-Avg"  here could be additional check that FP1-Avg => FP1 etc.. but not nessesary for chellenge conditions
#     sfreq = eeg.fs
#     info = mne.create_info(channel_names, sfreq, ch_types="eeg")
#     Rawdata = RawArray(data=eeg_data, info=info)
#
#     Rawdata.notch_filter(50.0)
#     Rawdata.resample(200, n_jobs=5)
#
#     signals = Rawdata.get_data()
#
#     # t = np.arange(0, eeg.data.shape[1]) / eeg.fs
#     hypMask = labram_algorithm(model, device, 200, signals, ch_names=standard_1020_subset)
#     hypMask = np.repeat(hypMask,200)
#     hyp = Annotations.loadMask(hypMask, 200)
#     hyp.saveTsv(outFile)

