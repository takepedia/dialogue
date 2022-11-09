import glob

import numpy as np
from scipy.io.wavfile import read

DIR_DATA_TRAIN = "../asset/new_vowels/train/"
DIR_DATA_TEST = "../asset/new_vowels/test/"
DIR_OUTPUT = "../output"
EXT_WAV = ".wav"
WINDOW_SIZE = 512
SAMPLING_RATE = 16000

speaker_list = ["spk1", "spk2", "spk3", "spk4"]
cepstrum_vector_dim = 15  # ケプストラムベクトルの次数
trim_threshold = 500  # トリミングで平均値がこれ以下である部分を切る
trim_convovlve_size = 1000  # トリミングで取る平均の個数
cepstrum_index = 10  # ケプストラムの計算でローパスリフタを掛ける時に残す個数


def trim_data(data):
    """
    data のうち，連続する trim_convolve_size 個の平均値が trim_threshold 以下である部分を
    前後から削除する．

    Parameters
    ----------
    data : numpy.ndarray
        トリミングするデータ

    Returns
    -------
    numpy.ndarray
        トリミングされたデータ
    """
    data_abs = np.abs(data)  # マイナスもあるので絶対値を取る
    convolve_window = np.ones(trim_convovlve_size) / trim_convovlve_size
    section_average = np.convolve(data_abs, convolve_window, mode="same")
    not_silent_index = np.where(section_average > trim_threshold)
    if len(not_silent_index[0]) > 0:
        trimmed_data = data[not_silent_index[0][0] : not_silent_index[0][-1]]
    else:
        trimmed_data = data
    return trimmed_data


def window_data(data: np.ndarray) -> np.ndarray:
    """
    data に hamming 窓を掛ける

    Parameters
    ----------
    data : numpy.ndarray
        窓を掛けるデータ

    Returns
    -------
    numpy.ndarray
        hamming 窓を掛けたデータ
    """
    window_func = np.hamming(data.size)
    windowed_data = window_func * data
    return windowed_data


def calc_cepstrum_coef(data: np.ndarray) -> np.ndarray:
    """
    data のケプストラム係数を求める．FFT -> log power spec -> IFFT

    Parameters
    ----------
    data : numpy.ndarray
        データ

    Returns
    -------
    numpy.ndarray
        データのケプストラム係数
    """
    data_size = data.size
    data_windowed = window_data(data)
    data_fft = np.fft.fft(data_windowed)
    data_power_spectrum = np.log10(np.absolute(data_fft) ** 2 / data_size)
    data_cepstrum_coef = np.real(np.fft.ifft(data_power_spectrum, data_size))
    return data_cepstrum_coef


def calc_cepstrum_vector(data):
    """
    data のケプストラムベクトルを求める．

    Parameters
    ----------
    data : numpy.ndarray
        データ

    Returns
    -------
    numpy.ndarray
        データのケプストラムベクトル
    """
    data_size = data.size
    index = 0
    num_skip = 0
    cepstrum_vector = np.zeros(cepstrum_vector_dim)

    while index + WINDOW_SIZE <= data_size:
        data_using = data[index : index + WINDOW_SIZE]
        cepstrum_coefs = calc_cepstrum_coef(data_using)
        cepstrum_vector += cepstrum_coefs[1 : cepstrum_vector_dim + 1]
        num_skip += 1
        index += WINDOW_SIZE // 2

    if num_skip > 0:
        cepstrum_vector = cepstrum_vector / num_skip
    return cepstrum_vector


def calc_euclid_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    ２つのベクトルのユークリッド距離を求める．

    Parameters
    ----------
    vec1 : numpy.ndarray
        ベクトル１
    vec2 : numpy.ndarray
        ベクトル２

    Returns
    -------
    float
        vec1 と vec2 のユークリッド距離
    """
    return np.linalg.norm(vec1 - vec2)


def generate_cepstrum_vector_dict() -> dict:
    """訓練データから辞書ベクトルを作成する

    Returns
    -------
    dict
        辞書ベクトル
    """
    cepstrum_vector_dict = {}
    for speaker in speaker_list:
        cepstrum_vector = np.zeros(cepstrum_vector_dim)
        num_file = 0
        for filename in glob.glob(
            DIR_DATA_TRAIN + speaker + "/" + speaker + "*" + EXT_WAV
        ):
            _, data = read(filename)
            trimmed_data = trim_data(data)
            cepstrum_vector += calc_cepstrum_vector(trimmed_data)
            num_file += 1
        cepstrum_vector_dict[speaker] = cepstrum_vector / num_file
    return cepstrum_vector_dict


def identify_speaker(cepstrum_vector_dict: dict, data_test: np.ndarray) -> str:
    """辞書データとテストデータから話者を推定する

    Parameters
    ----------
    cepstrum_vector_dict : dict
        辞書データ
    data_test : np.ndarray
        テストデータ

    Returns
    -------
    str
        推定された話者名
    """
    data_test_trimmed = trim_data(data_test)
    data_size_test_trimmed = data_test_trimmed.size

    dist_dict = {speaker: 0 for speaker in cepstrum_vector_dict.keys()}
    index = 0
    repeat_time = (data_size_test_trimmed - WINDOW_SIZE) // (WINDOW_SIZE // 2)
    for _ in range(repeat_time):
        cepstrum_vector_test = calc_cepstrum_vector(
            data_test_trimmed[index : index + WINDOW_SIZE]
        )
        for speaker, cepstrum_vector_by_speaker in cepstrum_vector_dict.items():
            dist = calc_euclid_distance(
                cepstrum_vector_by_speaker, cepstrum_vector_test
            )
            dist_dict[speaker] += dist
        index += WINDOW_SIZE // 2
    dist_min_speaker = min(dist_dict, key=dist_dict.get)
    return dist_min_speaker
