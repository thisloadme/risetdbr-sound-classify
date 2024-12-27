import os
import librosa
import numpy as np
import random
import json

current_directory = os.getcwd()

audio_list_path = current_directory + "/audio"
onlyfilesdata = [f for f in os.listdir(audio_list_path) if os.path.isfile(os.path.join(audio_list_path, f))]

random.shuffle(onlyfilesdata)
audio_data = onlyfilesdata

SAMPLING_RATE = 44100
MFCC_MAX_LEN = 160
MFCC_NUM = 20
mfcc_array = []
audio_target = []

target2idx = {
    "cat": 0,
    "dog": 1
}

base_array = np.zeros((MFCC_NUM, MFCC_MAX_LEN))
for idx, audio_file in enumerate(audio_data):

    split_data = audio_file.split('_')
    tag = split_data[0]

    arr_target = np.zeros(len(target2idx))
    arr_target[target2idx[tag]] = 1

    audio_target.append(arr_target.tolist())
    
    audio_path = audio_list_path + '/' + audio_file
    x, sr = librosa.load(audio_path)

    mfccs = librosa.feature.mfcc(y=x, n_mfcc=MFCC_NUM)
    len_audio = min(len(mfccs[0]), MFCC_MAX_LEN)

    new_mfcc = base_array
    new_mfcc[:, :len_audio] = mfccs[:, :len_audio]

    mfcc_array.append(new_mfcc.tolist())
    print(idx)

with open(current_directory + '/mfcc_array_20.json', 'w', encoding='utf-8') as f:
    array_data = {
        "mfcc_array": mfcc_array,
        "audio_target": audio_target
    }

    json.dump(array_data, f)