import glob

import lib
from scipy.io.wavfile import read

cepstrum_vector_dict = lib.generate_cepstrum_vector_dict()

correct_file_list = []
incorrect_file_list = []

for speaker in lib.speaker_list:
    for filename in glob.glob(
        lib.DIR_DATA_TEST + speaker + "/" + speaker + "*" + lib.EXT_WAV
    ):
        _, data = read(filename)
        speaker_identified = lib.identify_speaker(cepstrum_vector_dict, data)
        if speaker_identified == speaker:
            correct_file_list.append(filename)
        else:
            incorrect_file_list.append(filename)

num_correct = len(correct_file_list)
num_incorrect = len(incorrect_file_list)
accuracy = num_correct / (num_correct + num_incorrect)

print(f"correct: {num_correct}")
print(f"incorrect: {num_incorrect}")
print(f"accuracy: {accuracy}")

incorrect_spk = {spk: 0 for spk in lib.speaker_list}
for filename in incorrect_file_list:
    for spk in lib.speaker_list:
        if spk in filename:
            incorrect_spk[spk] += 1
print(*incorrect_spk.items(), sep="\n")
