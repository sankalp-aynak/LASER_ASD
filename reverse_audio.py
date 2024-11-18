import glob
from tqdm import tqdm
from pydub import AudioSegment
import os
import pandas as pd

data_path = "/nobackup/le/LoCoNet/ASW_Dataset/clips_audios/test"
new_data_path = "/nobackup/le/LoCoNet/ASW_Dataset/clips_audios_reverse/test_reverse_1"

def reverse(file, save_path):
    original = AudioSegment.from_wav(file)
    reverse = original.reverse()
    reverse.export(save_path, format = "wav")


for videoName in tqdm(os.listdir(data_path)):
    audio_data_path = data_path + "/" + videoName
    files = os.listdir(audio_data_path)
    for file in files:
        audio_file_path = audio_data_path + "/" + file
        # print(audio_file_path)
        # print(new_data_path + "/" + videoName + "/" + file)
        # quit()
        os.makedirs(new_data_path + "/" + videoName, exist_ok=True)
        reverse(audio_file_path, new_data_path + "/" + videoName + "/" + file)

df = pd.read_csv("/nobackup/le/LoCoNet/ASW_Dataset/csv/test_orig.csv")
df['label'] = "NOT_SPEAKING"
df.to_csv("/nobackup/le/LoCoNet/ASW_Dataset/csv/test_orig_modified.csv")