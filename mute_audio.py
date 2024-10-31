import glob
from tqdm import tqdm
from pydub import AudioSegment


data_path = "/nobackup/le/LoCoNet/AVA_Dataset/orig_audios/trainval"
new_data_path = "/nobackup/le/LoCoNet/AVA_Dataset/orig_audios_mute/trainval"

files = glob.glob(data_path + "/*")

def reverse(file):
    original = AudioSegment.from_wav(file)
    muted_audio = original - 100 
    muted_audio.export(new_data_path + "/" + file.split('/')[-1], format = "wav")


for file in tqdm(files, total = len(files)):
    reverse(file)