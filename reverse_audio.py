import glob
from tqdm import tqdm
from pydub import AudioSegment


data_path = "/nobackup/le/LoCoNet/AVA_Dataset/orig_audios/trainval"
new_data_path = "/nobackup/le/LoCoNet/AVA_Dataset/orig_audios_reverse/trainval"

files = glob.glob(data_path + "/*")

def reverse(file):
    original = AudioSegment.from_wav(file)
    reverse = original.reverse()
    reverse.export(new_data_path + "/" + file.split('/')[-1], format = "wav")


for file in tqdm(files, total = len(files)):
    reverse(file)