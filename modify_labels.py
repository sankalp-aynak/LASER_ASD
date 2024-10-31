import pandas as pd

df = pd.read_csv("/nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig.csv")

df['label_id'] = 0
df['label'] = "NOT_SPEAKING"

df.to_csv("/nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv")