import os
import subprocess
from torch.utils.tensorboard import SummaryWriter
import torchaudio

writer = SummaryWriter("results/")
spk = input("Speaker:")
key = input("Key:")
for model in os.listdir("logs/44k"):
    if "G_" in model and model != "G_0.pth":
        model_itera = model.split('_')[1].split('.')[0]
        str = "python inference_main.py -m \"" + os.path.join("logs/44k", model) + "\"" \
                + " -c \"configs/config.json\" -n \"富士山下.wav\" -t "+ key +" -s \"" \
                + spk + "\" -fmp"
        print(str)
        os.system(str)

for wav in os.listdir("results"):
    if ".flac" in wav:
        audio, _ = torchaudio.load(os.path.join("results", wav))
        writer.add_audio("iter:{}".format(wav.split('G')[1].split('_')[1]),audio, 0)

writer.close()
