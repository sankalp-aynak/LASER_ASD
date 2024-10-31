from dataLoader_multiperson_landmark import train_loader
from utils.tools import *
from dlhammer.dlhammer import bootstrap

cfg = bootstrap(print_cfg=False)
    
cfg = init_args(cfg)

loader = train_loader(cfg, trialFileName = cfg.trainTrialAVA, \
                          audioPath      = os.path.join(cfg.audioPathAVA , 'train'), \
                          visualPath     = os.path.join(cfg.visualPathAVA, 'train'), \
                          num_speakers=cfg.MODEL.NUM_SPEAKERS,
                          )

for i in range(1000):
    print(f"finish {i}")
    loader.__getitem__(i)