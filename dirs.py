import os
import platform

DEF_LOCAL_DIR = os.path.dirname(os.path.realpath(__file__))
DEF_DATA_DIR = os.path.join(DEF_LOCAL_DIR, 'dataset')
DEF_CTW_DIR = os.path.join(DEF_DATA_DIR, 'ctw')
DEF_SVT_DIR = os.path.join(DEF_DATA_DIR, 'svt', 'img')
DEF_TTT_DIR = os.path.join(DEF_DATA_DIR, 'totaltext')
DEF_IC13_DIR = os.path.join(DEF_DATA_DIR, 'ICDAR2013')
DEF_IC15_DIR = os.path.join(DEF_DATA_DIR, 'ICDAR2015')
DEF_IC19_DIR = os.path.join(DEF_DATA_DIR, 'ICDAR2019')
DEF_MSRA_DIR = os.path.join(DEF_DATA_DIR, 'MSRA-TD500')
DEF_ICV15_DIR = os.path.join(DEF_DATA_DIR, 'ICDAR2015_video')
DEF_MINE_DIR = os.path.join(DEF_DATA_DIR, 'minetto')

if(platform.system().lower()[:7]=='windows'):
    DEF_SYN_DIR = "D:\\development\\SynthText"
elif(os.path.exists("/BACKUP/yom_backup/SynthText")):
    DEF_SYN_DIR = "/BACKUP/yom_backup/SynthText"
else:
    DEF_SYN_DIR = os.path.join(DEF_DATA_DIR, 'SynthText')

DEF_WORK_DIR = "/BACKUP/yom_backup" if(platform.system().lower()[:7]!='windows' and os.path.exists("/BACKUP/yom_backup"))else DEF_LOCAL_DIR