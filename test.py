from dataset.semi import SemiDataset
from model.semseg.deeplabv2 import DeepLabV2
from utils.dataset_name import *
from utils.utils import count_params, meanIOU, color_map
from utils.dataset_name import *
import argparse
from copy import deepcopy
import numpy as np
import os
from PIL import Image
import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.models.SEUNet import *
os.environ["CUDA_VISIBLE_DEVICES"] = " 6, 7"


NUM_CLASSES = {'DFC22': 12, 'iSAID': 15, 'MER': 9, 'MSL': 9, 'Vaihingen': 5, 'GID-15': 15}
DATASET = 'GID-15'     # ['DFC22', 'iSAID', 'MER', 'MSL', 'Vaihingen', 'GID-15']


DFC22_class, DFC22_color_map = DFC22()


# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/Vaihingen/1-8_20_1/SDC/deeplabv2_resnet101_71.75.pth'
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/DFC22/1-8_0.5/dualmatch_SGM/adaptive_decay_1.0_to_0.5/SACM_2/SGM/' \
#           'ResUNet_resnet50_dualmatch_SGM_0.00_adaptive_decay_1.00_to_0.50_weight_0.50_epoch_111_42.56.pth'
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/GID-15/1-4_0.5/dualmatch/adaptive_decay_1.0_to_0.9/SACM_6/' \
#           'ResUNet_resnet50_dualmatch_adaptive_decay_1.00_to_0.90_weight_0.50_epoch_51_80.16.pth'

# MEMORY_WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/GID-15/1-4_0.5/train_prototype/_1.0_to_0.8/_/SGM/' \
#           'ResUNet_resnet50_train_prototype_0.50__1.00_to_0.80_weight_0.50_epoch_1_73.25.pth'
# WEIGHTS = '/data1/users/lvliang/project_123/ClassHyPer-master/ClassHyPer-master/splits/vaihingen/1_16/save/SEResUNet/SGMAC_SGM_EMA0.5/checkpoint-best.pth'
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/dataset/splits/Vaihingen/1-8/save/SEResUNet/20230828_214457_seed1234/checkpoint-best.pth'
# MEMORY_WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/DFC22/1-8_0.5/train_SGM/_1.0_to_0.5/_/SGM/' \
#           'ResUNet_resnet50_train_SGM_0.10__1.00_to_0.50_weight_0.50_epoch_4_42.24.pth'
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/DFC22/1-8_0.5/train_SGM/_1.0_to_0.5/_/SGM/' \
#           'ResUNet_resnet50_train_SGM_0.10__1.00_to_0.50_weight_0.50_epoch_4_42.24.pth'

# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/DFC22/1-8_20_0.5/SDC/ResUNet_resnet50_dualmatch_adaptive_decay_0.95_to_0.8_weight_0.50_epoch_111_41.41.pth'
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/DFC22/1-8_20_0.5/SDC/ResUNet_resnet50_fixmatch_single_0.95_weight_0.50_withoutCutmix_epoch_114_38.63.pth'
# WEIGHTS = '/data1/users/lvliang/project_123/LSST-master/LSST-master/output/DFC22/1-8_0.2/models/ResUNet_resnet50_38.78.pth'

# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/DFC22/1-8_0.5/RanPaste/__to_/_/ResUNet_resnet50_RanPaste_epoch_192_36.45.pth'
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/DFC22/1-8_0.5/RanPaste/__to_/_/ResUNet_resnet50_RanPaste_epoch_192_36.45.pth'
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/DFC22/1-8_0.5/RanPaste/__to_/_/ResUNet_resnet50_RanPaste_epoch_192_36.45.pth'
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/DFC22/1-8_0.5/RanPaste/__to_/_/ResUNet_resnet50_RanPaste_epoch_192_36.45.pth'
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/DFC22/1-8_20_1/SDC/ResUNet_resnet50_my_strategy_sup_only_33.87.pth'

#DFC22 1-4
#sup_only
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/DFC22/1-4_0.5/sup_only/__to_/_/ResUNet_resnet50_39.27.pth'
#FixMatch
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/DFC22/1-4_0.5/fixmatch/single_0.95_to_0.95/_/ResUNet_resnet50_fixmatch_single_0.95_weight_0.50_epoch_87_39.27.pth'
#ICNet
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/DFC22/1-4_0.5/icnet/__to_/_/ResUNet_resnet50_icnet_epoch_104_35.92.pth'
#RanPaste
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/DFC22/1-4_0.5/RanPaste/__to_/_/ResUNet_resnet50_icnet_ema_epoch_146_36.82.pth'
#LSST
# WEIGHTS = '/data1/users/lvliang/project_123/LSST-master/LSST-master/output/DFC22/1-4_0.2/models/ResUNet_resnet50_40.71.pth'
#WSCL
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/DFC22/1-4_0.5/WSCL/__to_/SDC_/ResUNet_resnet50_40.97.pth'



#iSAID 100
#sup_only
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/iSAID/100_0.5/sup_only/_0.0_to_0.0/_/ResUNet_resnet50_58.48.pth'
#FixMatch
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/iSAID/100_0.5/fixmatch/single_0.95_to_0.95/_/' \
#           'ResUNet_resnet50_fixmatch_single_0.95_weight_0.50_epoch_50_65.30.pth'
#ICNet
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/iSAID/100_0.5/icnet/__to_/_/' \
#           'ResUNet_resnet50_icnet_d_epoch_35_61.73.pth'

#RanPaste
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/iSAID/100_0.5/RanPaste/__to_/_/' \
#           'ResUNet_resnet50_RanPaste_epoch_65_66.68.pth'

#LSST
# WEIGHTS = '/data1/users/lvliang/project_123/LSST-master/LSST-master/output/iSAID/100_0.2/models/ResUNet_resnet50_sup_pretrained_66.16.pth'
#WSCL
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/iSAID/100_0.5/WSCL/__to_/SDC_/' \
#           'ResUNet_resnet50_65.52.pth'
#SGMCR
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/iSAID/100_0.5/train_SGM/_1.0_to_0.5/_/SGM/' \
#           'ResUNet_resnet50_train_SGM_0.10__1.00_to_0.50_weight_0.50_epoch_28_70.13.pth'
# MEMORY_WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/iSAID/100_0.5/train_SGM/_1.0_to_0.5/_/SGM/' \
#           'ResUNet_resnet50_train_SGM_0.10__1.00_to_0.50_weight_0.50_epoch_28_70.13.pth'

#iSAID 300
#sup_only
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/iSAID/300_0.5/sup_only/__to_/_/ResUNet_resnet50_74.31.pth'
#FixMatch
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/iSAID/300_0.5/fixmatch/single_0.95_to_0.95/_/' \
#           'ResUNet_resnet50_fixmatch_single_0.95_weight_0.50_epoch_120_76.08.pth'
#ICNet
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/iSAID/300_0.5/icnet/__to_/_/' \
#           'ResUNet_resnet50_icnet_epoch_47_71.79.pth'

#RanPaste
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/iSAID/300_0.5/RanPaste/__to_/_/' \
#           'ResUNet_resnet50_RanPaste_epoch_74_76.24.pth'

#LSST
# WEIGHTS = '/data1/users/lvliang/project_123/LSST-master/LSST-master/output/iSAID/300_0.2/models/' \
#           'ResUNet_resnet50_77.16.pth'
#WSCL
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/iSAID/300_0.5/WSCL/__to_/SDC_/' \
#           'ResUNet_resnet50_76.55.pth'
#SGMCR
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/iSAID/300_0.5/train_SGM/_1.0_to_0.5/_/SGM/' \
#           'ResUNet_resnet50_train_SGM_0.20__1.00_to_0.50_weight_0.50_epoch_45_80.18.pth'
# MEMORY_WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/iSAID/300_0.5/train_SGM/_1.0_to_0.5/_/SGM/' \
#           'ResUNet_resnet50_train_SGM_0.20__1.00_to_0.50_weight_0.50_epoch_45_80.18.pth'


#MER 1-8
#sup_only
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/MER/1-8_0.5/sup_only/__to_/_/ResUNet_resnet50_53.54.pth'
#FixMatch
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/MER/1-8_0.5/fixmatch/single_0.95_to_0.95/_/' \
#           'ResUNet_resnet50_fixmatch_single_0.95_weight_0.50_epoch_163_55.31.pth'
#ICNet
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/MER/1-8_0.5/icnet/__to_/_/' \
#           'ResUNet_resnet50_icnet_epoch_82_51.40.pth'

#RanPaste
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/MER/1-8_0.5/RanPaste/__to_/_/' \
#           'ResUNet_resnet50_RanPaste_epoch_123_54.82.pth'

#LSST
# WEIGHTS = '/data1/users/lvliang/project_123/LSST-master/LSST-master/output/MER/1-8_0.2/models/' \
#          'ResUNet_resnet50_55.78.pth'
#WSCL
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/MER/1-4_0.5/WSCL/__to_/SDC_/' \
#           'ResUNet_resnet50_58.31.pth'
#SGMCR
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/iSAID/300_0.5/train_SGM/_1.0_to_0.5/_/SGM/' \
#           'ResUNet_resnet50_train_SGM_0.20__1.00_to_0.50_weight_0.50_epoch_45_80.18.pth'
# MEMORY_WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/iSAID/300_0.5/train_SGM/_1.0_to_0.5/_/SGM/' \
#           'ResUNet_resnet50_train_SGM_0.20__1.00_to_0.50_weight_0.50_epoch_45_80.18.pth'




#MER 1-4
#sup_only
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/MER/1-4_0.5/sup_only/__to_/_/ResUNet_resnet50_56.29.pth'
#FixMatch
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/MER/1-4_0.5/fixmatch/single_0.95_to_0.95/_/' \
#           'ResUNet_resnet50_fixmatch_single_0.95_weight_0.50_epoch_203_57.54.pth'
#ICNet
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/MER/1-4_0.5/icnet/__to_/_/' \
#           'ResUNet_resnet50_icnet_d_epoch_120_53.07.pth'

#RanPaste
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/MER/1-4_0.5/RanPaste/__to_/_/' \
#           'ResUNet_resnet50_RanPaste_epoch_178_57.25.pth'

#LSST
# WEIGHTS = '/data1/users/lvliang/project_123/LSST-master/LSST-master/output/MER/1-4_0.2/models/' \
#          'ResUNet_resnet50_56.86.pth'
#WSCL
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/MER/1-4_0.5/WSCL/__to_/SDC_/ResUNet_resnet50_58.31.pth'
# SGMCR
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/iSAID/300_0.5/train_SGM/_1.0_to_0.5/_/SGM/' \
#           'ResUNet_resnet50_train_SGM_0.20__1.00_to_0.50_weight_0.50_epoch_45_80.18.pth'
# MEMORY_WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/iSAID/300_0.5/train_SGM/_1.0_to_0.5/_/SGM/' \
#           'ResUNet_resnet50_train_SGM_0.20__1.00_to_0.50_weight_0.50_epoch_45_80.18.pth'



#MSL 1-8
#sup_only
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/MSL/1-8_0.5/sup_only/__to_/_/ResUNet_resnet50_56.99.pth'
#FixMatch
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/MSL/1-8_0.5/fixmatch/single_0.95_to_0.95/_/' \
#           'ResUNet_resnet50_fixmatch_single_0.95_weight_0.50_epoch_17_56.27.pth'
#ICNet
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/MER/1-4_0.5/icnet/__to_/_/' \
#           'ResUNet_resnet50_icnet_d_epoch_120_53.07.pth'

#RanPaste
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/MSL/1-8_0.5/RanPaste/__to_/_/' \
#           'ResUNet_resnet50_RanPaste_epoch_33_56.60.pth'

#LSST
# WEIGHTS = '/data1/users/lvliang/project_123/LSST-master/LSST-master/output/MSL/1-8_0.2/models/' \
#          'ResUNet_resnet50_57.80.pth'
#WSCL
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/MSL/1-8_0.5/WSCL/__to_/SDC_/ResUNet_resnet50_59.31.pth'
# SGMCR
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/iSAID/300_0.5/train_SGM/_1.0_to_0.5/_/SGM/' \
#           'ResUNet_resnet50_train_SGM_0.20__1.00_to_0.50_weight_0.50_epoch_45_80.18.pth'
# MEMORY_WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/iSAID/300_0.5/train_SGM/_1.0_to_0.5/_/SGM/' \
#           'ResUNet_resnet50_train_SGM_0.20__1.00_to_0.50_weight_0.50_epoch_45_80.18.pth'


#MSL 1-4
#sup_only
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/MSL/1-4_0.5/sup_only/__to_/_/ResUNet_resnet50_56.22.pth'
#FixMatch
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/MSL/1-4_0.5/fixmatch/single_0.95_to_0.95/_/' \
#           'ResUNet_resnet50_fixmatch_single_0.95_weight_0.50_epoch_84_57.65.pth'
#ICNet
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/MER/1-4_0.5/icnet/__to_/_/' \
#           'ResUNet_resnet50_icnet_d_epoch_120_53.07.pth'

#RanPaste
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/MSL/1-8_0.5/RanPaste/__to_/_/' \
#           'ResUNet_resnet50_RanPaste_epoch_33_56.60.pth'

#LSST
# WEIGHTS = '/data1/users/lvliang/project_123/LSST-master/LSST-master/output/MSL/1-4_0.2/models/' \
#          'ResUNet_resnet50_58.79.pth'
#WSCL
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/MSL/1-4_0.5/WSCL/__to_/SDC_/ResUNet_resnet50_61.02.pth'
# SGMCR
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/iSAID/300_0.5/train_SGM/_1.0_to_0.5/_/SGM/' \
#           'ResUNet_resnet50_train_SGM_0.20__1.00_to_0.50_weight_0.50_epoch_45_80.18.pth'
# MEMORY_WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/iSAID/300_0.5/train_SGM/_1.0_to_0.5/_/SGM/' \
#           'ResUNet_resnet50_train_SGM_0.20__1.00_to_0.50_weight_0.50_epoch_45_80.18.pth'




#GID-15 1-8
#sup_only
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/GID-15/1-8_0.5/sup_only/__to_/_/ResUNet_resnet50_73.73.pth'
#FixMatch
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/MSL/1-4_0.5/fixmatch/single_0.95_to_0.95/_/' \
#           'ResUNet_resnet50_fixmatch_single_0.95_weight_0.50_epoch_84_57.65.pth'
#ICNet
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/MER/1-4_0.5/icnet/__to_/_/' \
#           'ResUNet_resnet50_icnet_d_epoch_120_53.07.pth'

#RanPaste
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/MSL/1-8_0.5/RanPaste/__to_/_/' \
#           'ResUNet_resnet50_RanPaste_epoch_33_56.60.pth'

#LSST
# WEIGHTS = '/data1/users/lvliang/project_123/LSST-master/LSST-master/output/GID-15/1-8_0.2/models/' \
#          'ResUNet_resnet50_74.78.pth'
#WSCL
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/GID-15/1-8_0.5/WSCL/__to_/SDC_/ResUNet_resnet50_76.01.pth'
# SGMCR
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/iSAID/300_0.5/train_SGM/_1.0_to_0.5/_/SGM/' \
#           'ResUNet_resnet50_train_SGM_0.20__1.00_to_0.50_weight_0.50_epoch_45_80.18.pth'
# MEMORY_WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/iSAID/300_0.5/train_SGM/_1.0_to_0.5/_/SGM/' \
#           'ResUNet_resnet50_train_SGM_0.20__1.00_to_0.50_weight_0.50_epoch_45_80.18.pth'



#GID-15 1-4
#sup_only
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/GID-15/1-4_0.5/sup_only/__to_/_/ResUNet_resnet50_76.27.pth'
#FixMatch
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/GID-15/1-8_0.5/fixmatch/single_0.95_to_0.95/_/' \
#           'ResUNet_resnet50_fixmatch_single_0.95_weight_0.50_epoch_103_70.01.pth'
#ICNet
WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/GID-15/1-4_0.5/fixmatch/single_0.95_to_0.95/_/' \
          'ResUNet_resnet50_fixmatch_single_0.95_weight_0.50_epoch_92_74.80.pth'

#RanPaste
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/MSL/1-8_0.5/RanPaste/__to_/_/' \
#           'ResUNet_resnet50_RanPaste_epoch_33_56.60.pth'

#LSST
# WEIGHTS = '/data1/users/lvliang/project_123/LSST-master/LSST-master/output/GID-15/1-4_0.2/models/' \
#          'ResUNet_resnet50_77.37.pth'
#WSCL
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/GID-15/1-4_0.5/WSCL/__to_/SDC_/ResUNet_resnet50_78.71.pth'
# SGMCR
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/iSAID/300_0.5/train_SGM/_1.0_to_0.5/_/SGM/' \
#           'ResUNet_resnet50_train_SGM_0.20__1.00_to_0.50_weight_0.50_epoch_45_80.18.pth'
# MEMORY_WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/iSAID/300_0.5/train_SGM/_1.0_to_0.5/_/SGM/' \
#           'ResUNet_resnet50_train_SGM_0.20__1.00_to_0.50_weight_0.50_epoch_45_80.18.pth'



#Vaihingen 1-8
#sup_only
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/Vaihingen/1-8_20_0.5/SDC/ResUNet_resnet50_my_strategy_sup_only_72.01.pth'
#FixMatch
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/Vaihingen/1-8_0.5/fixmatch/single_0.95_to_0.95/_/' \
#           'ResUNet_resnet50_fixmatch_single_0.95_weight_0.50_epoch_409_73.39.pth'
#ICNet
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/Vaihingen/1-8_0.5/icnet/__to_/_/' \
#           'ResUNet_resnet50_icnet_epoch_109_73.24.pth'

#RanPaste
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/Vaihingen/1-8_0.5/RanPaste/__to_/_/' \
#           'ResUNet_resnet50_RanPaste_epoch_161_74.17.pth'

#LSST
# WEIGHTS = '/data1/users/lvliang/project_123/LSST-master/LSST-master/output/Vaihingen/1-8_0.2/models/' \
#          'ResUNet_resnet50_73.47.pth'
#WSCL
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/Vaihingen/1-8_20_0.5/SDC/ ResUNet_resnet50_my_strategy_WSCL_74.39.pth'
# SGMCR
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/iSAID/300_0.5/train_SGM/_1.0_to_0.5/_/SGM/' \
#           'ResUNet_resnet50_train_SGM_0.20__1.00_to_0.50_weight_0.50_epoch_45_80.18.pth'
# MEMORY_WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/iSAID/300_0.5/train_SGM/_1.0_to_0.5/_/SGM/' \
#           'ResUNet_resnet50_train_SGM_0.20__1.00_to_0.50_weight_0.50_epoch_45_80.18.pth'



#Vaihingen 1-8
#sup_only
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/Vaihingen/1-4_0.5/sup_only/__to_/_/ResUNet_resnet50_73.60.pth'
#FixMatch
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/Vaihingen/1-4_0.5/fixmatch/single_0.0_to_0.0/_/' \
#           'ResUNet_resnet50_fixmatch_single_0.00_weight_0.50_epoch_272_74.42.pth'
#ICNet
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/Vaihingen/1-4_0.5/icnet/__to_/_/' \
#           'ResUNet_resnet50_icnet_d_epoch_311_74.99.pth'

#RanPaste
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/Vaihingen/1-8_0.5/RanPaste/__to_/_/' \
#           'ResUNet_resnet50_RanPaste_epoch_161_74.17.pth'

#LSST
# WEIGHTS = '/data1/users/lvliang/project_123/LSST-master/LSST-master/output/Vaihingen/1-4_0.2/models/' \
#          'ResUNet_resnet50_74.38.pth'
#WSCL
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/Vaihingen/1-4_0.5/WSCL/__to_/SDC_/ResUNet_resnet50_74.80.pth'
# SGMCR
# WEIGHTS = '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/MER/1-8_0.5/dualmatch/adaptive_decay_1.0_to_0.25/_/' \
#           'ResUNet_resnet50_dualmatch_adaptive_decay_1.00_to_0.25_weight_0.50_epoch_59_57.81.pth'
# MEMORY_WEIGHTS =  '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/exp/MER/1-8_0.5/train_SGM/_1.0_to_0.5/_/SGM/' \
#           'ResUNet_resnet50_train_SGM_0.05__1.00_to_0.50_weight_0.50_epoch_47_58.06.pth'

DFC22_DATASET_PATH = '/data1/users/lvliang/project_123/dataset/DFC22/'
iSAID_DATASET_PATH = '/data1/users/lvliang/project_123/dataset/iSAID/'
MER_DATASET_PATH = '/data1/users/lvliang/project_123/dataset/MER/'
MSL_DATASET_PATH = '/data1/users/lvliang/project_123/dataset/MSL/'
Vaihingen_DATASET_PATH = '/data1/users/lvliang/project_123/dataset/Vaihingen/WCSL_crop/Vaihingen/'
# Vaihingen_DATASET_PATH ='/data1/users/lvliang/project_123/WSCL-main/WSCL-main/dataset/splits/Vaihingen/1-8/save/SEResUNet/20230828_170334_seed1234/'

GID15_DATASET_PATH = '/data1/users/lvliang/project_123/dataset/GID-15/'





# GID15_DATASET_PATH = 'Your local path'

def parse_args():
    parser = argparse.ArgumentParser(description='WSCL Framework')

    # basic settings
    parser.add_argument('--data-root', type=str, default=GID15_DATASET_PATH)
    parser.add_argument('--dataset', type=str, choices=['DFC22', 'iSAID', 'MER', 'MSL', 'Vaihingen', 'GID-15'],
                        default=DATASET)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--backbone', type=str, choices=['resnet50', 'resnet101'], default='resnet101')
    parser.add_argument('--model', type=str, choices=['deeplabv3plus', 'pspnet', 'deeplabv2', 'ResUNet'],
                        default='deeplabv2')
    parser.add_argument('--save-path', type=str, default='test_results/' + WEIGHTS.split('/')[-1].replace('.pth', ''))

    args = parser.parse_args()
    return args


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main(args):
    create_path(args.save_path)

    # model = DeepLabV2(args.backbone, NUM_CLASSES[args.dataset])
    model =se_resnext50_32x4d(num_classes=NUM_CLASSES[args.dataset], pretrained=None)
    #
    # checkpoint = torch.load(WEIGHTS)
    # memory = torch.load(MEMORY_WEIGHTS)['class_momory']

    # model.load_state_dict(checkpoint['state_dict'], strict=True)

    # model = DataParallel(model).cuda()
    model.load_state_dict(torch.load(WEIGHTS))
    model = DataParallel(model).cuda()



    # model = model.cuda()

    valset = SemiDataset(args.dataset, args.data_root, 'val', None)
    valloader = DataLoader(valset, batch_size=8, shuffle=False, pin_memory=False, num_workers=4, drop_last=False)
    eval(model, valloader, args, None)
    # eval(model, valloader, args, None)

def eval(model, valloader, args, memory=None):

    model.eval()
    tbar = tqdm(valloader)

    data_list = []
    mask_list = []
    pred_list = []
    id_list = []

    with torch.no_grad():
        for batch_idx, (img_batch, mask_batch, id_batch) in enumerate(tbar):
        # for img, mask, id in tbar:
            img_batchimg = img_batch.cuda()
            pre_batch, _ = model(img_batch, memory)
            # _, pred = model(img, memory)
            pre_batch = torch.argmax(pre_batch, dim=1).cpu().numpy()
            # prototype_pred = torch.argmax(prototype_pred, dim=1).cpu().numpy()
            # pred = torch.argmax(pred[0], dim=1).cpu().numpy()
            # pred_visual = out_to_rgb(pred)
            # gt_visual = out_to_rgb(mask)
            data_list.append([mask_batch.numpy().flatten(), pre_batch.flatten()])
            # for i in range(len(img_batch)):
            #     img = img_batch[i].unsqueeze(0).cuda()
            #     pred = pre_batch[i]
            #     mask = mask_batch[i]
            #     input_string = 'images/44-2013-0384-6705-LA93-0M50-E080.tif labels/44-2013-0384-6705-LA93-0M50-E080.tif'
            #     parts = id_batch[i].split(' ')  # 使用空格作为分隔符分割字符串
            #     if len(parts) > 1:
            #         labels_part = parts[1]  # 获取分割后的第二部分，即 'labels/44-2013-0384-6705-LA93-0M50-E080.tif'
            #         # 如果需要去掉 'labels/' 部分，可以使用字符串切片
            #         labels_content = labels_part[len('labels/'):]  # 获取 '44-2013-0384-6705-LA93-0M50-E080.tif'
            #         print(labels_content+'\n')
            #     else:
            #         print("No labels found in the input string.")
            #
            #     result = Image.fromarray(out_to_rgb(mask))
            #     result.save('/data1/users/lvliang/project_123/WSCL-main/WSCL-main/visual_result/' + DATASET + '/' + 'GT' +'/' + labels_content )

        filename = os.path.join(args.save_path, 'result.txt')
        get_iou(data_list, NUM_CLASSES[args.dataset], filename, DATASET)


def out_to_rgb(out_index):
    colormap = np.array(DFC22_color_map)
    # colormap = np.array([[0, 0, 0], [0, 125, 0], [0, 0, 125]])
    rgb_img = colormap[np.array(out_index)].astype(np.uint8)
    return rgb_img



def get_iou(data_list, class_num, save_path=None, dataset_name=None):
    from multiprocessing import Pool
    from utils.metric import ConfusionMatrix

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()

    if dataset_name == 'MSL' or dataset_name == 'MER':
        classes, _ = MARS()
    elif dataset_name == 'iSAID':
        classes, _ = iSAID()
    elif dataset_name == 'GID-15':
        classes, _ = GID15()
    elif dataset_name == 'Vaihingen':
        classes, _ = Vaihingen()
    elif dataset_name == 'DFC22':
        classes, _ = DFC22()

    for i, iou in enumerate(j_list):
        print('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], j_list[i] * 100))

    print('meanIOU {:.2f}'.format(aveJ * 100) + '\n')
    if save_path:
        with open(save_path, 'w') as f:
            for i, iou in enumerate(j_list):
                f.write('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], j_list[i] * 100) + '\n')
            f.write('meanIOU {:.2f}'.format(aveJ * 100) + '\n')


if __name__ == '__main__':
    args = parse_args()

    if args.data_root is None:
        args.data_root = {'GID-15': GID15_DATASET_PATH,
                          'iSAID': iSAID_DATASET_PATH,
                          'MER': MER_DATASET_PATH,
                          'MSL': MSL_DATASET_PATH,
                          'Vaihingen': Vaihingen_DATASET_PATH,
                          'DFC22': DFC22_DATASET_PATH}[args.dataset]

    print(args)
    main(args)
