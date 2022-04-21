import sys
sys.path.append('./Stage_I')
sys.path.append('./Stage_II')

import data_preprocess
import model_stage1
import model_stage2
# ======================================================================================================================
# Data preprocessing
# ======================================================================================================================
#import data
raw_data = data_preprocess.info()
#preprocess
#stage-I
mask_data_s1 = data_preprocess.data_stage1(raw_data)
#stage-II
mask_data_s2 = data_preprocess.data_stage2(raw_data)

#plot ditribution plot
data_preprocess.dist_plot(mask_data_s1,1)
data_preprocess.dist_plot(mask_data_s2,2)

#form a minor dataset
ori_s1,bal_s1 = data_preprocess.minor_dataset1(mask_data_s1)
ori_s2,bal_s2 = data_preprocess.minor_dataset2(mask_data_s2)

#minor dataset label distribution
data_preprocess.dist_minor(ori_s1,1)
data_preprocess.dist_minor(bal_s1,2)
data_preprocess.dist_minor(ori_s2,3)
data_preprocess.dist_minor(bal_s2,4)

# ======================================================================================================================
# Stage-I
# ======================================================================================================================
#Stage-I model and trainning
model_stage1.model_dataset_choose(0,ori_s1,bal_s1)
model_stage1.model_dataset_choose(1,ori_s1,bal_s1)
model_stage1.model_dataset_choose(3,ori_s1,bal_s1)
model_stage1.model_dataset_choose(4,ori_s1,bal_s1)
# ======================================================================================================================
# Stage-II
# ======================================================================================================================
#Stage-II model and trainning
model_stage2.model_dataset_choose(0,ori_s2,bal_s2)
model_stage2.model_dataset_choose(1,ori_s2,bal_s2)
model_stage2.model_dataset_choose(3,ori_s2,bal_s2)
model_stage2.model_dataset_choose(4,ori_s2,bal_s2)