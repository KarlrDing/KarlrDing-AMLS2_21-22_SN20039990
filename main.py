import sys
sys.path.append('./Stage_I')
sys.path.append('./Stage_II')

import data_preprocess
import model_stage1
import model_stage2

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
ori_s1,bal_s1 = data_preprocess.minor_dataset(mask_data_s1)
ori_s2,bal_s2 = data_preprocess.minor_dataset(mask_data_s2)

#minor dataset label distribution
data_preprocess.dist_minor(ori_s1,1)
data_preprocess.dist_minor(bal_s1,2)
data_preprocess.dist_minor(ori_s2,3)
data_preprocess.dist_minor(bal_s2,4)