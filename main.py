import data_preprocess

#import data
raw_data = data_preprocess.info()
#preprocess
#stage-I
mask_data_s1 = data_preprocess.data_stage1(raw_data)
#stage-II
mask_data_s2 = data_preprocess.data_stage2(raw_data)