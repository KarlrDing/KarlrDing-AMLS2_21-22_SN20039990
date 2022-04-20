#prepare dataset
#import libraries
import pandas as pd 
from datasets import Dataset

#file path
path_img = './img'
path_dat = './dataset'
path_fil = '/clean_extended_train.csv'
#parameters
LEN_CONS = 60

#print dataset basic information
def info():
	raw_data = pd.read_csv(path_dat + path_fil)
	print('Numbers of samples: '+str(raw_data.shape[0]))
	print('Columns names: '+str(raw_data.columns.values))
	print('Rating categories: '+str(raw_data['rating'].unique()))
	print('\n')
	return raw_data

#prepare dataset for stage-I
def data_stage1(raw_data):
	task1_data = raw_data.copy()
	#follow the BERT input label format
	#0 stands for negative, 1 stands for positive
	task1_data.loc[task1_data['rating']<=3,'rating'] = 0
	task1_data.loc[task1_data['rating']>=4,'rating'] = 1
	#clear empty value
	complete_data = task1_data.dropna().reindex()
	#select data whose length is below LEN_CONS
	mask = (complete_data['review'].str.len() <= LEN_CONS)
	mask_data = complete_data.loc[mask]
	#show rating distribution
	print('Stage-I rating distribution is:\n')
	print(mask_data['rating'].value_counts())
	return mask_data
#prepare dataset for stage-II
def data_stage2(raw_data):
	task2_data = raw_data.copy()
	#follow the BERT input label format
	task2_data['rating'] = task2_data['rating'] - 1
	#clear empty value
	complete_data = task2_data.dropna().reindex()
	#select data whose length is below LEN_CONS
	mask = (complete_data['review'].str.len() <= LEN_CONS)
	mask_data = complete_data.loc[mask]
	#show rating distribution
	print('Stage-II rating distribution is:\n')
	print(mask_data['rating'].value_counts())
	return mask_data