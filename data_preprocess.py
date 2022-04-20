#prepare dataset
#import libraries
import pandas as pd 
from datasets import Dataset

#file path
path_sg1 = './Stage_I'
path_sg2 = './Stage_II'
path_img = '/img'
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
#distribution plot of samples
def dist_plot(mask_data,path_code=1):
	if path_code == 1:
		path = path_sg1
		pass
	elif path_code == 2:
		path = path_sg2
		pass

	import matplotlib.pyplot as plt
	import seaborn as sns
	#configure figure size and font size
	fig,ax = plt.subplots(figsize = (3.8,2.6))
	plt.rcParams['font.size'] = 9.7
	#set labels
	ax.set_xlabel("Sentiment tendency")
	ax.set_ylabel("Numbers of reviews")
	#set ticks
	ax.set_xticks([0,1])
	ax.set_xticks([0,1])
	ax.set_xticklabels(['Negative','Positive'])
	#seaborn histplot
	sns.histplot(mask_data['rating'], kde=True, ax = ax, linewidth = 1.6)
	#save plot in two files : png and svg (vector)
	plt.savefig(path+path_img+'/'+'distribution'+'.png')
	plt.savefig(path+path_img+'/'+'distribution'+'.svg')
	pass
#form a smaller dataset
def minor_dataset1(mask_data):
	#original ratio
	data_0 = mask_data.loc[mask_data['rating'] == 0].head(2000)
	data_1 = mask_data.loc[mask_data['rating'] == 1].head(18000)
	minor_data_ori = pd.concat([data_0, data_1])
	#balanced ratio
	data_0 = mask_data.loc[mask_data['rating'] == 0].head(10000)
	data_1 = mask_data.loc[mask_data['rating'] == 1].head(10000)
	minor_data_bal = pd.concat([data_0, data_1])
	return minor_data_ori,minor_data_bal
def minor_dataset2(mask_data):
	#original ratio
	data_0 = mask_data.loc[mask_data['rating'] == 0].head(16400)
	data_1 = mask_data.loc[mask_data['rating'] == 1].head(1600)
	data_2 = mask_data.loc[mask_data['rating'] == 2].head(1000)
	data_3 = mask_data.loc[mask_data['rating'] == 3].head(600)
	data_4 = mask_data.loc[mask_data['rating'] == 4].head(400)
	minor_data_ori = pd.concat([data_0, data_1, data_2, data_3, data_4])
	#balanced ratio
	data_0 = mask_data.loc[mask_data['rating'] == 0].head(4000)
	data_1 = mask_data.loc[mask_data['rating'] == 1].head(4000)
	data_2 = mask_data.loc[mask_data['rating'] == 2].head(4000)
	data_3 = mask_data.loc[mask_data['rating'] == 3].head(4000)
	data_4 = mask_data.loc[mask_data['rating'] == 4].head(4000)
	minor_data_bal = pd.concat([data_0, data_1, data_2, data_3, data_4])
	return minor_data_ori,minor_data_bal
#show distribution
def dist_minor(minor_data,path_code = 1):
	if path_code == 1:
		path = path_sg1
		name = 'ori'
		pass
	elif path_code == 2:
		path = path_sg1
		name = 'bal'
		pass
	elif path_code == 3:
		path = path_sg2
		name = 'ori'
		pass
	elif path_code == 4:
		path = path_sg2
		name = 'bal'
		pass

	distribution = minor_data['review'].str.len().astype(int)
	print(distribution.describe().apply(lambda x: format(x, 'f')))
	#boxplot
	import matplotlib.pyplot as plt
	import seaborn as sns
	#configure figure size and font size
	fig,ax = plt.subplots(figsize = (3.8,2.6))
	plt.rcParams['font.size'] = 9.7
	#seaborn boxplot
	ax = sns.boxplot(distribution)
	plt.savefig(path+path_img+'/'+'dist_box_'+name+'.png')
	plt.savefig(path+path_img+'/'+'dist_box_'+name+'.svg')
	pass