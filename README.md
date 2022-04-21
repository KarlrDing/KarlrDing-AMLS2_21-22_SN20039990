# AML2 REPORT
### KarlrDing-AMLS2_21-22_SN20039990
***
Intro:
This project is based on a competition:  **2020  \[Open\] Shopee Code League - Sentiment Analysis** on the Kaggle. Task is aims to built a Shopee Product Review Sentiment Analyzer. I subtask it into two stages: the first stage is to divide the reviews into positive, negative reviews. The second stage is to classify the reviews according to 5 starts.

Stage-I: Sentiment analysis positive / negative reviews.
Stage-II: Sentiment analysis 1,2,3,4,5 stars rating.

---
Structure:
Under the project folder there are three folders: **Stage_I**, **Stage_II** and **dataset** .
1. Dataset store original data (in .csv)
2. Stage_I store stage-I python file, images(in png and svg) and test result. (recall F1-score presicion etc.)
3. Stage_II store stage-II python file, images(in png and svg) and test result. (recall F1-score presicion etc.)
---
Code:
This project can run in two kinds of format: jupyternotebook and python file.
If you want to use jupyter notebook, just use the .ipynb files.(**stage_1.ipynb** and **stage_2.ipynb**) They include all the code and result.
If you want to use python file, using **main.py**. It will call **data_preprocess.py** and each **stage.py** file in the folder.