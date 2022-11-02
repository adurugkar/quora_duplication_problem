# importing library
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings 
warnings.filterwarnings('ignore')
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
from sklearn.preprocessing import normalize
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pickle
import mlflow
import os
from quora import *
from prefect import flow, task

@flow
def main(path:str):
    # Define parameters
    Target_column = 'is_duplicate'
    data_path = path

    # load the data
    dataframe = load_data(data_path)
    dataframe = dataframe.head(5000)

    # data cleaning
    clean_data = data_cleaning(dataframe)

    # feature extraction 
    extract_data = feature_extract(clean_data)

    # word2vec
    question_1 = TF_word2vec_q1(clean_data)
    question_2 = TF_word2vec_q2(clean_data)
    

    # merg dataset into one file
    final_data = marge_data(question_1,question_2, extract_data)

    # Identify target varible
    input_data = dataframe[Target_column]
    target_data =final_data[:len(input_data)]


    # split the data into train and test
    x_train,x_test,y_train,y_test = split_data(target_data, input_data,test_ratio=0.25) 

    # model training
    param_ = {  
              'n_estimators': [40, 60, 80], 
              'max_depth': range(1, 4), 
              'learning_rate': [1e-3], 
              'min_child_weight': range(1, 4), 
             }
    model = model_train(x_train,y_train,xgb.XGBClassifier(),param=param_)
    y_pre = model.predict(x_test)
    print(accuracy_score(y_pre,y_test))

# final run
main('./train.csv')