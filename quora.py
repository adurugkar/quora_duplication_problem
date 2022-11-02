import os
from sklearn.metrics import accuracy_score
from typing import final
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import re, os , sys
import warnings
warnings.filterwarnings('ignore')
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
import distance
import tqdm 
from sklearn.model_selection import train_test_split, GridSearchCV
import mlflow
import xgboost  as xgb
from nltk.corpus import stopwords
import spacy
from prefect import task, flow

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your',
 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
 "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 
'each', 'few', 'more', 'most', 'other', 'some', 'such', 'nor', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't",
 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
"needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]


@task
def load_data(path:str)->pd.DataFrame:
    try:
        if os.path.isfile(path):
            df = pd.read_csv(path)
            df = df.head(50000)
            return df
        else:
            print('check tha data path')
    except Exception as e:
        print(e)
@task   
def null_check(data=pd.DataFrame, flags=str)->pd.DataFrame:
    try:
        df = data
        if df.isnull().sum().sum():
            nan = df[df.isnull().any(1)]
            print(nan)
            print('-'*50)
            if flags == 'del':
                clean_data = df.dropna()
            else:
                clean_data = df.fillna(flags)
            return clean_data
        else:
            
            print('data_set has non null values')
        return data
    except Exception as e:
        print(e)

# funation for text preprocessing (Expanding contractions )
def preprocess_text(x):
    x = str(x).lower()
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
                           .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
                           .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
                           .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                           .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                           .replace("€", " euro ").replace("'ll", " will")
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)
    x = re.sub(r"http\S+", "", x)
    x = re.sub('\W', ' ', x) 
    bfs = BeautifulSoup(x) # removing html tage form the text
    x = bfs.get_text()
    x = x.strip()
    return x
@task
def data_cleaning(data):
    data['Cl_question1'] = data['question1'].apply(preprocess_text)
    data['Cl_question2'] = data['question2'].apply(preprocess_text)
    return data


# function removing stopword and stemming or lemmatizer
@task
def removing_stopword(data):
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    data = data.split()
    x = [lemmatizer.lemmatize(word) for word in data if word not in stopwords]
    x = ' '.join(x)
    x = x.strip()
    return x

# function for lemmitization
@task
def lemm_data(data):
    try:

        lemm = pd.DataFrame()
        lemm['id'] = data['id']
        lemm['lemm_data_q1'] = data.question1.apply(removing_stopword)
        lemm['lemm_data_q2'] = data.question2.apply(removing_stopword)
        return lemm.merge(data[['id','qid1','qid2','is_duplicate']],how='left',on='id')
    except Exception as e:
        print(e)
    
def doesMatch (q, match):
    q1, q2 = q['question1'], q['question2']
    q1 = str(q1).split()
    q2 = str(q2).split()
    if len(q1)>0 and len(q2)>0 and q1[match]==q2[match]:
        return 1
    else:
        return 0
    
def get_longest_substr_ratio(a, b):
    strs = list(distance.lcsubstrings(a, b))
    if len(strs) == 0:
        return 0
    else:
        return len(strs[0]) / (min(len(a), len(b)) + 1)
    

def common_stop_words_ratio(q,value):
    q1_tokens =str( q.question1).split()
    q2_tokens = str(q.question2).split()
    
    #Get the stopwords in Questions
    q1_stops = set([word for word in q1_tokens if word in stopwords])
    q2_stops = set([word for word in q2_tokens if word in stopwords])
    
    common_stop_count = len(q1_stops.intersection(q2_stops))
    if value == 'min':
        token_features = common_stop_count / (min(len(q1_stops), len(q2_stops)) + 0.0001)
    elif value == 'max':
        token_features = common_stop_count / (max(len(q1_stops), len(q2_stops)) + 0.0001)
    return token_features
@task
def feature_extract(data):
    try:
    

        print('feature_extraction_start.....')
        data['len_q1'] = data.question1.str.len()
        data['len_q2'] = data.question2.str.len()
        data['q1_word'] = data.question1.apply(lambda x: len(str(x).split(' ')))
        data['q2_word'] = data.question2.apply(lambda x: len(str(x).split(' ')))
        
        data['total_word'] = data['q1_word'] + data['q2_word']
        data['differ_word_num'] = abs(data['q1_word'] - data['q2_word'])
        data['same_first_word'] = data.apply(lambda x: doesMatch(x, 0) ,axis=1)
        data['same_last_word'] = data.apply(lambda x: doesMatch(x, -1) ,axis=1)
        data['total_unique_word'] = data.apply(lambda x: len(set(str(x.question1).split()).union(set(str(x.question2).split()))) ,axis=1)
        data['total_unique_word_withoutstopword_num'] = data.apply(lambda x: len(set(str(x.question1).split()).union(set(str(x.question2).split())) - set(stopwords)) ,axis=1)
        data['total_unique_word_num_ratio'] = data['total_unique_word'] / data['total_word']
        print('......')
        data['common_word'] = data.apply(lambda x: len(set(str(x.question1).split()).intersection(set(str(x.question2).split()))) ,axis=1)
        data['common_word_ratio'] = data['common_word'] / data['total_unique_word'] # word share
        data['word_share'] = data['common_word']/data['total_word']
        data['common_word_ratio_min'] = data['common_word'] / data.apply(lambda x: min(len(set(str(x.question1).split())), len(set(str(x.question2).split()))) ,axis=1) 
        data['common_word_ratio_max'] = data['common_word'] / data.apply(lambda x: max(len(set(str(x.question1).split())), len(set(str(x.question2).split()))) ,axis=1) 
        
        data['common_stop_word_ratio_min'] = common_stop_words_ratio(data,'min')
        data['common_stop_word_ratio_max'] = common_stop_words_ratio(data, 'max')
        
        data['common_word_withoutstopword'] = data.apply(lambda x: len(set(str(x.question1).split()).intersection(set(str(x.question2).split())) - set(stopwords)) ,axis=1)
        data['common_word_withoutstopword_ratio'] = data['common_word_withoutstopword'] / data['total_unique_word_withoutstopword_num']
        
        data['common_word_withoutstopword_ratio_min'] = data['common_word_withoutstopword'] / data.apply(lambda x: min(len(set(str(x.question1).split()) - set(stopwords)), len(set(str(x.question2).split()) - set(stopwords))) ,axis=1) 
        data['common_word_withoutstopword_ratio_max'] = data['common_word_withoutstopword'] / data.apply(lambda x: max(len(set(str(x.question1).split()) - set(stopwords)), len(set(str(x.question2).split()) - set(stopwords))) ,axis=1) 
        
        print('fuzzy features...')
        print('fuzz_ratio.....')
        data["fuzz_ratio"] = data.apply(lambda x: fuzz.ratio(str(x.question1), str(x.question2)), axis=1)
        
        print('fuzz_partial_ratio.....')
        data["fuzz_partial_ratio"] = data.apply(lambda x: fuzz.partial_ratio(str(x.question1), str(x.question2)), axis=1)
        
        print('fuzz_token_set_ratio.....')
        data["fuzz_token_set_ratio"] = data.apply(lambda x: fuzz.token_set_ratio(str(x.question1), str(x.question2)), axis=1)
        
        print('fuzz_token_sort_ratio.....')
        data["fuzz_token_sort_ratio"] = data.apply(lambda x: fuzz.token_sort_ratio(str(x.question1), str(x.question2)), axis=1)
        
        print('longest_substr_ratio.....')
        data["longest_substr_ratio"]  = data.apply(lambda x: get_longest_substr_ratio(str(x.question1), str(x.question2)), axis=1)
        data.fillna(0, inplace=True)
        return data
    except Exception as e:
        print(e)

#TF-ITF with word2vec
@task
#TF-ITF with word2vec

def TF_word2vec_q1(data):
    try:
        df = pd.DataFrame()
        questions = list(data['question1']) + list(data['question2'])

        tfidf = TfidfVectorizer(lowercase=False,)
        tfidf.fit_transform(questions)

        # dict key:word and value:tf-idf score
        word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf.idf_))
        nlp = spacy.load('en_core_web_sm')


        vecs1 =[]
        # tqdm is used to print the progress bar
        print('vectorization start.....')
        for qu1 in tqdm.tqdm(list(data['question1'])):
            doc1 = nlp(qu1)
            #384 is the number of dimensions of vectors
            mean_vec1 = np.zeros([len(doc1), len(doc1[0].vector)]) 
            for word1 in doc1:
                #word2vec
                vec1 = word1.vector
                #fetch df score
                try:
                    idf = word2tfidf[str(word1)]
                except:
                    idf = 0

                # compute final vec
                mean_vec1 += vec1 * idf
            mean_vec1 = mean_vec1.mean(axis= 0)
            vecs1.append(mean_vec1)

        df['q1_feats_m'] = list(vecs1)
        df2_q1 = pd.DataFrame(df.q1_feats_m.values.tolist(), index= data.index)
        return df2_q1
    except Exception as e:
        print(e)

def TF_word2vec_q2(data):
    try:
        df = pd.DataFrame()
        questions = list(data['question1']) + list(data['question2'])

        tfidf = TfidfVectorizer(lowercase=False,)
        tfidf.fit_transform(questions)

        # dict key:word and value:tf-idf score
        word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf.idf_))
        nlp = spacy.load('en_core_web_sm')


        vecs2 =[]
        # tqdm is used to print the progress bar
        print('vectorization start.....')
        for qu1 in tqdm.tqdm(list(data['question2'])):
            doc1 = nlp(qu1)
            #384 is the number of dimensions of vectors
            mean_vec1 = np.zeros([len(doc1), len(doc1[0].vector)]) 
            for word1 in doc1:
                #word2vec
                vec1 = word1.vector
                #fetch df score
                try:
                    idf = word2tfidf[str(word1)]
                except:
                    idf = 0

                # compute final vec
                mean_vec1 += vec1 * idf
            mean_vec1 = mean_vec1.mean(axis= 0)
            vecs2.append(mean_vec1)

        df['q1_feats_m'] = list(vecs2)
        df2_q2 = pd.DataFrame(df.q1_feats_m.values.tolist(), index= data.index)
        return df2_q2
    except Exception as e:
        print(e)

@task
def merge_data(question1,question2,feature_data_pr):
    try:    
        question1['id']=feature_data_pr['id']
        question2['id']=feature_data_pr['id']

        df1  = question1.merge(question2, on='id',how='left')
        final  = feature_data_pr.merge(df1, on='id',how='left')
        return final
    except Exception as e:
        print(e)

@task
def split_data(x,y,test_ratio):
    try:
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=test_ratio)
        return x_train,x_test,y_train,y_test
    except Exception as e:
        print(e)
        
@task
def model_train(x_train,y_train,model:object, param:dict):

    mlflow.set_tracking_uri('sqlite:///mlflow5.db')
    mlflow.set_experiment('Quora_pair_question_problem5')
    mlflow.sklearn.autolog(max_tuning_runs=None)

    with mlflow.start_run():
        param_grid = param
        
        xg = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv = 5,
            scoring='neg_mean_squared_error',
            return_train_score= True,
            n_jobs = 1
        )
        xg.fit(x_train,y_train)
        
        #disabling autologging
        mlflow.sklearn.autolog(disable=True)
        print(xg.best_params_)
        t = xg.best_params_
    return model(learning_rate=t['learning_rate'],max_depth=t['max_depth'],min_child_weight=['min_child_weight'],n_estimators=t['n_estimators'])

@flow
def main(path:str,data_number:int):
    df_num = data_number
    # Define parameters
    Target_column = 'is_duplicate'
    data_path = path

    # load the data
    dataframe = load_data(data_path)
    dataframe = dataframe.head(df_num)

    # data cleaning
    clean_data = data_cleaning(dataframe)

    # feature extraction 
    extract_data = feature_extract(clean_data)
    print(extract_data.shape)

    # word2vec
    question_1 = TF_word2vec_q1(clean_data)
    question_2 = TF_word2vec_q2(clean_data)

    # merg dataset into one file
    final_data = merge_data(question_1,question_2,extract_data)

    # Identify target varible
    input_data = final_data['is_duplicate']
    target_data =final_data.drop(['is_duplicate'],axis=1)


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
main('./train.csv',500)