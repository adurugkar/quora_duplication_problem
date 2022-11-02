import streamlit as st
import pandas as pd

df = pd.read_csv('Data/train.csv')

st.set_page_config(page_title="About", page_icon="")

st.title("Quora Question Pair Similarity")
st.markdown("There are almost 100 million monthly users on Quora, therefore it's not surprising that many questions have similar wording. Multiple inquiries with the same objective can make readers feel as though they must respond to various versions of the same question, while also making seekers spend more time looking for the best solution to their problem. Canonical questions are highly valued on Quora because they provide active writers and seekers a better experience and more long-term value. Therefore, the primary goal of the research is to determine if a pair of questions are comparable or not. This can be helpful for providing prompt responses to questions that have already been addressed. Source: Kaggle")
st.header("Problem Statement")
st.markdown("Identify which questions asked on Quora are duplicates of questions that have already been asked.")
st.sidebar.header("About Project")
st.header("Data Overview")
st.markdown("Train.csv contains 5 columns : qid1, qid2, question1, question2, is_duplicate. Total we have 404290 entries. Splitted data into train and test with 70% and 30%.")
st.dataframe(df.head())