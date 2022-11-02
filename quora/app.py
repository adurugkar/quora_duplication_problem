import streamlit as st
import helper
import pickle
from PIL import Image
from fuzzywuzzy import fuzz

model = pickle.load(open('Xgb.pkl','rb'))
img = Image.open("Image/Ques.png")
st.image(img,width=200)
st.title("Quora Questions Pairs App")

st.markdown("Fill the First question and Second question text inputs and click"
                   " the button Check if duplicates.")

q1 = st.text_input("First question:", max_chars=512)
q2 = st.text_input("Second question:", max_chars=512)

if st.button('Check if duplicates'):
    query = helper.query_point_creator(q1,q2)
    result = model.predict(query)[0]

    #fuz_ratio = fuzz.ratio(q1,q2)
    #if fuz_ratio==1.00:
     #   st.header ('Question is dupicate')
    #elif fuz_ratio < 50:
     #   st.header('question is not duplicate')
    #elif fuz_ratio > 75:
      #  st.header('Question almost same')
    if result:
        st.header('Duplicate')
    else:
        st.header('Not Duplicate')

st.sidebar.success("select page above")