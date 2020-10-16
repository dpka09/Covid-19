import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Covid Data Prediction App 
This app predicts the probability of Covid infection!!!

""")

st.sidebar.header("User Input Parameters")

def user_input():
    aged_65_older = st.sidebar.slider('aged_65_older',1,100, 10)  
    cardiovasc_death_rate = st.sidebar.slider('cardiovasc_death_rate',500,700,550)     

user_input()

df = pd.read_csv("covid19.csv")





