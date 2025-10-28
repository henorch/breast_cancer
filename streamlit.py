import  streamlit as st
import numpy as np
import joblib

#loading my model
model = joblib.load("breast_model.pkl")

#Putting in the title
st.title("ML Breast Cancer Early Detection and Support Decsion")