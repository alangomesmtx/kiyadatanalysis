import os
import streamlit as st
import pandas as pd


def viewFile(df):
    st.info("The input file:")
    st.write(df)
    st.info("The list of columns:")
    st.write(df.columns)
    st.info("Brief description about the dataframe created:")
    st.write(df.describe())


def app():
    st.title("Welcome")

    st.markdown("#")
    st.info("About this WebApp", icon="ℹ️")
    st.write("This is a project by Alan Gomes and Asmi Kochrekar."
             "In this project we have added all that we have been taught on Data Science"
             "and Model Creation")
    st.markdown("#")
    st.info("How it works", icon="ℹ️")
    st.write("Each page contains a type of model creation and its analysis. User will have to"
             "upload their dataset in the form of a .csv file and proceed with model"
             "creation and analysis.")
