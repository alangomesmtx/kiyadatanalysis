import streamlit as st
from streamlit_option_menu import option_menu

import classification
import clustering
import welcome
import lrm

st.set_page_config(
    page_title="Kiya.ai Data Analytics",
    page_icon="📊"
)


with st.sidebar:
    app = option_menu(
        menu_title="Navigation",
        options=["Welcome", "Regression Model", "Classification", "Clustering"]
    )
if app == 'Welcome':
    welcome.app()
if app == 'Regression Model':
    lrm.app()
if app == 'Classification':
    classification.app()
if app == 'Clustering':
    clustering.app()
