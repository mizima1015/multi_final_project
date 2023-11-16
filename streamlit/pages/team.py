# -*- coding:utf-8 -*-

import streamlit as st

def show_team():
    st.set_page_config(layout="wide")
    # image_path = "./images/team.png"
    # st.image(image_path, caption='', use_column_width=True)

    st.markdown("""
        <style>
        .reportview-container {
        background: url('https://raw.githubusercontent.com/mizima1015/multi_final_project/main/streamlit/images/main2.png');
        background-size: cover;
        }
        </style>
     """, unsafe_allow_html=True)

if __name__ == "__main__":
    show_team()
