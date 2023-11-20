# -*- conding:utf-8 -*-
import streamlit as st
import pandas as  pd

def show_interesting():
    st.set_page_config(layout="wide")
    image_path = "./images/interesting.png"
    st.image(image_path, caption='', use_column_width=True)


if __name__ == "__main__":
    show_interesting()