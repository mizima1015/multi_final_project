# -*- coding:utf-8 -*-

import streamlit as st

def show_team():
    st.set_page_config(layout="wide")
    image_path = "./images/team.png"
    st.image(image_path, caption='', use_column_width=True)


if __name__ == "__main__":
    show_team()
