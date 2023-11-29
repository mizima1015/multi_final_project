# -*- coding:utf-8 -*-

import streamlit as st

def show_team():
    st.set_page_config(layout="wide")
    image_path = "./images/team.png"
    st.image(image_path, caption='', use_column_width=True)
    st.sidebar.markdown("[Tableu(서울시)](https://public.tableau.com/app/profile/.13586515/viz/df3_1_re/12?publish=yes)")
    st.sidebar.markdown("[Tableu(자치구)](https://public.tableau.com/app/profile/.13586515/viz/df3_2_re/24?publish=yes)")
    


if __name__ == "__main__":
    show_team()
