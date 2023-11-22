# -*- coding:utf-8 -*-

import streamlit as st
import streamlit.components.v1 as components

def show_dashboard():
    st.set_page_config(layout="wide")
    st.title("Tableau 대시보드")
    
    tableau_url = "https://public.tableau.com/app/profile/.13586515/viz/_1_17004659183630/1?publish=yes"
    tableau_url2 = "https://public.tableau.com/app/profile/.13586515/viz/_2_17006268549610/22?publish=yes"
    st.markdown(f"[서울시 전체 대시보드로 가기]({tableau_url})", unsafe_allow_html=True)
    st.markdown(f"[구별 대시보드로 가기]({tableau_url2})", unsafe_allow_html=True)
    components.iframe(tableau_url, width=1000, height=1500)


if __name__ == "__main__":
    show_dashboard()
