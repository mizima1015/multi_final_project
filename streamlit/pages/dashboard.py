# -*- coding:utf-8 -*-

import streamlit as st
import streamlit.components.v1 as components

def show_dashboard():
    st.title("Tableau 대시보드")
    tableau_url = "https://naver.com"
    st.markdown(f"[여기를 클릭하여 대시보드를 보세요]({tableau_url})", unsafe_allow_html=True)

    components.iframe(tableau_url, width=900, height=800)

if __name__ == "__main__":
    show_dashboard()
