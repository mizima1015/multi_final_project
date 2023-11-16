# -*- coding:utf-8 -*-

import streamlit as st
import streamlit.components.v1 as components

def show_dashboard():
    st.set_page_config(layout="wide")
    st.title("Tableau 대시보드")
    tableau_url = "http://deliverease-multifpp.com"
    st.markdown(f"[여기를 클릭하여 대시보드를 보세요]({tableau_url})", unsafe_allow_html=True)

    # components.iframe(tableau_url, width=1000, height=1500)


if __name__ == "__main__":
    show_dashboard()
