# -*- conding:utf-8 -*-
import streamlit as st
import pandas as  pd

def show_interesting():
    st.set_page_config(layout="wide")
    image_path = "./images/interesting.png"
    st.image(image_path, caption='', use_column_width=True)
    html_css = """
    <style>

    th,td{
        border-bottom: 1px solid #ddd;
    }
    img{
        margin-bottom:20px;
    }
    .highlight{
        font-size: 34px;
        font-weight: bold;
        color: rgb(255, 0, 0);
        font-family: 'Arial', sans-serif;
    }
    .multicampus{
        font-size: 24px;
        color: rgb(255,229,180);
    }
    # special-text{
        font-size: 20px;
        color: #0000ff;
    }
    .footer {
    position: relative;
    top: 30px;
    bottom: 0;
    left: 0px;
    right:30px;
    width: 100%;
    background: linear-gradient(to right, rgb(204, 193, 80), rgb(247, 245, 225));
    color: #888;
    text-align: center;
    padding: 1px;
    font-size: 12px;
    }
    </style>

    

    

    <div class="footer">
        <p> 추가로 더 알아봤으면 하는 변수가 있다면 아래 메일로 문의주시기 바랍니다. </p>   
        <p> mizimaaz@gmail.com  /  2023 DeliverEase </p>
    </div>

    """
    st.markdown(html_css, unsafe_allow_html=True)

if __name__ == "__main__":
    show_interesting()