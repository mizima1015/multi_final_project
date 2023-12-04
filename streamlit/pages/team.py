# -*- coding:utf-8 -*-

import streamlit as st

def show_team():
    st.set_page_config(layout="wide")
    image_path = "./images/team.png"
    st.image(image_path, caption='', use_column_width=True)
    st.sidebar.markdown("[Tableu(서울시)](https://public.tableau.com/app/profile/.13586515/viz/df3_1_re/12?publish=yes)")
    st.sidebar.markdown("[Tableu(자치구)](https://public.tableau.com/app/profile/.13586515/viz/df3_2_re/24?publish=yes)")
    
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
    background: linear-gradient(to right, rgb(225, 236, 247), rgb(240, 245, 250));
    color: #888;
    text-align: center;
    padding: 1px;
    font-size: 12px;
    }
    </style>

    

    

    <div class="footer">
        김하은 | <a href="https://haeunkim48.tistory.com/">Blog</a>,  <a href="https://github.com/haeunkim48">Gihub</a>, 
        심영은 | <a href="https://github.com/Ezraelyes">Gihub</a>,
        이나윤 | <a href="https://data-analytics-nayoonee.tistory.com/">Blog</a>,  <a href="https://github.com/Nayoon-Lee">Gihub</a>,
        이상혁 | <a href="https://sang-hyeok0214.tistory.com/">Blog</a>,  <a href="https://github.com/Sang-Hyeok-Lee">Gihub</a>,
        이승준 | <a href="https://mizima-data.tistory.com/">Blog</a>,  <a href="https://github.com/mizima1015">Gihub</a>,
        최연우 | <a href="https://beimmersedin.tistory.com/">Blog</a>,  <a href="https://github.com/beimmersedin">Gihub</a>
        </p>
        <p> mizimaaz@gmail.com  /  2023 DeliverEase </p>
    </div>

    """
    st.markdown(html_css, unsafe_allow_html=True)


if __name__ == "__main__":
    show_team()
