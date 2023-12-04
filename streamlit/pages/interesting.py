# -*- conding:utf-8 -*-
import streamlit as st
import pandas as  pd




def show_interesting():
    image_path = "./images/interesting.png"
    st.image(image_path, caption='', use_column_width=True)
    st.sidebar.markdown("[Tableu(서울시)](https://public.tableau.com/app/profile/.13586515/viz/df3_1_re/sheet14_1?publish=yes)")
    st.sidebar.markdown("[Tableu(자치구)](https://public.tableau.com/app/profile/.13586515/viz/df3_2_re/sheet20?publish=yes)")
    
    with st.expander("페이지 설명"):
        st.write("""
                 이 페이지에는 수치화가 어려운, 한 번쯤 생각해봤을 법한 변수들을 생각나는데로 확인해보고 그래프를 그려봤습니다. \n
                 카테고리를 선택하면 해당 카테고리에 속하는 여러 변수들을 확인할 수 있습니다.
            """)
    
    categories = {
        '게임 출시': ['./images/zelda.png'],
        '명절': ['./images/2021_설.png','./images/2022_설.png', './images/2023_설.png', './images/2021_추석.png','./images/2022_추석.png'],
        '음반 발매': ['./images/BTS.png', './images/stray.png'],
        '정책': ['./images/거리두기_디지털가전.png', './images/거리두기_생활건강.png'],
        '행사 및 이벤트 기간': ['./images/2021_할로윈_의류.png', './images/2022_할로윈_의류.png','./images/2021_할로윈_잡화.png','./images/2022_할로윈_잡화.png'],
        '휴대폰 새 기종 출시': ['./images/iPhone14.png', './images/s23.png'],        
    
        }   

    # selectbox를 사용하여 카테고리 선택
    category = st.selectbox('Choose a category:', list(categories.keys()))

    # 선택된 카테고리에 해당하는 모든 이미지 표시
    if category:
        image_paths = categories[category]
        images_per_row = 3
        num_rows = (len(image_paths) + images_per_row - 1) // images_per_row
    
        # 각 줄에 대한 반복
        for i in range(num_rows):
            cols = st.columns(images_per_row)
            for j in range(images_per_row):
                image_index = i * images_per_row + j
                if image_index < len(image_paths):
                    with cols[j]:
                        st.image(image_paths[image_index], width=430)

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
        <p> 추가로 더 알아봤으면 하는 변수가 있다면 아래 메일로 문의주시기 바랍니다. </p>   
        <p> mizimaaz@gmail.com  /  2023 DeliverEase </p>
    </div>

    """
    st.markdown(html_css, unsafe_allow_html=True)
 
if __name__ == "__main__":

    show_interesting()