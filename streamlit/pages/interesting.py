# -*- conding:utf-8 -*-
import streamlit as st
import pandas as  pd


st.set_page_config(layout="wide")

def show_interesting():
    image_path = "./images/interesting.png"
    st.image(image_path, caption='', use_column_width=True)
    st.sidebar.markdown("[Tableu(서울시)](https://public.tableau.com/app/profile/.13586515/viz/df3_1_re/12?publish=yes)")
    st.sidebar.markdown("[Tableu(자치구)](https://public.tableau.com/app/profile/.13586515/viz/df3_2_re/24?publish=yes)")
    
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
    # 선택된 카테고리에 해당하는 모든 이미지 표시
    if category:
        image_paths = categories[category]
        # 한 줄에 표시할 이미지 수를 정합니다.
        images_per_row = 3
        # 이미지 수에 따라 필요한 컬럼 수를 계산합니다.
        num_rows = (len(image_paths) + images_per_row - 1) // images_per_row
    
        # 각 줄에 대한 반복
        for i in range(num_rows):
            # 현재 줄에 대한 컬럼 생성
            cols = st.columns(images_per_row)
            # 각 컬럼에 이미지 할당
            for j in range(images_per_row):
                # 이미지 인덱스 계산
                image_index = i * images_per_row + j
                if image_index < len(image_paths):  # 이미지 리스트를 넘어가지 않도록
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

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 

if __name__ == "__main__":
    add_bg_from_url()
    show_interesting()