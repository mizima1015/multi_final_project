# -*- coding:utf-8 -*-
import streamlit as st
from pages import dashboard, prediction, team, test2_prediction, test3_prediction, test_prediction


# 메인 페이지를 보여주는 함수입니다.
def main():
    st.title("메인화면")
    image_path = "./images/로고.png"
    st.image(image_path, caption='', use_column_width=True)
    # 메인 페이지의 내용을 여기에 추가합니다.
    '''
    이렇게 글을 쓰면 내용이 추가 되던가? 
    '''

    st.title("HTML CSS 마크다운 적용")
    html_css = """
    <style>
    th,td{
        border-bottom: 1px solid #ddd;
    }
    </style>

    <table>
        <thead>
            <tr>
                <th>이름</th>
                <th>나이</th>
                <th>직업</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>LEE</td>
                <td>28</td>
                <td>데이터 분석가</td>
            </tr>
            <tr>
                <td>somebody</td>
                <td>399</td>
                <td>프로덕트 오너</td>
            </tr>
        </tbody>
    </table>

"""

    st.markdown(html_css, unsafe_allow_html=True)

# Python 스크립트가 실행될 때 main() 함수를 호출합니다.
if __name__ == "__main__":
    main()

