# -*- coding:utf-8 -*-
import streamlit as st
from pages import dashboard, prediction, team, test3_prediction
 
# nohup streamlit run main.py > streamlit.log 2>&1 &
# ----- 서버 streamlit 실행-----

# 링크 : http://34.64.204.232:8501/
# 링크2: http://deliverease-multifpp.com
# ----- 서비스 접속 링크 -----

# sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 8501
# sudo apt-get install iptables-persistent
# ------ 포트포워딩 -----

# 메인 페이지를 보여주는 함수입니다.
def main():
    st.set_page_config(layout="wide")
    # st.title("메인화면")
    image_path = "./images/main3.png"
    st.image(image_path, caption='', use_column_width=True)
    
    # 메인 페이지의 내용을 여기에 추가합니다.

    # st.title("HTML CSS 마크다운 적용")
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

    안녕하세요? 저희는 <span class="highlight">DeliverEase</span> 팀입니다. \n
    저희는 국비지원 취업교육<span class="multicampus"> Multicampus 데이터분석가 & 엔지니어</span> 28회차를 수강한 교육생입니다.\n
    저희 프로젝트는 코로나 이후 비대면과 배달 등이 트랜드가 되면서 함께 늘어난 택배 물류량으로 인해 
    많은 택배 물류 운송업체에서 업무과중 문제가 심각하다고 생각되어 기획하게 되었습니다.\n
    프로젝트의 주제는 "택배 물류 데이터의 종합적 분석 및 예측 플랫폼" 입니다. \n
    사용자는 사이드바의 대시보드 탭으로 이동해 태블로 대시보드 링크를 얻어 대시보드를 사용할 수 있으며,
    prediction 탭으로 이동해 원하는 기간의 물류량을 예측해 볼 수 있습니다.\n
    저희 프로젝트로 인해 물류의 재고관리 그리고 물류회사분들의 업무과중 방지에 도움이 되었으면 하는 바람입니다.


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

    <div class="footer">
        <p> 여기에 원하는 푸터 정보를 넣으면 된다. @ 2023 회사명</p>
    </div>

"""
# 현재 footer는 relactive 상태, fixed로 고정 가능

    st.markdown(html_css, unsafe_allow_html=True)

# Python 스크립트가 실행될 때 main() 함수를 호출합니다.
if __name__ == "__main__":
    main()

