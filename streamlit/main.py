# -*- coding:utf-8 -*-
import streamlit as st
from pages import interesting, prediction, team
 
# nohup streamlit run main.py > streamlit.log 2>&1 &
# ----- 서버 streamlit 실행-----

# 링크 : http://34.64.204.232:8501/
# 링크2: http://deliverease-multifpp.com
# ----- 서비스 접속 링크 -----

# sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 8501
# sudo apt-get install iptables-persistent
# ------ 포트포워딩 -----

st.set_page_config(layout="wide")

def main():  
    image_path = "./images/main3.png"
    st.image(image_path, caption='', use_column_width=True)
    st.sidebar.markdown("[Tableu(서울시)](https://public.tableau.com/app/profile/.13586515/viz/df3_1_re/sheet14_1?publish=yes)")
    st.sidebar.markdown("[Tableu(자치구)](https://public.tableau.com/app/profile/.13586515/viz/df3_2_re/sheet20?publish=yes)")

    
    html_css = """
    <style>
    th,td{
        border-bottom: 1px solid #ddd;
    }
    img{
        margin-bottom:20px;
    }
    .highlight{
        font-size: 16px;
        font-weight: bold;
        color: rgb(255, 0, 0);
        font-family: 'Arial', sans-serif;
    }
    .highlightblack{
        font-size: 25px;
        font-weight: bold;
        font-family: 'Arial', sans-serif;
        color: rgb(74, 105, 240);
    }
     .highlightblack2{
        font-size: 20px;
        font-weight: bold;
        font-family: 'Arial', sans-serif;
    }
     .highlightblack3{
        font-size: 16px;
        font-weight: bold;
        font-family: 'Arial', sans-serif;
        color: rgb(161, 181, 227);
    }
     .highlightblack4{
        font-size: 16px;
        font-weight: bold;
        font-family: 'Arial', sans-serif;
    }
     .highlightblack5{
        font-size: 16px;
        font-weight: bold;
        font-family: 'Arial', sans-serif;
        color: rgb(250, 171, 67);
    }    
        

    @font-face {
        font-family: 'NanumGothic';
        src: url('./customFonts/NanumGothic-Bold.ttf') format('truetype');
    }
    .font1 {
        font-family: 'NanumGothic';
    }
    .multicampus{
        font-size: 24px;
        color: rgb(255,229,180);
    }
    # special-text{
        font-size: 20px;
        font-weight: bold;
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
    
    안녕하세요, <span class = highlightblack>DeliverEase</span>에 오신 것을 환영합니다. \n
    저희는 <span class = highlightblack5>[멀티잇]</span><span class = highlightblack2> 데이터 분석가 및 엔지니어</span> 28회차 국비 지원 취업 교육을 성공적으로 수료한 전문 팀입니다.\n
    2020년도 코로나 팬데믹으로 원격 업무 및 배송 트렌드 급증으로 물류량이 상승하면서, 운송업체의 업무 과중 문제가 부상했습니다. 저희 프로젝트는 이러한 도전에 대응하여 "택배 물류 데이터 종합 분석 및 예측 플랫폼"을 기획하였습니다.  사이드 바에는 Interesting, Prediction, Team, 그리고 Tableau link 가 있습니다.\n 
    <span class = highlightblack3>"Interesting"</span>에서는 수치화 어려운 변수를 시각적으로 확인하고, <span class = highlightblack3>"Prediction"</span>에서는 특정 기간 동안의 물류량을 확인해볼 수 있습니다. <span class = highlightblack3>"Team"</span>은 팀원의 역할을  간략히 소개하며, <span class = highlightblack3>"Tableau Link"</span>에서는 Tableau로 즉시 연결하여 더 심층적인 데이터 분석이 가능합니다.\n 
    Tableau 대시보드 에서는 물류 데이터를 모니터링하는 데 특화된 시각화 도구로, 직관적이고 다양한 차트 및 대시보드를 통해 조회기간 내 데이터 트래킹을 제공하고, Streamlit에서는 예측 결과를 사용자에게 사용자 친화적이고 인터랙티브한 예측 환경을 제공합니다.\n
    저희는 이 프로젝트를 통해 효율적인 물류 재고 관리와 관련 기업 종사자의 업무과중을 해소하는 데 일조하길 기대하고 있습니다. 여러분의 운영에 혁신적인 솔루션을 제공하기 위해 노력하고 있으니, 저희의 플랫폼을 확인하고 경험해보시기 바랍니다.\n   
    <br><br><br>
    <span class = highlightblack4><태블로 설명></span>\n
    태블로 대시보드는 <span class = highlightblack3>'서울시 전체 대시보드', '서울시 구별 대시보드'</span>로 두 가지를 만들었습니다. 이 대시보드에서는 태블로 시각화 분석 툴을 이용하여 지도시각화, 다양한 방식으로 표현한 그래프를 나타냈습니다. \n
    지도시각화로 총 택배물류량을 알 수 있고 서브의 위치를 알수 있습니다. 또한 선, 막대, 트리맵의 다양한 그래프 모양을 이용해 물류량과 추이변화량을 한눈에 알아보기 쉽게 나타냈습니다. 조회기간 내 물류량의 최소값과 최대값을 알 수 있고 증감율도 확인해볼 수 있습니다.


    <span class = highlight> 주의) 현재 Tableau는 21년 1월 부터 23년 8월 데이터만 포함되어 있어 해당 기간 내의 물류량만 확인 가능합니다.</span> 


    <div class="footer">
        <p> mizimaaz@gmail.com  /  2023 DeliverEase </p>
    </div>
    """
    st.markdown(html_css, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

