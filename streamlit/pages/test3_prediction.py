import streamlit as st
import pandas as pd
from datetime import datetime
import os
import joblib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, rc
import numpy as np

# 사용자 정의 폰트 경로
font_path = './customFonts/NanumGothic-Regular.ttf'
# 폰트 매니저에 폰트를 등록
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
rc('font', family=font_prop.get_name())
# 한글 마이너스 기호 문제 해결을 위한 설정
rc('axes', unicode_minus=False)

# 모델을 불러오는 함수
def load_model(model_name):
    filename = os.path.join('./models/', f'{model_name.replace("/", "_")}_xgb_model.pkl')
    loaded_model = joblib.load(filename)
    return loaded_model

# 원-핫 인코딩된 터미널 이름 컬럼을 생성하는 함수
def create_encoded_columns(terminal_name, all_terminals):
    encoded_data = {f'터미널이름_{terminal}': 0 for terminal in all_terminals}
    encoded_data[f'터미널이름_{terminal_name}'] = 1
    return encoded_data

# 막대 그래프를 그리는 함수
def plot_bar_chart(predictions, labels):
    # 색상 배열을 생성합니다. 이 배열은 예측값의 개수만큼의 색상을 viridis 색상맵에서 균등하게 선택합니다.
    colors = plt.cm.viridis(np.linspace(0, 1, len(predictions)))
    
    if len(predictions) == 1:
        bar_width = 0.2
    elif len(predictions) == 2:
        bar_width = 0.4
    else :
        bar_width =0.6

    plt.figure(figsize=(10, 7))
    bars = []  # 막대 객체를 담을 리스트

    # 각 예측에 대해 개별적으로 막대를 그리고 레이블을 지정합니다.
    for i, (label, prediction) in enumerate(zip(labels, predictions)):
        bar = plt.bar(label, prediction, color=colors[i], width=bar_width, label=label)
        bars.append(bar)

    # 막대 그래프 위에 값을 표시합니다.
    for bar in bars:
        for item in bar:
            yval = item.get_height()
            plt.text(item.get_x() + item.get_width() / 2, yval, round(yval, 1), va='bottom', ha='center')
    
    plt.xlabel('품목')
    plt.ylabel('물류량')
    plt.xticks(rotation=90)

    # x축의 범위를 설정하여 뚱뚱한 막대가 나오지 않도록 조정합니다.
    plt.xlim(-0.5, len(labels)-0.5)
    
    plt.tight_layout()
    
    # 범례를 추가합니다. 'best' 위치에 표시되도록 설정합니다.
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=3)



    st.pyplot(plt)
# 예측 페이지를 보여주는 함수
def show_prediction_page(all_terminals):
    st.set_page_config(layout="wide")
    st.title("물류량 예측")
    model_options = ['가구/인테리어', '기타', '도서/음반', '디지털/가전', '생활/건강', '스포츠/레저', '식품', '출산/육아', '패션의류', '패션잡화', '화장품/미용']
    selected_models = st.multiselect('품목을 선택해주세요', model_options)
    start_date = st.date_input('시작 날짜를 선택하세요', datetime.today())
    end_date = st.date_input('종료 날짜를 선택하세요', datetime.today())
    if start_date > end_date:
        st.error('시작 날짜는 종료 날짜보다 클 수 없습니다.')
        return
    population = st.number_input('총 인구', min_value=0)
    terminal_name = st.selectbox('터미널 이름 선택', all_terminals)
    encoded_columns = create_encoded_columns(terminal_name, all_terminals)

    if st.button('예측'):
        if 'predictions' not in st.session_state:
            st.session_state.predictions = []
            st.session_state.labels = []
        models = {model_name: load_model(model_name) for model_name in selected_models}
        for model_name in selected_models:
            total_prediction = 0
            loaded_model = models[model_name]
            for single_date in pd.date_range(start_date, end_date):
                # ... 데이터 준비 및 예측 로직 ...
                new_data = {                
                '20-30대 비율': [27.47],
                '20대 미만 비율': [17.36],
                '40-50대 비율': [33.83],
                '60-70대 비율': [18.32],
                '80대 이상 비율':[3.01],
                '공휴일': [0],
                '년': [single_date.year],
                '소득2천만원주민비율': [0.09],
                '소득3천만원주민비율': [0.28],
                '소득4천만원주민비율': [0.21],
                '소득5천만원주민비율': [0.13],
                '소득6천만원주민비율': [0.07],
                '소득7천만원이상주민비율': [0.18],
                '소득7천만원주민비율': [0.05],
                '요일': [2],
                '월': [single_date.month],
                '일': [single_date.day],
                '총 인구': [population],#534103
            # ... 기타 필요한 변수들
             }
                new_data.update(encoded_columns)
                new_data_df = pd.DataFrame(new_data)
                prediction = loaded_model.predict(new_data_df)
                total_prediction += prediction[0]
            current_time = datetime.now()
            st.session_state.predictions.append(total_prediction)
            st.session_state.labels.append(f"{model_name} (time:{current_time.strftime('%H:%M:%S')})")
        if len(st.session_state.predictions) > 12:
            st.session_state.predictions = st.session_state.predictions[-12:]
            st.session_state.labels = st.session_state.labels[-12:]
        st.write(f'선택된 날짜 범위에 대한 예측된 물류량 합산: {sum(st.session_state.predictions):.2f}')
        plt.clf()
        plot_bar_chart(st.session_state.predictions, st.session_state.labels)
    
    if st.button('그래프 초기화'):
        st.session_state.predictions.clear()
        st.session_state.labels.clear()
        st.experimental_rerun()
all_terminals = ['강남', '강동', '강북', '강서', '관악', '광진', '구로', '금천', '노원', '도봉', '동대문', '동작', '마포', '서대문', '서초', '성동', '성북', '송파', '양천', '영등포', '용산', '은평', '종로', '중구', '중랑']
if __name__ == "__main__":
    show_prediction_page(all_terminals)

