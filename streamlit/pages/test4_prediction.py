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
    
    plt.figure(figsize=(10, 7))
    bars = plt.bar(labels, predictions, color=colors)
    
    # 막대 그래프 위에 값을 표시합니다.
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va='bottom', ha='center')
    
    plt.xlabel('품목')
    plt.ylabel('물류량')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

# 예측 페이지를 보여주는 함수
def show_prediction_page(all_terminals):
    st.set_page_config(layout="wide")
    st.title("물류량 예측")
    model_options = ['가구/인테리어', '기타', '도서/음반', '디지털/가전', '생활/건강', '스포츠/레저', '식품', '출산/육아', '패션의류', '패션잡화', '화장품/미용']
    selected_models = st.multiselect('품목을 선택해주세요', ['전체'] + model_options)
    start_date = st.date_input('시작 날짜를 선택하세요', datetime.today())
    end_date = st.date_input('종료 날짜를 선택하세요', datetime.today())
    holiday = st.selectbox('공휴일인가요?', ['예', '아니오']) == '예'
    if start_date > end_date:
        st.error('시작 날짜는 종료 날짜보다 클 수 없습니다.')
        return
    population = st.number_input('총 인구', min_value=0)
    terminal_name = st.selectbox('터미널 이름 선택', all_terminals)
    encoded_columns = create_encoded_columns(terminal_name, all_terminals)

    if st.button('예측'):
        # '전체'가 선택된 경우, 모든 품목의 예측값 합산
        if '전체' in selected_models and len(selected_models) == 1:
            selected_models = model_options
        elif '전체' in selected_models:
            selected_models.remove('전체')
        
        predictions = []
        labels = []
        total_prediction = 0
        # 모든 선택된 품목에 대해 예측을 실행
        for model_name in selected_models:
            # 모델 로드
            loaded_model = load_model(model_name)
            # 예측 수행
            for single_date in pd.date_range(start_date, end_date):
                # 데이터 준비
                new_data = {                
                '20-30대 비율': [27.47],
                '20대 미만 비율': [17.36],
                '40-50대 비율': [33.83],
                '60-70대 비율': [18.32],
                '80대 이상 비율':[3.01],
                '공휴일': [holiday],
                '년': [single_date.year],
                '소득2천만원주민비율': [0.09],
                '소득3천만원주민비율': [0.28],
                '소득4천만원주민비율': [0.21],
                '소득5천만원주민비율': [0.13],
                '소득6천만원주민비율': [0.07],
                '소득7천만원이상주민비율': [0.18],
                '소득7천만원주민비율': [0.05],
                '요일': [single_date.day_of_week],
                '월': [single_date.month],
                '일': [single_date.day],
                '총 인구': [population],#534103
            # ... 기타 필요한 변수들
             }
                new_data.update(encoded_columns)
                new_data_df = pd.DataFrame(new_data)

                # 예측
                prediction = loaded_model.predict(new_data_df)[0]
                total_prediction += prediction

            # 결과 저장
            predictions.append(total_prediction)
            labels.append(f"{model_name} / {terminal_name}")
        
        # 전체 물류량 그래프를 추가
        if '전체' in st.multiselect('품목을 선택해주세요', ['전체'] + model_options):
            predictions.insert(0, sum(predictions))
            labels.insert(0, '전체')

        # 그래프 그리기
        plt.clf()
        plot_bar_chart(predictions, labels)

        # 예측 결과 출력
        st.write(f'선택된 날짜 범위에 대한 예측된 물류량 합산: {sum(predictions):.2f}')

if __name__ == "__main__":
    show_prediction_page(['강남', '강동', '강북', ...])  # 터미널 이름 리스트를 채워주세요.
