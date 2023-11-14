import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
import joblib

# 기존 함수들...
# 모델을 불러오는 함수
def load_model(model_name):
    # 모델 파일명은 슬래시 대신 밑줄로 구분된 형식이어야 합니다.
    filename = os.path.join('./models/', f'{model_name.replace("/", "_")}_xgb_model.pkl')
    loaded_model = joblib.load(filename)
    return loaded_model

# 원-핫 인코딩된 터미널 이름 컬럼을 생성하는 함수
def create_encoded_columns(terminal_name, all_terminals):
    encoded_data = {f'터미널이름_{terminal}': 0 for terminal in all_terminals}
    encoded_data[f'터미널이름_{terminal_name}'] = 1
    return encoded_data

# 예측 페이지를 보여주는 함수
def show_prediction_page(all_terminals):
    st.title("물류량 예측")

    # 품목 선택
    model_options = ['가구/인테리어', '기타', '도서/음반', '디지털/가전', '생활/건강', 
                     '스포츠/레저', '식품', '출산/육아', '패션의류', '패션잡화', '화장품/미용']
    model_name = st.selectbox('품목을 선택해주세요', model_options)

    # 날짜 범위 선택
    start_date = st.date_input('시작 날짜를 선택하세요', datetime.today())
    end_date = st.date_input('종료 날짜를 선택하세요', datetime.today())

    if start_date > end_date:
        st.error('시작 날짜는 종료 날짜보다 클 수 없습니다.')
        return

    # 기타 사용자 입력
    population = st.number_input('총 인구', min_value=0)
    # ... 기타 필요한 변수들을 입력받습니다.
    
    terminal_name = st.selectbox('터미널 이름 선택', all_terminals)
    encoded_columns = create_encoded_columns(terminal_name, all_terminals)
    
    # 예측 버튼을 생성합니다.
    if st.button('예측'):
        total_prediction = 0
        # 선택된 날짜 범위에 대해 반복하여 예측 수행
        for single_date in pd.date_range(start_date, end_date):
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

            # 선택된 모델로 로드합니다.
            loaded_model = load_model(model_name)

            # 모델을 사용하여 예측을 수행합니다.
            prediction = loaded_model.predict(new_data_df)
            
            # 예측 결과를 합산합니다.
            total_prediction += prediction[0]

        # 총 예측 결과를 사용자에게 보여줍니다.
        st.write(f'선택된 날짜 범위에 대한 예측된 물류량 합산: {total_prediction:.2f}')

# 모든 가능한 터미널 이름 리스트와 모델 옵션...
# 모든 가능한 터미널 이름 리스트
all_terminals = ['강남', '강동', '강북', '강서', '관악', '광진', '구로', '금천', '노원', '도봉', 
                 '동대문', '동작', '마포', '서대문', '서초', '성동', '성북', '송파', '양천', 
                 '영등포', '용산', '은평', '종로', '중구', '중랑']

# 예측 페이지를 보여줍니다.
if __name__ == "__main__":
    show_prediction_page(all_terminals)

