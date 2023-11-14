# -*- coding:utf-8 -*-

import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd


def show_prediction():
    st.title("물류량 예측해보기")
    # 데이터 input 할 수 있게 하기
    # joblib으로 모델 불러오기



# 모델 파일을 불러오는 함수
def load_model(model_name):
    filename = os.path.join('./models', f'{model_name}_xgb_model.pkl')
    loaded_model = joblib.load(filename)
    return loaded_model

# 원-핫 인코딩된 터미널 이름 컬럼을 생성하는 함수
def create_encoded_columns(terminal_name):
    # 모든 가능한 터미널 이름을 나열합니다. 실제 모델에서 사용된 정확한 리스트를 사용해야 합니다.
    all_terminals = ['강남', '강동', '강북', '강서', '관악', '광진', '구로', '금천', '노원', '도봉', 
                     '동대문', '동작', '마포', '서대문', '서초', '성동', '성북', '송파', '양천', 
                     '영등포', '용산', '은평', '종로', '중구', '중랑']
    
    # 선택된 터미널에 대해서만 1을 설정하고 나머지는 0으로 설정합니다.
    encoded_data = {f'터미널이름_{terminal}': 1 if terminal == terminal_name else 0 for terminal in all_terminals}
    return encoded_data

def lets_prediction():
    print

'''
# 예측하려는 데이터 포인트를 준비
new_data = {
    '20-30대 비율': [27.47],
    '20대 미만 비율': [17.36],
    '40-50대 비율': [33.83],
    '60-70대 비율': [18.32],
    '80대 이상 비율':[3.01],
    '공휴일': [0],
    '년': [2023],
    '소득2천만원주민비율': [0.09],
    '소득3천만원주민비율': [0.28],
    '소득4천만원주민비율': [0.21],
    '소득5천만원주민비율': [0.13],
    '소득6천만원주민비율': [0.07],
    '소득7천만원이상주민비율': [0.18],
    '소득7천만원주민비율': [0.05],
    '요일': [2], 
    '월': [11],
    '일': [14],
    '총 인구': [534103],
}

# 원-핫 인코딩된 터미널 이름 컬럼을 추가합니다. 여기서는 '강남'을 예로 듭니다.
encoded_columns = create_encoded_columns('강남')
new_data.update(encoded_columns)

# 데이터 프레임으로 변환합니다.
new_data_df = pd.DataFrame(new_data)

# 모델명을 지정합니다. 예시로 '가구_인테리어'를 사용합니다.
model_name = '가구_인테리어'

# 모델을 불러옵니다.
model = load_model(model_name)

# 예측을 수행합니다.
prediction = model.predict(new_data_df)

# 예측 결과를 출력합니다.
print(f'예측된 {model_name.replace("_", "/")} 물류량:', prediction)
'''

if __name__ == "__main__":
    show_prediction()