import streamlit as st
import pandas as pd
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
# from google.cloud import storage
# 한글폰트 적용
# 폰트 적용
import os
import matplotlib.font_manager as fm  # 폰트 관련 용도 as fm


# GCS에서 모델 파일을 불러오는 함수
def load_model_from_gcs(bucket_name, model_file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(model_file_name)
    local_model_file = f'/tmp/{model_file_name}'
    blob.download_to_filename(local_model_file)
    model = joblib.load(local_model_file)
    return model

# 한글 폰트 설정
plt.rcParams['font.family'] = 'NanumBarunGothic'
plt.rcParams['axes.unicode_minus'] = False

# 막대 그래프를 그리는 함수
def plot_bar_chart(predictions, labels):
    plt.figure(figsize=(10, 5))
    plt.bar(labels, predictions, color='skyblue')
    plt.xlabel('품목')
    plt.ylabel('물류량')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

# 예측 페이지를 보여주는 함수
def show_prediction_page(all_terminals):
    st.title("물류량 예측")

    # 품목 선택
    model_options = [
        '가구/인테리어', '기타', '도서/음반', '디지털/가전', '생활/건강',
        '스포츠/레저', '식품', '출산/육아', '패션의류', '패션잡화', '화장품/미용'
    ]
    model_name = st.selectbox('품목을 선택해주세요', model_options)

    # 날짜 범위 선택
    start_date = st.date_input('시작 날짜를 선택하세요', datetime.today())
    end_date = st.date_input('종료 날짜를 선택하세요', datetime.today())

    if start_date > end_date:
        st.error('시작 날짜는 종료 날짜보다 클 수 없습니다.')
        return

    # 기타 사용자 입력
    population = st.number_input('총 인구', min_value=0)

    terminal_name = st.selectbox('터미널 이름 선택', all_terminals)
    
    # 예측 버튼을 생성합니다.
    if st.button('예측'):
        total_prediction = 0
        predictions_list = []
        labels_list = []
        
        # 선택된 날짜 범위에 대해 반복하여 예측 수행
        for single_date in pd.date_range(start_date, end_date):
            new_data = {
                # 예시 데이터 - 실제 데이터로 교체 필요
                '년': [single_date.year],
                '월': [single_date.month],
                '일': [single_date.day],
                '총 인구': [population],
                # ... 기타 필요한 변수들을 입력받습니다.
            }
            new_data_df = pd.DataFrame(new_data)
            
            # 모델 로드
            loaded_model = load_model_from_gcs('your-bucket-name', 'model-file.pkl')
            
            # 예측 수행
            prediction = loaded_model.predict(new_data_df)
            
            # 예측 결과 합산
            total_prediction += prediction[0]
            predictions_list.append(prediction[0])
            current_time = datetime.now()
            labels_list.append(f"{model_name} ({current_time.strftime('%H:%M:%S')})")

        # 예측 결과 출력
        st.write(f'선택된 날짜 범위에 대한 예측된 물류량 합산: {total_prediction:.2f}')
        
        # 막대 그래프로 결과 표시
        plot_bar_chart(predictions_list, labels_list)

all_terminals = [
    '강남', '강동', '강북', '강서', '관악', '광진', '구로', '금천', '노원', '도봉', 
    '동대문', '동작', '마포', '서대문', '서초', '성동', '성북', '송파', '양천', 
    '영등포', '용산', '은평', '종로', '중구', '중랑'
]

if __name__ == "__main__":
    show_prediction_page(all_terminals)


