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
    filename = os.path.join('./models/', f'{model_name.replace("/", "_")}_xgb_model2.pkl')
    loaded_model = joblib.load(filename)
    return loaded_model

# 원-핫 인코딩된 터미널 이름 컬럼을 생성하는 함수
def create_encoded_columns(terminal_name, all_terminals):
    # 모든 가능한 구 이름을 나열합니다.
    encoded_data = {f'WORK_GU_{terminal}': 1 if terminal == terminal_name else 0 for terminal in all_terminals}
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
    plt.xticks(range(len(labels)),[label.split("_")[0] for label in labels],rotation=90)

    # x축의 범위를 설정하여 뚱뚱한 막대가 나오지 않도록 조정합니다.
    plt.xlim(-0.5, len(labels)-0.5)
    
    plt.tight_layout()   
    # 범례를 추가합니다. 'best' 위치에 표시되도록 설정합니다.
    plt.legend([label.split("_")[1] + "/" + label.split("_")[2] for label in labels],loc='lower center', bbox_to_anchor=(0.5, 1), ncol=3)

    st.pyplot(plt)

# 예측 페이지를 보여주는 함수
def show_prediction_page(all_terminals):
    st.set_page_config(layout="wide")
    st.title("물류량 예측")
    # type_options = ['전체 물류량', '수입', '수출']
    # selected_types = st.selectbox['어떤 물류를 확인해 볼까요?', type_options]
    model_options = ['전체', '가구/인테리어', '기타', '도서/음반', '디지털/가전', '생활/건강', '스포츠/레저', '식품', '출산/육아', '패션의류', '패션잡화', '화장품/미용']
    selected_models = st.multiselect('품목을 선택해주세요', model_options)
    start_date = st.date_input('시작 날짜를 선택하세요', datetime.today())
    end_date = st.date_input('종료 날짜를 선택하세요', datetime.today())
    holiday_list = ['아니오', '예']
    holiday = st.selectbox('공휴일인가요?', holiday_list)
    holiday = 1 if holiday == '예' else 0
    if start_date > end_date:
        st.error('시작 날짜는 종료 날짜보다 클 수 없습니다.')
        return
    terminal_name = st.selectbox('터미널 이름 선택', all_terminals)
    encoded_columns = create_encoded_columns(terminal_name, all_terminals)

    if st.button('예측'):
        if '전체' in selected_models and len(selected_models) > 1:
            st.error('전체를 선택한 경우 다른 품목은 선택할 수 없습니다.')
            return
        
        if 'predictions' not in st.session_state:
            st.session_state.predictions = []
            st.session_state.labels = []

        total_predictions = []
        labels = []

        if '전체' in selected_models:
            selected_models = model_options[1:]  # 전체 품목을 가져옵니다.

        models = {model_name: load_model(model_name) for model_name in selected_models}
        
        
        for model_name in selected_models:
            total_prediction = 0
            loaded_model = models[model_name]
            for single_date in pd.date_range(start_date, end_date):
                new_data = {
                '년': [single_date.year],
                '월': [single_date.month],
                '일': [single_date.day],
                '요일': [single_date.day_of_week],
                '공휴일': [holiday],
            # ... 기타 필요한 변수들
             }
            year = max(2021, min(single_date.year, 2023))
            if year == 2023:
                gu_df = gu_df23
            if year == 2022:
                gu_df = gu_df22
            if year == 2021:
                gu_df = gu_df21

                new_data.update(dict(zip(gu_columns, gu_df)))
                new_data.update(encoded_columns)
                new_data_df = pd.DataFrame(new_data)
                prediction = loaded_model.predict(new_data_df)
                total_prediction += max(prediction[0], 0)
            current_time = datetime.now()
            label = f"{model_name}_{terminal_name}_{start_date.strftime('%Y-%m-%d')}~{end_date.strftime('%Y-%m-%d')}_(time:{current_time.strftime('%H:%M:%S')})"
            labels.append(label)
            total_predictions.append(total_prediction)
        
        st.session_state.predictions.extend(total_predictions)
        st.session_state.labels.extend(labels)

        if len(st.session_state.predictions) > 12:
            st.session_state.predictions = st.session_state.predictions[-12:]
            st.session_state.labels = st.session_state.labels[-12:]

        st.write(f'예측된 물류량 합산: {sum(st.session_state.predictions):.2f}')
        plt.clf()
        plot_bar_chart(st.session_state.predictions, st.session_state.labels)

    if st.button('그래프 초기화'):
        st.session_state.predictions.clear()
        st.session_state.labels.clear()
        st.experimental_rerun()

all_terminals = ['강남구', '강동구', '강북구', '강서구', '관악구', '광진구', '구로구', '금천구', '노원구', '도봉구', 
                     '동대문구', '동작구', '마포구', '서대문구', '서초구', '성동구', '성북구', '송파구', '양천구', 
                     '영등포구', '용산구', '은평구', '종로구', '중구', '중랑구']

# 필요한 컬럼들을 리스트로 정의합니다.
gu_columns = ['TOTAL_PEOPLE', 'RATE_UNDER 20s', 'RATE_20s-30s', 'RATE_40s-50s', 
                 'RATE_60s-70s', 'RATE_OVER 80s', 'RATE_20M_Won', 'RATE_30M_Won', 'RATE_40M_Won', 'RATE_50M_Won', 'RATE_60M_Won', 'RATE_70M_Won', 
                 'RATE_OVER_70M_Won', 'APT_PRICE_INDEX', 'APT_NUM', 'APT_RATE', 'NUM_INSU', 'INSU_MONEY', 'FOOD_TRASH', 'BIRTH_RATE', 'TOTAL_OIL', 
                 'GASOLINE', 'KEROSENE', 'DIESEL', 'LPG', 'OTHER_OIL', 'TOTAL_GASUSE', 'GASUSE_HOME', 'GASUSE_COMMON', 'GASUSE_WORK', 'GASUSE_IND', 'GASUSE_HEAT', 
                 'GASUSE_TRF', 'NUM_HYDCAR', 'TOTAL_ALONE', 'ALONE_65-79', 'ALONE_OVER_80', 'TOTAL_ALONE_BASIC', 'ALONE_BASIC_65-79', 'ALONE_BASIC_OVER_80', 
                 'TOTAL_ALONE_LOW', 'ALONE_LOW_65-79', 'ALONE_LOW_OVER_80', 'TOTAL_ALONE_COMM', 'ALONE_COMM_65-79', 'ALONE_COMM_OVER_80', 'TOTAL_HOUSE', 'NUM_DANDOK', 
                 'NUM_COMDAN', 'NUM_MULTI', 'NUM_WITHSALES', 'NUM_TOWNH', 'NUM_DASEDE', 'NUM_NOLIVE', 'PEOPEL_DENS', 'TOTAL_SD', 'SD_1MEM', 'SD_2MEM', 'SD_3MEM', 'SD_4MEM', 
                 'SD_5MEM', 'SD_6MEM', 'SD_7MEM', 'SD_8MEM', 'SD_9MEM', 'SD_10MEM', 'RATE_VAC_1ST', 'RATE_VAC_2ND', 'RATE_VAC_3RD', 'RATE_VAC_4TH', 'RATE_VAC_VER2', 
                 'TOTAL_AVG_AGE', 'MEN_AVG_AGE', 'WOMEN_AVG_AGE', 'TOTAL_LIB', 'TOTAL_REC', 'RESTAURANTS', 'CATERING', 'FOOD_PROCESS', 'FOOD_TRANS_SALES', 
                 'HEALTH_FUNCTION_SALES', 'TOTAL_ELEC', 'RESIDENTIAL', 'GENERAL', 'EDUCATIONAL', 'INDUSTRIAL', 'AGRICULTURAL', 'STREETLIGHT', 'NIGHTUSE', 'MANUFACTURING', 
                 'RETAIL', 'TOTAL_MED']
gu_df23 = [387940.0, 11.4, 28.57, 30.86, 24.89, 4.28, 0.12, 0.39, 0.27, 0.11, 0.05, 0.02, 0.03, 95.7, 95.0, 95.95959595959596, 11211742.0, 587758007.0, 86.7, 0.65, 784.0, 264.0, 10.0, 271.0, 238.0, 0.0, 151513.0, 111741.0, 14726.0, 3914.0, 784.0, 13.0, 20334.0, 15.0, 20695.0, 15724.0, 4971.0, 7778.0, 5365.0, 2413.0, 470.0, 295.0, 175.0, 12447.0, 10064.0, 2383.0, 118357.0, 17751.0, 1668.0, 12765.0, 3318.0, 4466.0, 35535.0, 1484.0, 21092.0, 187413.0, 83492.0, 45637.0, 30591.0, 21598.0, 4854.0, 955.0, 221.0, 39.0, 17.0, 9.0, 87.92, 87.19, 66.06, 15.39, 13.41, 46.2, 45.4, 47.0, 5.0, 6974.0, 5095.0, 215.0, 579.0, 319.0, 766.0, 1161372.0, 561516.0, 481863.0, 31298.0, 72104.0, 378.0, 10630.0, 3582.0, 47.0, 768.0, 815.0] 
gu_df22 = [390140.0, 11.73, 28.82, 31.26, 24.17, 4.03, 0.12, 0.39, 0.27, 0.11, 0.05, 0.02, 0.03, 95.7, 95.0, 95.95959595959596, 11211742.0, 587758007.0, 86.7, 0.659, 784.0, 264.0, 10.0, 271.0, 238.0, 0.0, 151513.0, 111741.0, 14726.0, 3914.0, 784.0, 13.0, 20334.0, 15.0, 20695.0, 15724.0, 4971.0, 7778.0, 5365.0, 2413.0, 470.0, 295.0, 175.0, 12447.0, 10064.0, 2383.0, 118357.0, 17751.0, 1668.0, 12765.0, 3318.0, 4466.0, 35535.0, 1484.0, 21092.0, 187413.0, 83492.0, 45637.0, 30591.0, 21598.0, 4854.0, 955.0, 221.0, 39.0, 17.0, 9.0, 87.92, 87.19, 66.06, 15.39, 13.41, 45.3, 44.5, 46.1, 5.0, 6974.0, 5095.0, 215.0, 579.0, 319.0, 766.0, 1161372.0, 561516.0, 481863.0, 31298.0, 72104.0, 378.0, 10630.0, 3582.0, 47.0, 768.0, 815.0]
gu_df21 = [391885.0, 12.05, 28.94, 31.81, 23.52, 3.67, 0.12, 0.39, 0.27, 0.11, 0.05, 0.02, 0.03, 95.7, 95.0, 95.95959595959596, 11211742.0, 587758007.0, 86.7, 0.65, 784.0, 264.0, 10.0, 271.0, 238.0, 0.0, 150392.0, 109582.0, 13247.0, 3699.0, 1422.0, 2.0, 22440.0, 15.0, 19260.0, 14851.0, 4409.0, 7074.0, 4865.0, 2209.0, 1366.0, 833.0, 533.0, 10820.0, 9153.0, 1667.0, 114488.0, 17995.0, 1701.0, 12967.0, 3327.0, 4423.0, 33916.0, 1503.0, 21188.0, 187413.0, 83492.0, 45637.0, 30591.0, 21598.0, 4854.0, 955.0, 221.0, 39.0, 17.0, 9.0, 87.92, 87.19, 66.06, 15.39, 13.41, 45.7, 44.9, 46.5, 5.0, 6974.0, 5095.0, 215.0, 579.0, 319.0, 766.0, 1161372.0, 561516.0, 481863.0, 31298.0, 72104.0, 378.0, 10630.0, 3582.0, 47.0, 768.0, 815.0]

if __name__ == "__main__":
    show_prediction_page(all_terminals)
