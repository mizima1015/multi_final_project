import streamlit as st
import pandas as pd
from datetime import datetime
import os
import joblib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, rc
import numpy as np
from datetime import timedelta


# 사용자 정의 폰트 경로
font_path = './customFonts/NanumGothic-Regular.ttf'
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
rc('font', family=font_prop.get_name())
rc('axes', unicode_minus=False)



# 모델을 불러오는 함수
def load_model(model_name, model_type):
    model_suffix = {
        '전체 물류량': '',
        '수신': '_rec',
        '발송': '_send'
    }
    filename = os.path.join('./models/', f'{model_name.replace("/", "_")}{model_suffix[model_type]}_xgb_model.pkl')
    loaded_model = joblib.load(filename)
    return loaded_model

# 원-핫 인코딩된 터미널 이름 컬럼을 생성하는 함수
def create_encoded_columns(terminal_name, all_terminals):
    encoded_data = {f'WORK_GU_{terminal}': 1 if terminal == terminal_name else 0 for terminal in all_terminals}
    return encoded_data

def create_rec_encoded_columns(city_name, all_city, terminal_name, all_terminals):
    encoded_city = {f'SEND_CITY_{city}': 1 if city == city_name else 0 for city in all_city}
    encoded_terminal = {f'REC_GU_{terminal}': 1 if terminal == terminal_name else 0 for terminal in all_terminals}
    return {**encoded_city, **encoded_terminal}

def create_send_encoded_columns(city_name, all_city, terminal_name, all_terminals):
    encoded_terminal = {f'SEND_GU_{terminal}': 1 if terminal == terminal_name else 0 for terminal in all_terminals}
    encoded_city = {f'REC_CITY_{city}': 1 if city == city_name else 0 for city in all_city}
    return {**encoded_terminal, **encoded_city}

def hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.strip('#')
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:], 16)
    return (r/255, g/255, b/255, alpha)

# 막대 그래프를 그리는 함수
def plot_bar_chart(predictions, labels, city_name=None):
    COLORS = [
        "#FF5733", "#C70039", "#fca649", "#2a8ff5",
        "#56ff38", "#FF5733", "#fa7fea", "#f0fc90",
        "#9ab2f5", "#FFC300", "#DAF7A6"
    ]

    session_groups = {}
    for label in labels:
        session_id = label.split('_')[-2]
        if session_id not in session_groups:
            session_groups[session_id] = []
        session_groups[session_id].append(label)
    
    plt.figure(figsize=(10, 7))
    if len(predictions) == 1:
        bar_width = 0.2
    elif len(predictions) == 2:
        bar_width = 0.4
    else :
        bar_width =0.6

    # 색상 변화를 위한 설정
    for session_id, group_labels in session_groups.items():
        base_color = COLORS[int(session_id) % len(COLORS)]
        num_bars = len(group_labels)
        color_alpha_step = 1 / (num_bars + 1)

        for i, label in enumerate(group_labels):
            prediction_index = labels.index(label)
            prediction = predictions[prediction_index]
            alpha = 1 - (i * color_alpha_step)
            rgba_color = hex_to_rgba(base_color, alpha)
            plt.bar(prediction_index, prediction, color=rgba_color, width=bar_width)

            # 막대 그래프 위에 값을 표시합니다.
            yval = prediction
            if len(predictions) > 11:
                plt.text(prediction_index, yval, round(yval, 1),
                         va='center', ha='right', rotation=90, color='black', fontsize=8)
            else:
                plt.text(prediction_index, yval, round(yval, 1),
                         va='bottom', ha='center', color='black', fontsize=8)

                
    rotation_angle = 0 if len(labels) <= 5 else (60 if len(labels) <= 11 else 90)

    plt.xlabel('품목')
    plt.ylabel('물류량')
    plt.xticks(range(len(labels)),[f"{label.split('_')[1]}({label.split('_')[0]})" for label in labels],rotation=rotation_angle)
    plt.xlim(-0.5, len(labels)-0.5)    
    plt.tight_layout()   
    legend_labels = [f"{label.split('_')[2]} / {label.split('_')[3]} / {label.split('_')[4]}" for label in labels]
    plt.legend(legend_labels, loc='lower center', bbox_to_anchor=(0.5, 1), ncol=3)
    st.pyplot(plt)



# 예측 페이지를 보여주는 함수
def show_prediction_page(all_terminals, gu_columns, all_city):
    st.title("물류량 예측")
    with st.expander("페이지 설명"):
        st.write("""
                이 곳에선 구별 터미널이 처리하는 전체 총 물류량, 지방으로 내보내는 발송 물류량, 지방에서 들어오는 수신물류량을 구분해 예측을 해보실 수 있습니다. \n
                그래프는 33개까지 누적해서 그릴 수 있으며, 
                 예측 버튼을 누를 때마다 다른 계열의 색으로 그래프가 그려집니다.\n
                그래프를 처음부터 그리고 싶다면 아래의 그래프 초기화 버튼을 눌러주시기 바랍니다.

                품목 중 전체 옵션을 선택하시면 11개의 모든 품목 그래프를 한 번에 그리실 수 있습니다.
            """)
        
    st.sidebar.markdown("[Tableu(서울시)](https://public.tableau.com/app/profile/.13586515/viz/df3_1_re/12?publish=yes)")
    st.sidebar.markdown("[Tableu(자치구)](https://public.tableau.com/app/profile/.13586515/viz/df3_2_re/24?publish=yes)")
    
    html_css ="""
    <style>
    th,td{
        border-bottom: 1px solid #ddd;
    }
    .highlight{
        font-size: 25px;
        font-weight: bold;
        font-family: 'Arial', sans-serif;
    </style>

    """
    st.markdown(html_css, unsafe_allow_html=True)
    if 'predictions' not in st.session_state:
        st.session_state['predictions'] = []
    if 'labels' not in st.session_state:
        st.session_state['labels'] = []
    if 'cumulative_totals' not in st.session_state:
        st.session_state['cumulative_totals'] = []

    type_options = ['전체 물류량', '수신', '발송']
    selected_types = st.selectbox('어떤 물류를 확인해 볼까요?', type_options)
    model_options = ['전체', '가구/인테리어', '기타', '도서/음반', '디지털/가전', '생활/건강', '스포츠/레저', '식품', '출산/육아', '패션의류', '패션잡화', '화장품/미용']
    selected_models = st.multiselect('품목을 선택해주세요', model_options)
    start_date = st.date_input('시작 날짜를 선택하세요', datetime.today())
    end_date = st.date_input('종료 날짜를 선택하세요', datetime.today())
    date_range = pd.date_range(start_date, end_date)
    selected_holidays = st.multiselect('공휴일로 지정할 날짜를 선택하세요. (없다면 넘어가기)', date_range)

    if start_date > end_date:
        st.error('시작 날짜는 종료 날짜보다 클 수 없습니다.')
        return
    terminal_name = st.selectbox('터미널 이름 선택', all_terminals)

    if selected_types in ['수신', '발송']:
        city_name = st.selectbox('도시를 선택해주세요', all_city)
    else:
        city_name = None

    if selected_types == '전체 물류량':
        encoded_columns = create_encoded_columns(terminal_name, all_terminals)
    elif selected_types == '수신':
        encoded_columns = create_rec_encoded_columns(city_name, all_city, terminal_name, all_terminals)
    elif selected_types == '발송':
        encoded_columns = create_send_encoded_columns(city_name, all_city, terminal_name, all_terminals)

    if st.button('예측'):
        if 'color_num' not in st.session_state:
            st.session_state['color_num'] = 0
        st.session_state['color_num'] += 1

        if '전체' in selected_models and len(selected_models) > 1:
            st.error('전체를 선택한 경우 다른 품목은 선택할 수 없습니다.')
            return
        
        if 'predictions' not in st.session_state:
            st.session_state.predictions = []
            st.session_state.labels = []

        total_predictions = []
        labels = []

        if '전체' in selected_models:
            selected_models = model_options[1:]

        models = {model_name: load_model(model_name, selected_types) for model_name in selected_models}
        
        for model_name in selected_models:
            total_prediction = 0
            loaded_model = models[model_name]
            for single_date in pd.date_range(start_date, end_date):
                is_holiday = single_date in selected_holidays or single_date.day_of_week == 6
                is_weekend = single_date.weekday() >= 5
                day_after_holiday = (single_date - timedelta(days=1)) in selected_holidays
                day_of_week = 1 if day_after_holiday and not is_weekend else single_date.weekday() + 1
                new_data = {
                'YEAR': [single_date.year],
                'MONTH': [single_date.month],
                'DAY': [single_date.day],
                'DAY_OF': [7 if is_holiday else day_of_week],
                'HOLIDAY': [1 if is_holiday else 0],
             }
                year = single_date.year
                if year >= 2023:
                    year = 2023
                if year <= 2021:
                    year = 2021
                gu_df = df2[(df2['YEAR'] == year) & (df2['WORK_GU'] == terminal_name)].iloc[0]
                for column in gu_columns:
                    new_data[column] = [gu_df[column]]
                
                new_data.update(encoded_columns)
                new_data_df = pd.DataFrame(new_data)
                prediction = loaded_model.predict(new_data_df)
                total_prediction += max(prediction[0], 0)
            current_time = datetime.now()
            
            # 예측 버튼을 클릭했을 때 레이블을 생성하는 부분
            if city_name:
                label = f"{selected_types}_{model_name}_{terminal_name}_{city_name}_{start_date.strftime('%Y-%m-%d')}~{end_date.strftime('%Y-%m-%d')}_{st.session_state['color_num']}_(time:{current_time.strftime('%H:%M:%S')})"
            else:
                label = f"{selected_types}_{model_name}_{terminal_name}_{start_date.strftime('%Y-%m-%d')}~{end_date.strftime('%Y-%m-%d')}_ _{st.session_state['color_num']}_(time:{current_time.strftime('%H:%M:%S')})"

            labels.append(label)
            
            total_predictions.append(total_prediction)
        
        st.session_state.cumulative_totals.append(sum(total_predictions))
        st.session_state.predictions.extend(total_predictions)
        st.session_state.labels.extend(labels)

        if len(st.session_state.predictions) > 33:
            st.session_state.predictions = st.session_state.predictions[-33:]
            st.session_state.labels = st.session_state.labels[-33:]
            
        if len(st.session_state.cumulative_totals) > 12:
            st.session_state.cumulative_totals = st.session_state.cumulative_totals[-12:]

        st.write(f'예측된 물류량 합산: {sum(st.session_state.predictions):.2f}')
        plt.clf()
        plot_bar_chart(st.session_state.predictions, st.session_state.labels, city_name)

    if st.button('그래프 초기화'):
        st.session_state.predictions.clear()
        st.session_state.labels.clear()
        st.session_state.cumulative_totals.clear()
        st.experimental_rerun()
        plot_bar_chart(st.session_state.predictions, st.session_state.labels, selected_types, city_name)
    
    html_css = """
    <style>
    th,td{
        border-bottom: 1px solid #ddd;
    }
    img{
        margin-bottom:20px;
    }
    .multicampus{
        font-size: 24px;
        color: rgb(255,229,180);
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

    """
    st.markdown(html_css, unsafe_allow_html=True)

all_terminals = ['강남구', '강동구', '강북구', '강서구', '관악구', '광진구', '구로구', '금천구', '노원구', '도봉구', 
                     '동대문구', '동작구', '마포구', '서대문구', '서초구', '성동구', '성북구', '송파구', '양천구', 
                     '영등포구', '용산구', '은평구', '종로구', '중구', '중랑구']

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

all_city = ['강원','경기','경남', '경북', '광주', '대구', '대전', '부산', '서울', 
            '세종', '울산', '인천', '전남', '전북', '제주', '충남', '충북']

df2 = pd.read_csv("./models/work_info.csv")


def add_bg_from_url():
    total = round(sum(st.session_state.predictions), 1) if st.session_state.predictions else '-'
    max_val = round(max(st.session_state.predictions), 1) if st.session_state.predictions else '-'
    min_val = round(min(st.session_state.predictions), 1) if st.session_state.predictions else '-'
    avg_prediction = round(np.mean(st.session_state.predictions), 1) if st.session_state.predictions else '-'

    # 'total'과 'max_val'을 float 타입으로 변환합니다.
    total_float = float(total) if st.session_state.predictions else 0
    max_val_float = float(max_val) if st.session_state.predictions else 0
    min_val_float = float(min_val) if st.session_state.predictions else 0
    avg_val_float = float(avg_prediction) if st.session_state.predictions else 0
    

    # 'total'이 0보다 클 경우에만 계산을 수행합니다.
    to_max = round((max_val_float / total_float) * 100, 1) if total_float > 0 else '-'
    to_min = round((min_val_float / total_float) * 100, 1) if total_float > 0 else '-'
    av_max = round(((max_val_float/avg_val_float)*100)-100,1) if total_float > 0 else '-'
    av_min = round(((1-(min_val_float/avg_val_float))*100),1) if total_float > 0 else '-'

    cumulative_totals = st.session_state.cumulative_totals
    # 테이블 셀에 값을 채우는데 사용할 최대 인덱스
    max_index = len(cumulative_totals) - 1
    # 첫 번째 행과 두 번째 행의 HTML 초기화
    first_row_html = ""
    second_row_html = ""

    # 증감율을 계산하기 위한 이전 값 초기화
    previous_value = None

    # 첫 6차 예측까지의 값을 첫 번째 행에 추가
    for i in range(6):
        current_value = round(float(cumulative_totals[i]), 1) if i <= max_index else "-"
        change_percent = ""
        if previous_value is not None and current_value != "-":
            change_percent = f"({round((current_value - previous_value) / previous_value * 100, 1):+}%)"
        value_with_change = f"{current_value}{change_percent}"
        first_row_html += f"<td>{value_with_change}</td>"
        previous_value = current_value if current_value != "-" else previous_value

    # 7차부터 11차 예측까지의 값을 두 번째 행에 추가
    for i in range(6, 12):
        current_value = round(float(cumulative_totals[i]), 1) if i <= max_index and i < len(cumulative_totals) else "-"
        change_percent = ""
        if previous_value is not None and current_value != "-":
            change_percent = f"({round((current_value - previous_value) / previous_value * 100, 1):+}%)"
        value_with_change = f"{current_value}{change_percent}"
        second_row_html += f"<td>{value_with_change}</td>"
        previous_value = current_value if current_value != "-" else previous_value

    st.markdown(
         f"""
            <div>
            <그래프 분석>

            </div>
            <table>
                <thead>
                    <tr>
                        <th>총물류량(건)</th>
                        <th>평균값</th>                        
                        <th>최댓값</th>
                        <th>백분위(%)</th>
                        <th>평균대비증감(%)</th>
                        <th>최솟값</th>
                        <th>백분위(%)</th>
                        <th>평균대비증감(%)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>{total}</td>
                        <td>{avg_prediction}</td>
                        <td>{max_val}</td>
                        <td>{to_max}</td>
                        <td>+ {av_max}</td>
                        <td>{min_val}</td>
                        <td>{to_min}</td>
                        <td>- {av_min}</td>
                    </tr>
                </tbody>
            </table>
            <div>
            <p></p>        
            <예측 회차별 증감>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>1차 예측</th>
                        <th>2차 예측</th>
                        <th>3차 예측</th>
                        <th>4차 예측</th>
                        <th>5차 예측</th>
                        <th>6차 예측</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        {first_row_html}
                    </tr>
                    <tr>
                        <th>7차 예측</th>
                        <th>8차 예측</th>
                        <th>9차 예측</th>
                        <th>10차 예측</th>
                        <th>11차 예측</th>
                        <th>12차 예측</th>
                    </tr>
                    <tr>
                        {second_row_html}
                    </tr>
                </tbody>
            </table>

            <div class="footer">
            <p> 추가로 더 알아봤으면 하는 변수가 있다면 아래 메일로 문의주시기 바랍니다. </p>   
            <p> mizimaaz@gmail.com  /  2023 DeliverEase </p>
        
            현재 예측 모델을 만들기 위해 사용한 변수 columns는 총 110개이며, 그 리스트는 아래와 같습니다.

            columns_list=['YEAR', 'MONTH', 'DAY', 'DAY_OF', 'HOLIDAY', 'TOTAL_PEOPLE', 'RATE_UNDER 20s', 'RATE_20s-30s', 'RATE_40s-50s', 'RATE_60s-70s', 'RATE_OVER 80s', 'RATE_20M_Won', 'RATE_30M_Won', 'RATE_40M_Won', 'RATE_50M_Won', 'RATE_60M_Won', 'RATE_70M_Won', 'RATE_OVER_70M_Won', 'APT_PRICE_INDEX', 'APT_NUM', 'APT_RATE', 'NUM_INSU', 'INSU_MONEY', 'FOOD_TRASH', 'BIRTH_RATE', 'TOTAL_OIL', 'GASOLINE', 'KEROSENE', 'DIESEL', 'LPG', 'OTHER_OIL', 'TOTAL_GASUSE', 'GASUSE_HOME', 'GASUSE_COMMON', 'GASUSE_WORK', 'GASUSE_IND', 'GASUSE_HEAT', 'GASUSE_TRF', 'NUM_HYDCAR', 'TOTAL_ALONE', 'ALONE_65-79', 'ALONE_OVER_80', 'TOTAL_ALONE_BASIC', 'ALONE_BASIC_65-79', 'ALONE_BASIC_OVER_80', 'TOTAL_ALONE_LOW', 'ALONE_LOW_65-79', 'ALONE_LOW_OVER_80', 'TOTAL_ALONE_COMM', 'ALONE_COMM_65-79', 'ALONE_COMM_OVER_80', 'TOTAL_HOUSE', 'NUM_DANDOK', 'NUM_COMDAN', 'NUM_MULTI', 'NUM_WITHSALES', 'NUM_TOWNH', 'NUM_DASEDE', 'NUM_NOLIVE', 'PEOPEL_DENS', 'TOTAL_SD', 'SD_1MEM', 'SD_2MEM', 'SD_3MEM', 'SD_4MEM', 'SD_5MEM', 'SD_6MEM', 'SD_7MEM', 'SD_8MEM', 'SD_9MEM', 'SD_10MEM', 'RATE_VAC_1ST', 'RATE_VAC_2ND', 'RATE_VAC_3RD', 'RATE_VAC_4TH', 'RATE_VAC_VER2', 'TOTAL_AVG_AGE', 'MEN_AVG_AGE', 'WOMEN_AVG_AGE', 'TOTAL_LIB', 'TOTAL_REC', 'RESTAURANTS', 'CATERING', 'FOOD_PROCESS', 'FOOD_TRANS_SALES', 'HEALTH_FUNCTION_SALES', 'TOTAL_ELEC', 'RESIDENTIAL', 'GENERAL', 'EDUCATIONAL', 'INDUSTRIAL', 'AGRICULTURAL', 'STREETLIGHT', 'NIGHTUSE', 'MANUFACTURING', 'RETAIL', 'TOTAL_MED', 'work_gu']
    
            </div>
         """,
         unsafe_allow_html=True
     )



if __name__ == "__main__":
    show_prediction_page(all_terminals, gu_columns, all_city)
    add_bg_from_url()