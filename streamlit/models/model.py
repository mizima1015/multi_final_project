# 모델을 만들 때 사용한 코드 (work) (send와 rec는 변형이 필요)

import pandas as pd
df = pd.read_csv("df.csv")
df.head()
num_rows = len(df)
print("행의 개수:", num_rows)
df.drop('SUB_NAME', axis=1, inplace = True)
# 요일을 숫자로 변환합니다 (예: '월'=1, '화'=2, ..., '일'=7)
days = {'월': 1, '화': 2, '수': 3, '목': 4, '금': 5, '토': 6, '일': 7}
df['DAY_OF'] = df['DAY_OF'].map(days)
df.drop('DATE', axis=1, inplace = True)
dependent_vars = [
    'FURNITURE/INTERIOR',
    'ETC',
    'BOOK/ALBUM',
    'DIGITAL/HOME APP',
    'LIFE/HEALTH',
    'SPORTS/LEISURE',
    'FOOD',
    'CHILDBIRTH/PARENTING',
    'FASHION/CLOTHES',
    'FASHION/ACC',
    'COSMETICS/BEAUTY'    
]
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error

# 데이터 로딩
data = df.copy()

# 원-핫 인코딩
encoder = OneHotEncoder(sparse_output=False)
data_encoded = encoder.fit_transform(data[['WORK_GU']])
encoded_work_gu_df = pd.DataFrame(data_encoded, columns=encoder.get_feature_names_out(['WORK_GU']))

# 인코딩된 컬럼을 기존 데이터에 합치고, 원래 'WORK_GU' 컬럼은 삭제
data = data.drop('WORK_GU', axis=1)
data = pd.concat([data, encoded_work_gu_df], axis=1)

# 특성 및 타겟 설정
X = data.drop(dependent_vars, axis=1)
y = data[dependent_vars]

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 하이퍼파라미터 분포 설정
param_distributions = {
    'n_estimators': range(50, 401, 50),
    'max_depth': range(3, 11),
    'learning_rate': np.arange(0.01, 0.3, 0.01),
    'subsample': np.arange(0.5, 1.01, 0.1),
    'colsample_bytree': np.arange(0.5, 1.01, 0.1),
    'min_child_weight': range(1, 6),
    'gamma': np.arange(0.0, 5.1, 0.1)
}

# 각 종속 변수에 대한 최적 모델을 저장할 딕셔너리
best_models = {}
r2_scores = {}
rmse_scores = {}

# 각 종속 변수에 대해 최적화 수행
for target in dependent_vars:
    # 모델 초기화, enable_categorical=True를 사용하여 범주형 특성 지원 활성화
    xgb_model = XGBRegressor(enable_categorical=True)
    
    # RandomizedSearchCV 설정
    random_search = RandomizedSearchCV(
        estimator=xgb_model, 
        param_distributions=param_distributions, 
        n_iter=100,
        scoring='neg_mean_squared_error', 
        cv=5,
        verbose=1,
        random_state=42
    )
    
    # 해당 타겟에 대한 훈련 데이터
    y_train_target = y_train[target]
    
    # 교차 검증과 하이퍼파라미터 튜닝을 실행
    random_search.fit(X_train, y_train_target)
    
    # 최적의 모델 저장
    best_models[target] = random_search.best_estimator_
    
    # 해당 타겟에 대한 테스트 데이터
    y_test_target = y_test[target]
    
    # 테스트 데이터셋에 대한 예측 수행
    y_pred_target = best_models[target].predict(X_test)
    
    # R^2 값 계산
    r2_scores[target] = r2_score(y_test_target, y_pred_target)
    
    # RMSE 계산
    rmse_scores[target] = np.sqrt(mean_squared_error(y_test_target, y_pred_target))

# 결과 출력
print("R^2 scores for each dependent variable:")
for target, r2 in r2_scores.items():
    print(f"{target}: {r2}")

print("\nRMSE scores for each dependent variable:")
for target, rmse in rmse_scores.items():
    print(f"{target}: {rmse}")
for category in dependent_vars:
    mean_value = df[category].mean()
    print(category + ": " + str(mean_value))
import os
import joblib

# 저장할 디렉토리 지정 (현재 디렉토리를 사용)
save_dir = "./models/"

# 디렉토리가 존재하지 않으면 생성
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 훈련된 모델들을 .pkl 파일로 저장합니다.
for target, model in best_models.items():
    # 파일 이름에서 '/' 문자를 '_'로 교체하여 안전한 파일 이름을 생성합니다.
    safe_filename = target.replace('/', '_')
    
    # 파일 경로를 생성합니다.
    filename = os.path.join(save_dir, f'{safe_filename}_xgb_model.pkl')
    
    # 모델을 파일로 저장합니다.
    joblib.dump(model, filename)
    print(f'{target} 모델이 "{filename}" 파일로 저장되었습니다.')
import os
import joblib

def load_model(model_name):
    # 모델 파일 경로 지정
    filename = os.path.join('./models/', f'{model_name}_xgb_model.pkl')
    
    # 모델 파일 불러오기
    loaded_model = joblib.load(filename)
    return loaded_model

# 모델명을 지정
model_name = 'BOOK_ALBUM'

# 모델 불러오기
model = load_model(model_name)

# 모델이 사용하는 특성 이름 확인
booster = model.get_booster()
input_features = booster.feature_names
print("모델에 필요한 입력 컬럼:", input_features)

import pandas as pd

all_terminals = ['강남구', '강동구', '강북구', '강서구', '관악구', '광진구', '구로구', '금천구', '노원구', '도봉구', 
                     '동대문구', '동작구', '마포구', '서대문구', '서초구', '성동구', '성북구', '송파구', '양천구', 
                     '영등포구', '용산구', '은평구', '종로구', '중구', '중랑구']
years = [2021, 2022, 2023]
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

# 빈 데이터프레임 생성
df2 = pd.DataFrame(columns=['YEAR', 'WORK_GU'] + gu_columns)

for year in years:
    for terminal in all_terminals:
        # 'YEAR' 컬럼으로 먼저 필터링
        year_df = df[df['YEAR'] == year]

        # 'WORK_GU' 컬럼에 대해 현재 terminal을 필터링
        gu_df = year_df[year_df['WORK_GU'] == terminal][gu_columns]

        # 마지막 행을 리스트로 변환
        if not gu_df.empty:  # 데이터프레임이 비어 있지 않은 경우에만 진행
            gu_values = gu_df.iloc[-1].tolist()
            # 현재 연도와 구 이름을 추가하여 새 행 생성
            new_row = pd.DataFrame([{'YEAR': year, 'WORK_GU': terminal, **dict(zip(gu_columns, gu_values))}])
            # 새로운 행을 df2에 추가
            df2 = pd.concat([df2, new_row], ignore_index=True)

# df2를 확인
print(df2)


import os
import pandas as pd
import joblib

# 모델 파일을 불러오는 함수
def load_model(model_name):
    filename = os.path.join('./models/', f'{model_name}_xgb_model.pkl')
    loaded_model = joblib.load(filename)
    return loaded_model

# 원-핫 인코딩된 구 이름 컬럼을 생성하는 함수
def create_encoded_columns(terminal_name, all_terminals):
    # 모든 가능한 구 이름을 나열합니다.
    encoded_data = {f'WORK_GU_{terminal}': 1 if terminal == terminal_name else 0 for terminal in all_terminals}
    return encoded_data

# 사용자 입력
input_year = 2024
month = 11
day = 20
day_of = 1
holiday = 0
terminal_name = '강남구'

year = max(2021, min(input_year, 2023))

# 모델명을 지정합니다.
model_name = 'BOOK_ALBUM'

# 모든 가능한 구 이름을 나열합니다. 실제 모델에서 사용된 정확한 리스트를 사용해야 합니다.
all_terminals = ['강남구', '강동구', '강북구', '강서구', '관악구', '광진구', '구로구', '금천구', '노원구', '도봉구', 
                     '동대문구', '동작구', '마포구', '서대문구', '서초구', '성동구', '성북구', '송파구', '양천구', 
                     '영등포구', '용산구', '은평구', '종로구', '중구', '중랑구']

# 원-핫 인코딩된 구 이름 컬럼을 생성합니다.
encoded_columns = create_encoded_columns(terminal_name, all_terminals)

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


# 예측하려는 데이터 포인트를 준비합니다.
new_data = {
    'YEAR': [year],
    'MONTH': [month],
    'DAY': [day],
    'DAY_OF': [day_of],
    'HOLIDAY': [holiday]
}

gu_df = df2[(df2['YEAR'] == year) & (df2['WORK_GU'] == terminal_name)].iloc[0]

for column in gu_columns:
    new_data[column] = [gu_df[column]]
    
# 구별 데이터와 원-핫 인코딩된 컬럼을 new_data 딕셔너리에 추가합니다.
encoded_columns = create_encoded_columns(terminal_name, all_terminals)
new_data.update(encoded_columns)

# 데이터 프레임으로 변환합니다.
new_data_df = pd.DataFrame(new_data)

# 모델을 불러옵니다.
model = load_model(model_name)

# 예측을 수행합니다.
prediction = model.predict(new_data_df)

# 예측 결과를 출력합니다.
print(f'예측된 {model_name.replace("_", "/")} 물류량:', prediction)

df2.to_csv("work_info.csv", index=False)
