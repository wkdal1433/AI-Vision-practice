import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# 데이터 불러오기
train = pd.read_csv('./train.csv')  # 실제 데이터 경로로 수정해야 함
test = pd.read_csv('./test.csv')    # 실제 데이터 경로로 수정해야 함
building_info = pd.read_csv('./building_info.csv')  # 실제 데이터 경로로 수정해야 함
submission = pd.read_csv('./sample_submission.csv')  # 실제 데이터 경로로 수정해야 함

# 데이터 전처리
# 날짜 데이터 처리
train['일시'] = pd.to_datetime(train['일시'])
test['일시'] = pd.to_datetime(test['일시'])

# 건물 정보와 훈련/테스트 데이터 병합
train = train.merge(building_info, on='건물번호', how='left')
test = test.merge(building_info, on='건물번호', how='left')

# 불필요한 컬럼 제거
train.drop(['건물번호', '연면적(m2)', 'ESS저장용량(kWh)', 'PCS용량(kW)'], axis=1, inplace=True)
test.drop(['건물번호', '연면적(m2)', 'ESS저장용량(kWh)', 'PCS용량(kW)'], axis=1, inplace=True)

# 비전기냉방설비운영, 태양광보유 컬럼의 'Y', 'N' 값을 1, 0으로 변환
train['비전기냉방설비운영'] = train['비전기냉방설비운영'].map({'Y': 1, 'N': 0})
train['태양광보유'] = train['태양광보유'].map({'Y': 1, 'N': 0})
test['비전기냉방설비운영'] = test['비전기냉방설비운영'].map({'Y': 1, 'N': 0})
test['태양광보유'] = test['태양광보유'].map({'Y': 1, 'N': 0})

# 훈련 데이터와 레이블 분리
X_train = train.drop('전력소비량(kWh)', axis=1)
y_train = train['전력소비량(kWh)']

# 테스트 데이터
X_test = test

# 훈련 데이터를 훈련 세트와 검증 세트로 분할
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# XGBoost 모델 학습
xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=42)
xgb_model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_val, y_val)], verbose=100)

# LightGBM 모델 학습
lgb_model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=42)
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=100)

# 검증 세트 예측
val_predictions_xgb = xgb_model.predict(X_val)
val_predictions_lgb = lgb_model.predict(X_val)

# 검증 세트 평가 (Root Mean Squared Error)
val_rmse_xgb = mean_squared_error(y_val, val_predictions_xgb, squared=False)
val_rmse_lgb = mean_squared_error(y_val, val_predictions_lgb, squared=False)

print(f"XGBoost Validation RMSE: {val_rmse_xgb}")
print(f"LightGBM Validation RMSE: {val_rmse_lgb}")

# 두 모델의 예측값 평균
val_predictions_avg = (val_predictions_xgb + val_predictions_lgb) / 2

# 테스트 데이터 예측
test_predictions_xgb = xgb_model.predict(X_test)
test_predictions_lgb = lgb_model.predict(X_test)

# 두 모델의 예측값 평균
test_predictions_avg = (test_predictions_xgb + test_predictions_lgb) / 2

# 제출 파일 생성
submission['answer'] = test_predictions_avg
submission.to_csv('submission.csv', index=False)  # 실제 데이터 경로로 수정해야 함