import pandas as pd
import numpy as np
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import linregress

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.common import ListDataset
from gluonts.torch import DeepAREstimator
from gluonts.dataset.split import split
import matplotlib.pyplot as plt



def get_predict(sales_data): 
        if sales_data.empty:
            raise ValueError("No sales data provided.")
        
          
        # 기본 데이터 가져오기
        temp = pd.read_csv(r'C:\Users\user\upstage3-python\09.ML_project_team03\assets\data\temp.csv')
        test = pd.read_csv(r'C:\Users\user\upstage3-python\09.ML_project_team03\assets\data\test.csv')
        
        # 지진 이상치 대체하기
        get_normal_sales_during_earthquake(temp)
        
        
        # 모델 적용하기               
        store = int(sales_data['store_nbr'].iloc[0])
        family = sales_data['family'].iloc[0]
        
        print(f"###################################### start store: {store}, family: {family}######################################")
        ###################################### test data set : gluonts deepAR ######################################
        
        #최종 test set을 위한 셋팅
        deepAR = pd.DataFrame()
        deepAR['date'] = pd.date_range(start='2017-08-16', end='2017-08-31')
        deepAR['store_nbr'] = store
        deepAR['family'] = family

        test['date'] = pd.to_datetime(test['date'])
        filtered_test = test[(test['store_nbr'] == store) & (test['family'] == family)]
        deepAR = pd.merge(deepAR, filtered_test[['date', 'onpromotion']], on='date', how='left')

        # gluonts 돌리기 위한 데이터셋
        df = temp[(temp['date']>='2016-01-01')&(temp['date']<='2017-08-15')&(temp['store_nbr']==store)&(temp['family']==family)]
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        ## sales 컬럼 생성을 위한 예측
        forecast = get_gluonts(df, 'sales')

        deepAR['sales']  = forecast[0].mean

        ## transactions 컬럼 생성을 위한 예측
        # Prepare ListDataset
        forecast = get_gluonts(df, 'transactions')
        
        deepAR['transactions']  = forecast[0].mean

        # rolling: not to make NaN, concat with dataset before two weeks
        deepAR_before= temp[(temp['date']>='2017-08-01')&(temp['date']<='2017-08-15')&(temp['store_nbr']==store)&(temp['family']==family)][['date', 'store_nbr', 'family', 'sales', 'transactions', 'onpromotion']]

        deepAR_after= pd.concat([deepAR_before, deepAR], axis=0)

        deepAR_after['date'] = pd.to_datetime(deepAR_after['date'])

        # rolling
        get_rolling(deepAR_after)

        # cut date
        deepAR = deepAR_after[deepAR_after['date']>='2017-08-16']

        # day_of_week
        pd.options.mode.copy_on_write = True
        deepAR['date'] = pd.to_datetime(deepAR['date'])
        deepAR.loc[:, 'day_of_week'] = deepAR['date'].dt.day_name()
        deepAR.loc[:, 'day_of_week'] = deepAR['day_of_week'].astype('category')
        deepAR = pd.get_dummies(deepAR, columns=['day_of_week'])

        # onpromotion_lag
        for lag in [1]:
            deepAR.loc[:, f'onpromotion_lag{lag}'] = deepAR['onpromotion'].shift(lag).fillna(0)
       
        # drop the useless columns       
        deepAR_origin = deepAR.copy()
        deepAR.drop(columns=['onpromotion','date', 'store_nbr', 'family', 'sales'], inplace=True)
        
        # order
        columns_order = ['onpromotion_lag1', 'transactions', 'slope7', 'std7', 'mean7', 'skew7', 'kurt7', 'min7', 'max7', 'slope14', 'std14', 'mean14', 'skew14', 'kurt14', 'min14', 'max14','day_of_week_Wednesday', 'day_of_week_Thursday', 'day_of_week_Friday', 'day_of_week_Saturday', 'day_of_week_Sunday', 'day_of_week_Monday', 'day_of_week_Tuesday']
        deepAR = deepAR.reindex(columns=columns_order)
        
        
        ################################### final : RandomForestRegressor ######################################      

        
        # 저장된 모델 불러오기
        if store==7:
            
            # 파일 이름에 사용할 수 없는 문자 제거
            cleaned_family = re.sub(r'[\\/:*?"<>|]', '', family)

            with open(f'rf_model_{store:02d}_{cleaned_family}.pkl', 'rb') as f:
                rf_model = pickle.load(f)
            # 예측
            pickle_model_pred = rf_model.predict(deepAR)
            
            # Add predictions to test_for_pred dataframe
            deepAR_origin['predicted_sales'] = pickle_model_pred

        else:
            data = temp[(temp['year'].isin([2016, 2017])) & (temp.store_nbr == store) & (temp.family == family)][['date', 'sales', 'onpromotion_lag1', 'transactions', 'day_of_week']]
        
            # day_of_week
            data['day_of_week'] = data['day_of_week'].astype('category')
            data = pd.get_dummies(data, columns=['day_of_week'])
            
            # rolling
            get_rolling(data)
            
            data.dropna(inplace=True)
            
            # Features 및 target 설정
            X = data[['onpromotion_lag1', 'transactions', 'slope7', 'std7', 'mean7', 'skew7', 'kurt7', 'min7', 'max7', 'slope14', 'std14', 'mean14', 'skew14', 'kurt14', 'min14', 'max14','day_of_week_Wednesday', 'day_of_week_Thursday', 'day_of_week_Friday', 'day_of_week_Saturday', 'day_of_week_Sunday', 'day_of_week_Monday', 'day_of_week_Tuesday']]
            y = data['sales']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            rf = RandomForestRegressor(n_estimators=1, random_state=42)

            rf.fit(X_train, y_train)
            
            # Predict
            test_pred = rf.predict(deepAR)

            # Add predictions to test_for_pred dataframe
            deepAR_origin['predicted_sales'] = test_pred




        return deepAR_origin 
        



def get_normal_sales_during_earthquake(temp):
            
        temp['date'] = pd.to_datetime(temp['date'])

        # 필터링할 날짜 범위 설정
        target_start_date = "2016-04-16"
        target_end_date = "2016-04-29"
        source_start_date = "2016-04-09"
        source_end_date = "2016-04-15"

        # 타겟 데이터와 소스 데이터 필터링
        target_temp = temp[(temp['date'] >= target_start_date) & (temp['date'] <= target_end_date)]
        source_temp = temp[(temp['date'] >= source_start_date) & (temp['date'] <= source_end_date)]

        # 데이터 대체 함수
        def replace_sales(target_temp, source_temp):
            new_sales = []
            
            for index, row in target_temp.iterrows():
                matching_source_temp = source_temp[
                    (source_temp['store_nbr'] == row['store_nbr']) &
                    (source_temp['family'] == row['family']) &
                    (source_temp['day_of_week'] == row['day_of_week'])
                ]
                
                if not matching_source_temp.empty:
                    source_row = matching_source_temp.iloc[0]
                    
                    # 트랜잭션 비례로 sales 계산
                    transaction_ratio = row['transactions'] / source_row['transactions']
                    adjusted_sales = source_row['sales'] * transaction_ratio
                    new_sales.append(adjusted_sales)
                else:
                    new_sales.append(row['sales'])  # 매칭되는 소스 데이터가 없을 경우 원래 sales 값을 사용
            
            target_temp['sales'] = new_sales
            return target_temp

        # 타겟 데이터를 대체 데이터로 변경
        temp.loc[(temp['date'] >= target_start_date) & (temp['date'] <= target_end_date), 'sales'] = replace_sales(target_temp, source_temp)['sales']


# gluonts deepAR 함수 정의
def get_gluonts(df, target_column):
        if df.empty:
            raise ValueError("The dataframe is empty. Please provide a non-empty dataframe.")
    
        # Split the data for training and testing
        training_data = ListDataset(
            [{"start": df.index[0],
            "target": df[target_column].values}],
            freq="D"
        )

        test_data = ListDataset(
            [{"start": df.index[0], 
            "target": df[target_column].values}],
            freq="D"
        )

        # Train the model and make predictions
        model = DeepAREstimator(
            prediction_length=16, freq="D", trainer_kwargs={"max_epochs": 1}
        ).train(training_data)

        forecasts = list(model.predict(test_data))
          
        return forecasts
    


# get_rolling 함수 정의
def get_rolling(df):
    df['slope7'] = df['sales'].rolling(7).apply(get_slope, raw=True)
    df['std7'] = df['sales'].rolling(7).std()
    df['mean7'] = df['sales'].rolling(7).mean()
    df['skew7'] = df['sales'].rolling(7).skew()
    df['kurt7'] = df['sales'].rolling(7).kurt()
    df['min7'] = df['sales'].rolling(7).min()
    df['max7'] = df['sales'].rolling(7).max()

    df['slope14'] = df['sales'].rolling(14).apply(get_slope, raw=True)
    df['std14'] = df['sales'].rolling(14).std()
    df['mean14'] = df['sales'].rolling(14).mean()
    df['skew14'] = df['sales'].rolling(14).skew()
    df['kurt14'] = df['sales'].rolling(14).kurt()
    df['min14'] = df['sales'].rolling(14).min()
    df['max14'] = df['sales'].rolling(14).max()
    
# get_slope 함수 정의
def get_slope(array):
    y = np.array(array)
    x = np.arange(len(y))
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return slope