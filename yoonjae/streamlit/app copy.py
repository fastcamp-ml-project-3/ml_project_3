import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import re
from prediction import get_predict

#from PIL import Image
#import requests
#from bokeh.plotting import figure
#from bokeh.models import HoverTool
#import joblib

# Get the current directory path
#current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the model from the pickle file
#model_path = os.path.join(current_dir, 'model.pkl')
#model = joblib.load(model_path)

# Load the scaler from the pickle file
#encoder_path = os.path.join(current_dir, 'encoder.pkl')
#encoder = joblib.load(encoder_path)

###################################################### Set Page Configurations ######################################################
st.set_page_config(page_title="[kaggle] Store Sales-Time Series Forecasting App", page_icon="fas fa-chart-line", layout="wide", initial_sidebar_state="auto")

# Loading GIF
#gif_url = "https://raw.githubusercontent.com/Gilbert-B/Forecasting-Sales/main/app/salesgif.gif"

###################################################### Set up sidebar ######################################################
st.sidebar.header('Navigation')
menu = ['Prediction', 'EDA', 'Lesson-Learned']
choice = st.sidebar.selectbox("Select an option", menu)


st.sidebar.header('The Required Information')
store_nbr = [str(i) for i in range(1, 55)]
selected_store_nbr = st.sidebar.selectbox("store_nbr", store_nbr)


family = ['AUTOMOTIVE', 'BABY CARE', 'BEAUTY', 'BEVERAGES', 'BOOKS',
    'BREAD/BAKERY', 'CELEBRATION', 'CLEANING', 'DAIRY', 'DELI', 'EGGS',
    'FROZEN FOODS', 'GROCERY I', 'GROCERY II', 'HARDWARE',
    'HOME AND KITCHEN I', 'HOME AND KITCHEN II', 'HOME APPLIANCES',
    'HOME CARE', 'LADIESWEAR', 'LAWN AND GARDEN', 'LINGERIE',
    'LIQUOR,WINE,BEER', 'MAGAZINES', 'MEATS', 'PERSONAL CARE',
    'PET SUPPLIES', 'PLAYERS AND ELECTRONICS', 'POULTRY',
    'PREPARED FOODS', 'PRODUCE', 'SCHOOL AND OFFICE SUPPLIES',
    'SEAFOOD']
selected_family = st.sidebar.selectbox("family", family)


# 초기 날짜 값 설정
initial_start_date = pd.to_datetime("2017-08-16")
initial_end_date = pd.to_datetime("2017-08-31")

# 날짜 입력 받기
selected_start_date = st.sidebar.date_input("Start Date", value=initial_start_date, key='start_date')
selected_end_date = st.sidebar.date_input("End Date", value=initial_end_date, key='end_date')

# onpromotion 입력 받기
selected_onpromotion = st.sidebar.number_input("How many products are on promotion?", min_value=0, step=1)

predicted_data = pd.DataFrame(columns=['Start Date', 'End Date', 'store_nbr', 'family', 'onpromotion', 'Predicted Sales'])



###################################################### section : Predction ######################################################
# 디버깅 용도
#st.write(f"Selected Family: {selected_family}")

# Choose section
if choice == 'Prediction':

    st.markdown("<h1 style='text-align: center;'>[kaggle] Store Sales Time Series Forecasting</h1>", unsafe_allow_html=True)
    #st.markdown("<p style='text-align: center;'>This is a Sales Forecasting App.</p>", unsafe_allow_html=True)

    
    if st.button('Predict'):
        if selected_start_date > selected_end_date:
            st.error("Start date should be earlier than the end date.")
        else:
            with st.spinner('Predicting sales... It will take about 2 minutes...'):
                sales_data = pd.DataFrame({
                    #'date': pd.date_range(start=selected_start_date, end=selected_end_date),
                    'store_nbr': [selected_store_nbr],
                    'family': [selected_family]
                    #'onpromotion': [selected_onpromotion] * len(pd.date_range(start=selected_start_date, end=selected_end_date))
                })
                try:
                    deepAR_origin, feature_importances = get_predict(sales_data)

                    # CSV 파일을 메모리 내에서 생성
                    csv_buffer = io.StringIO()
                    deepAR_origin.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    st.session_state.csv_data = csv_data

                    st.success(f"Predicted Sales is: # {deepAR_origin['predicted_sales']}")
                    
                    # 첫 10개 행을 테이블로 표시
                    st.write("Predicted Sales Data:")
                    st.dataframe(deepAR_origin)

                    # 파일 이름에 사용할 수 없는 문자 제거
                    cleaned_family = re.sub(r'[\\/:*?"<>|]', '', selected_family)
                    st.session_state.file_name = f'predictions_{int(selected_store_nbr):02d}_{cleaned_family}.csv'
                    st.success('File is ready for download!')
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    if 'csv_data' in st.session_state and 'file_name' in st.session_state:
        st.download_button(
            label="Download predictions as CSV",
            data=st.session_state.csv_data,
            file_name=st.session_state.file_name,
            mime='text/csv'
        )
        

        # Plot the feature importances
        plt.figure(figsize=(12, 8))
        plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importances from RandomForest')
        plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature at the top
        plt.show()

                        


    



###################################################### section : EDA ######################################################
elif choice == 'EDA':

    st.markdown("<h1 style='text-align: center;'>[kaggle] Store Sales Time Series Forecasting</h1>", unsafe_allow_html=True)
    #st.markdown("<p style='text-align: center;'>This is a Sales Forecasting App.</p>", unsafe_allow_html=True)

    
    
    if st.button('Explore train.csv'):

        with st.spinner('Drawing ... It will take about 2 minutes...'):
                # 디버깅 용도
                
                for i in range(1, 55):
                    # train 데이터 가져오기
                    train = pd.read_csv(r'C:\Users\user\upstage3-python\09.ML_project_team03\assets\data\train.csv')
                    
                    # 날짜 열을 datetime 형식으로 변환
                    train['date'] = pd.to_datetime(train['date'])
                    
                    # selected_store_nbr와 selected_family를 사용하여 데이터 필터링
                    filtered_data = train[(train['store_nbr'] == i) & (train['family'] == selected_family)]
                    
                    # 연도의 첫 날만 추출
                    filtered_data['year'] = filtered_data['date'].dt.year
                    year_start_dates = filtered_data.drop_duplicates('year')['date']
                    
                    # 데이터 시각화
                    plt.figure(figsize=(20, 3))
                    plt.bar(filtered_data['date'], filtered_data['sales'], color='blue')
                    plt.xlabel('Year')
                    plt.ylabel('Sales')
                    plt.title(f'Store Number {i} & Family {selected_family}')
                    
                    # 연도의 첫 날에 해당하는 날짜만 눈금으로 설정
                    plt.gca().set_xticks(year_start_dates)
                    plt.gca().set_xticklabels(year_start_dates.dt.year, rotation=90)
                    
                    # x축에 눈금 그리기
                    plt.grid(axis='x', which='both')
                    
                    st.pyplot(plt)
        
    #if st.button('transactions.csv'):

     #   with st.spinner('Drawing ... It will take about 2 minutes...'):    
      #      transactions = pd.read_csv(r'C:\Users\user\upstage3-python\09.ML_project_team03\assets\data\transactions.csv')
            
            
    
    if st.button('Missing Values: transactions & sales'):

        with st.spinner('Drawing ... It will take about 2 minutes...'):
            
            transactions = pd.read_csv(r'C:\Users\user\upstage3-python\09.ML_project_team03\assets\data\transactions.csv')
            temp = pd.read_csv(r'C:\Users\user\upstage3-python\09.ML_project_team03\assets\data\temp.csv')
            temp['date'] = pd.to_datetime(temp['date'])
            
        # Input form
        col1, col2 = st.columns(2)
    
        with col1:
            ################ 1
            transactions_dict = {}
            # 데이터 프레임 생성
            tran = pd.DataFrame(transactions[(transactions['store_nbr']==int(selected_store_nbr))])

            # 날짜를 datetime 형식으로 변환
            tran['date'] = pd.to_datetime(tran['date'])

            # 전체 날짜 범위 생성
            date_range = pd.date_range(start='2013-01-01', end='2017-08-15')

            # 데이터 프레임을 날짜 범위로 리인덱싱
            tran = tran.set_index('date').reindex(date_range).reset_index()

            # 결측치(NaN)를 원하는 값으로 채우기 (예: 0)
            tran['transactions'] = tran['transactions'].fillna(0)
            tran['store_nbr'] = tran['store_nbr'].fillna(int(selected_store_nbr))

            # 컬럼 이름 변경
            tran.columns = ['date', 'store_nbr','transactions']
    
            # 딕셔너리에 저장
            transactions_dict[f'transactions_{int(selected_store_nbr):02d}'] = tran          
            
            tran_missing = transactions_dict[f'transactions_{int(selected_store_nbr):02d}']        
                
            st.write(tran_missing)
            
            notransaction = len(tran_missing[(tran_missing['transactions']==0)&(tran_missing['date'].dt.year==2016)])
            
            st.success(f"The {int(selected_store_nbr)} store_nbr takes 0 transactions for {notransaction} days in 2016.")
        
        with col2:
            ################ 2
            # 특정 store family 비교
            sales0 =temp[(temp['store_nbr']==int(selected_store_nbr)) & (temp['family']==selected_family)][['date','sales','transactions']].reset_index(drop=True)
            df = pd.DataFrame(sales0)
            
            # 값이 0인 경우를 True로 변환
            df_zero = df[['sales', 'transactions']] == 0

            # y축 레이블로 사용할 연도 값을 추출
            df['year'] = df['date'].dt.year

            # 중복된 연도 레이블 제거
            df_zero.index = df['year']
            unique_years = df['year'].drop_duplicates().tolist()

            # 시각화
            plt.figure(figsize=(5, 10))
            ax = sns.heatmap(df_zero, cbar=False, cmap='viridis', yticklabels=df_zero.index)

            # 적절한 y축 눈금 설정
            step = max(1, len(df_zero) // 20)  # y축에 표시할 눈금의 간격 설정
            ax.set_yticks(range(0, len(df_zero), step))
            ax.set_yticklabels(df['date'].dt.date.iloc[::step], rotation=0)

            plt.title('Zero Values Heatmap')
            plt.xlabel('Columns')
            plt.ylabel('Year')
            plt.show()
            st.pyplot(plt)
            
            
            
            
            
    if st.button('Average Sales by Day_of_week'):

        with st.spinner('Drawing ... It will take about 2 minutes...'): 
            temp = pd.read_csv(r'C:\Users\user\upstage3-python\09.ML_project_team03\assets\data\temp2.csv')
            temp['date'] = pd.to_datetime(temp['date'])
            
            temp_2013 = temp[temp['date'].dt.year==2013]
            temp_2014 = temp[temp['date'].dt.year==2014]
            temp_2015 = temp[temp['date'].dt.year==2015]
            temp_2016 = temp[temp['date'].dt.year==2016]
            temp_2017 = temp[temp['date'].dt.year==2017]
                        
            for i in range(2013,2018):

                # 요일별 매출 집계
                sales_by_day = globals()[f'temp_{i}'].groupby('day_of_week')['sales'].mean().reset_index()

                # 요일 순서 정렬
                days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                sales_by_day['day_of_week'] = pd.Categorical(sales_by_day['day_of_week'], categories=days_order, ordered=True)
                sales_by_day = sales_by_day.sort_values('day_of_week')

                # 요일별 매출 시각화
                plt.figure(figsize=(20, 3))
                sns.barplot(data=sales_by_day, x='day_of_week', y='sales', palette='coolwarm')
                plt.title(f'Average Sales by Day of the Week ({i})')
                plt.xlabel('Day of the Week')
                plt.ylabel('Average Sales')
                plt.xticks(rotation=0)
                plt.show()
                st.pyplot(plt)
    
    if st.button('Corr : Number of Promotion-items & Sales'):

        with st.spinner('Drawing ... It will take about 2 minutes...'): 
            st.write(f"This data is for the years 2016 and 2017.")
              
            temp = pd.read_csv(r'C:\Users\user\upstage3-python\09.ML_project_team03\assets\data\temp2.csv')
            temp['date'] = pd.to_datetime(temp['date'])
            
            data = temp[(temp['year'].isin([2016,2017]))&(temp.store_nbr==int(selected_store_nbr))&(temp.family==selected_family)][['date', 'sales', 'onpromotion_lag1', 'transactions','day_of_week']]

            # 상관분석
            correlation = data['onpromotion_lag1'].corr(data['sales'])
            st.success(f'Correlation between promotion items and sales: {correlation}')
            
            # 시각화
            fig, ax1 = plt.subplots(figsize=(12, 6))

            color = 'tab:red'
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Promotion Items', color=color)
            ax1.plot(data['date'], data['onpromotion_lag1'], color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:blue'
            ax2.set_ylabel('Sales', color=color)
            ax2.plot(data['date'], data['sales'], color=color)
            ax2.tick_params(axis='y', labelcolor=color)

            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.title('Promotion Items and Sales Over Time')
            plt.show() 
            st.pyplot(fig)

    
    if st.button('outlier detection with onpormotion'):

        with st.spinner('Drawing ... It will take about 2 minutes...'):
            st.write(f"This red outlier is more than 4 standard deviations away from the mean.")
            
            temp = pd.read_csv(r'C:\Users\user\upstage3-python\09.ML_project_team03\assets\data\temp2.csv')
            temp['date'] = pd.to_datetime(temp['date'])
            # 필터링된 데이터 생성
            df = temp[(temp.store_nbr==int(selected_store_nbr))&(temp.family==selected_family)][['date', 'sales', 'onpromotion']]

            # 표준편차 방법으로 이상치 탐지 함수
            def find_outliers_std(df):
                outliers = pd.DataFrame()
                for column in df[['sales']]:  # sales 컬럼만 이상치 탐지
                    mean = df[column].mean()
                    std = df[column].std()
                    lower_bound = mean - 4 * std
                    upper_bound = mean + 4 * std
                    outliers[column] = (df[column] < lower_bound) | (df[column] > upper_bound)
                return outliers

            # 이상치 탐지
            outliers_std = find_outliers_std(df)

            # 그래프 생성
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

            # 첫 번째 그래프: sales 값
            bars = ax1.bar(df['date'], df['sales'], color=np.where(outliers_std['sales'], 'r', 'b'))
            ax1.set_ylabel('Sales')
            ax1.set_title(f'Store number {int(selected_store_nbr)} & {selected_family} - Sales')
            ax1.grid(True)

            # 두 번째 그래프: onpromotion 값
            ax2.bar(df['date'], df['onpromotion'], color='g')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('On Promotion')
            ax2.set_title(f'Store number {int(selected_store_nbr)} & {selected_family} - Sales')
            ax2.grid(True)

            # x축 레이블 회전
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            st.pyplot(plt)

            
    if st.button('lagging onpormotion'):

        with st.spinner('Drawing ... It will take about 2 minutes...'):
            temp = pd.read_csv(r'C:\Users\user\upstage3-python\09.ML_project_team03\assets\data\temp2.csv')
            temp['date'] = pd.to_datetime(temp['date'])
            # 필터링된 데이터 생성 & (temp['store_nbr'] == 25) 
            df = temp[(temp['year'].isin([2016, 2017])) &(temp.store_nbr==int(selected_store_nbr))& (temp['family'] == selected_family)][['date', 'sales', 'onpromotion']]

            # 데이터 시간순 정렬
            df = df.sort_values('date').reset_index(drop=True)

            # 상관 분석 함수
            def cross_correlation(df, lag):
                return df['onpromotion'].shift(lag).corr(df['sales'])

            # 다양한 시차에 대한 상관계수 계산
            lags = range(-10, 11)
            correlations = [cross_correlation(df, lag) for lag in lags]

            # 상관계수 시각화
            plt.figure(figsize=(10, 3))
            plt.plot(lags, correlations, marker='o')
            plt.xlabel('Lag (days)')
            plt.ylabel('Correlation')
            plt.title('Cross-correlation between Onpromotion and Sales')
            plt.grid(True)
            plt.axhline(0, color='black', linewidth=0.5)
            plt.show()
            st.pyplot(plt)
            
    if st.button('feature selection: corr heatmap'):

        with st.spinner('Drawing ... It will take about 2 minutes...'):
            st.write()
            temp = pd.read_csv(r'C:\Users\user\upstage3-python\09.ML_project_team03\assets\data\temp2.csv')
            temp['date'] = pd.to_datetime(temp['date'])
            
            sales53= temp[temp['store_nbr']==int(selected_store_nbr) & (temp['family']==selected_family)]
            heatmap_data = sales53.loc[:, ~sales53.columns.isin(['date', 'store_nbr', 'family','city','state','type','cluster','year','month','day'])]
            heatmap_data.loc[:, 'day_of_week'] = heatmap_data['day_of_week'].astype('category')
            heatmap_data = pd.get_dummies(heatmap_data, columns=['day_of_week'])
            heatmap_data.loc[:, 'season'] = heatmap_data['season'].astype('category')
            heatmap_data = pd.get_dummies(heatmap_data, columns=['season'])
            
            plt.figure(figsize=(25,15))
            #mask = np.triu(np.ones_like(merged_item_training_df.corr(), dtype=bool))
            sns.heatmap(heatmap_data.corr(), annot=True, cmap='BuPu', square=True, linewidths=1)
            st.pyplot(plt)
            
###################################################### section : Lesson-Learned ######################################################
elif choice == 'Lesson-Learned':
    st.markdown("<h1 style='text-align: center;'>[kaggle] Store Sales Time Series Forecasting</h1>", unsafe_allow_html=True)
    #st.markdown("<p style='text-align: center;'>This is a Sales Forecasting App.</p>", unsafe_allow_html=True)
    
    