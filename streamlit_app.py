import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import time
from datetime import datetime

train_all = pd.read_csv("./train_all.csv")    
# 사이드바 제목
st.sidebar.title("Preprocessing")
# 메인 페이지 콘텐츠
st.title('Machine Learning Project 3조')
st.markdown('<p style="font-size:37px; color:white;">Store Sales - Time Series Forecasting</p>', unsafe_allow_html=True)
st.markdown('<a href="https://www.kaggle.com/competitions/store-sales-time-series-forecasting" target="_blank">Kaggle Link</a>', unsafe_allow_html=True)

main_title = st.empty()
main_sub1 = st.empty()
main_sub2 = st.empty()
main_sub3 = st.empty()
main_sub4 = st.empty()
main_sub5 = st.empty()



# 사이드바에 버튼 추가
if st.sidebar.button('Data Load : test/submission'):
    main_title.title("EDA")
    main_sub1.write("가정1: Transaction(거래량)이 많을수록 Sales(매출액)가 크다")
    main_sub2.write("가정2: family와 onpromotion이 transactions에 영향을 즐것이다.")
    main_sub3.write("가정3: 요일(weekend)와 날짜(월,일), 휴일(holiday)가 transactions에 영향을 줄것이다.")
    main_sub4.write("가정4: 오일가격(oil)/지진(2016년: 일시적)이 transaction에 영향을 주지 않는다.")
    main_sub5.write("가정5: transaction의 family 별 이동평균선 15일 30일을 통해 2018년 예측할 15일간의 transaction을 예측할수 있을것이다.")
    st.write("데이터를 가져옵니다.")

    st.write("test.csv")
    test = pd.read_csv("../data/test.csv")
    test['date'] = pd.to_datetime(test['date'])

    st.dataframe(test)

    st.write("submission")
    submission = pd.read_csv("../data/sample_submission.csv")
    st.dataframe(submission)

# 사이드바에 선택 메뉴 추가
option = st.sidebar.selectbox(
    'Preprocessing: EDA',
    ('EDA', 'train', 'transactions', 'store', 'holiday', 'oil')
)

if option == 'EDA':
    pass
    st.write('EDA Start..')
elif option == 'train':
    pass
    st.write('transactions 데이터를 분석합니다.')
elif option == 'transactions':
    pass
    st.write('transactions 데이터를 분석합니다.')
elif option == 'store':
    pass
    st.write('store 데이터를 분석합니다.')
elif option == 'holiday':
    pass
    st.write('holiday 데이터를 분석합니다.')
else:
    pass
    st.write('oil 데이터를 분석합니다.')

st.sidebar.title("Year Select")
option2 = st.sidebar.selectbox(
    'Year Select',
    ('Year', '2013', '2014', '2015', '2016', '2017')
)

if option2 == 'Year':
    pass
    st.write('Year Select..')
elif option2 == '2013':
    pass
    st.write('2013년 데이터를 분석합니다.')
elif option2 == '2014':
    pass
    st.write('2014년 데이터를 분석합니다.')
elif option2 == '2015':
    pass
    st.write('2015년 데이터를 분석합니다.')
elif option2 == '2016':
    pass
    st.write('2016년 데이터를 분석합니다.')
elif option2 == '2017':
    pass
    st.write('2017년 데이터를 분석합니다.')    
else:
    pass
    st.write('2018년 데이터를 분석합니다.')


st.sidebar.title("Modeling")

if st.sidebar.button('Modeling Ready!'):
    progress_bar = st.progress(0)    

    for i in range(10):

        time.sleep(1)  
        # 진행바 업데이트
        progress_bar.progress((i + 1) / 10)
        st.write(f"데이터 {i + 1} 처리 중...")

option3 = st.sidebar.selectbox(
    'Model Select',
    ('Model', 'XGBoost', 'Random Forest', 'Others...')
)

if option3 == 'Model': 
    pass   
elif option3 == 'XGBoost':
    pass
    st.write('XGBoost로 분석합니다.')
elif option3 == 'Random Forest':
# Plotly 그래프 생성
# 데이터 로드
    predict = pd.read_csv("./test_2017.csv")
    predict['date'] = pd.to_datetime(predict['date'])

    train_all = pd.read_csv("./train_all.csv")
    train_all['date'] = pd.to_datetime(train_all['date'])

    test_2017 = train_all[(train_all['date'] >= '2017-07-01') & (train_all['date'] <= '2017-08-31')]
    test_2017['date'] = pd.to_datetime(test_2017['date'])

    # 일별 판매량 계산
    date_sales_sum = predict.groupby('date')['sales'].sum().reset_index()
    test_2017_grouped = test_2017.groupby(test_2017['date'])['sales'].sum().reset_index()

    # store_nbr 별 판매량 계산
    store_sales_2017 = test_2017.groupby(['store_nbr', test_2017['date']])['sales'].sum().reset_index()

    # 첫 번째 그래프: 일별 판매량 라인 그래프
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=test_2017_grouped['date'], y=test_2017_grouped['sales'],
                            mode='lines+markers', name='Daily Sales August 2017',
                            line=dict(color='green')))
    fig1.add_trace(go.Scatter(
        x=date_sales_sum['date'],
        y=date_sales_sum['sales'],
        mode='lines+markers',
        name='Predicted Sales Data',
        line=dict(color='magenta')
    ))
    fig1.update_layout(
        title='Daily Sales in August 2017',
        xaxis_title='Date',
        yaxis_title='Total Sales',
        template='plotly_white'
    )


    # 스트림릿에서 그래프 표시
    st.title('Sales Analysis for August 2017')
    st.plotly_chart(fig1, use_container_width=True)
    
    st.write('Random Forest로 분석합니다.')
else:
    pass
    st.write('...모델로 분석합니다.')






# if option == 'Basic Data':
# # 진행바 생성
#     progress_bar = st.progress(0)        
#     # 날씨 데이터 처리 및 표시
#     for i in range(20):
#         # 작업 진행을 위한 시뮬레이션 딜레이
#         time.sleep(1)  # 실제 환경에서는 실제 크롤링 로직이 위치할 것
#         # 진행바 업데이트
#         progress_bar.progress((i + 1) / 20)
#         # 날씨 정보 출력 (예제 정보, 실제 데이터에 따라 조정 필요)
#         st.write(f"데이터 {i + 1} 처리 중...")
# # 다른 옵션에 대한 처리 (예시)
# elif option == 'Transactions':
#     st.write("뉴스 크롤링 결과를 여기에 표시합니다.")
# elif option == 'Holiday':
#     st.write("Telegram 메시징 기능을 여기에 구현합니다.")
# else:
#     pass