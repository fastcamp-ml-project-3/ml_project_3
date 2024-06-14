import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import requests
from bokeh.plotting import figure
from bokeh.models import HoverTool
import joblib
import os
import io
import re
import pickle
from ml_project_3.yoonjae.streamlit.streamlit.prediction import get_predict

# Get the current directory path
current_dir = os.path.dirname(os.path.abspath(__file__))

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
                    deepAR_origin = get_predict(sales_data)

                    # CSV 파일을 메모리 내에서 생성
                    csv_buffer = io.StringIO()
                    deepAR_origin.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    st.session_state.csv_data = csv_data

                    st.success(f"Predicted Sales is: #{deepAR_origin['predicted_sales']}")
                    
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
                        


    



###################################################### section : EDA ######################################################
elif choice == 'EDA':

    st.markdown("<h1 style='text-align: center;'>[kaggle] Store Sales Time Series Forecasting</h1>", unsafe_allow_html=True)
    #st.markdown("<p style='text-align: center;'>This is a Sales Forecasting App.</p>", unsafe_allow_html=True)
    
 
        
    
###################################################### section : Lesson-Learned ######################################################
elif choice == 'Lesson-Learned':
    st.markdown("<h1 style='text-align: center;'>[kaggle] Store Sales Time Series Forecasting</h1>", unsafe_allow_html=True)
    #st.markdown("<p style='text-align: center;'>This is a Sales Forecasting App.</p>", unsafe_allow_html=True)
    
    