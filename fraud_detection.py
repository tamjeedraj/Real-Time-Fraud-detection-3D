import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import os

#1. Load Model (Simpler)
st.set_page_config(layout="wide", page_title="Real-Time Fraud Detection 3D App")

st.markdown("""
    <style>
    .css-1y4pm45l{padding-top: Orem;}
    .css-1lcbmhc{padding-top: Orem;}
    <style>
    """,
            unsafe_allow_html=True)

st.markdown("""
    <style>
    .block-container{padding-top: 1rem; padding-bottom: Orem; padding-left: 1rem; padding-right: 1rem;}
    h1{margin-top: 5rem; margin-bottom: 5rem; padding-top: 5rem;}
    <style>
    """,
            unsafe_allow_html=True)

st.markdown("""
    <style>
        .modebar{
            top: Opx !
important;
            right: auto !
            left: Opx !
important;
            opacity: 1;
            z-index: 1000;
        }
    </style>
    
    """,
            unsafe_allow_html=True
)


if not os.path.exists('fraud_detection_pipeline.pkl'):
    st.error("Error: 'fraud_detection_pipeline.pkl' file not found.")
    st.stop()

model = joblib.load('fraud_detection_pipeline.pkl')

#2. Sidebar for User Inputs
st.sidebar.title("Prediction Input")
#input1 = st.sidebar.number_input("Enter Input 1 (X-axis)", value = 0.0, format= "%.2f")
#input2 = st.sidebar.number_input("Enter Input 2 (Y-axis)", value = 0.0, format = "%.2f")

step = st.sidebar.number_input("Step (Time)", value=1.0, format = "%.2f")
typeval = st.sidebar.selectbox("Type of Transaction", ['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT'])
amount = st.sidebar.number_input("amount", value=1000.0 , format = "%.2f")
oldbalanceorg = st.sidebar.number_input("Old Balance (Sender)", value=10000.0, format = "%.2f")
newbalanceorig = st.sidebar.number_input("New Balance (Sender)", value = 9000.0, format = "%.2f")
oldbalancedest = st.sidebar.number_input("Old Balance Receiver", value=10000.0, format = "%.2f")
newbalancedest = st.sidebar.number_input("New Balance Receiver", value = 9000.0, format = "%.2f")

#input3 = st.sidebar.number_input("Enter Input 3 (amount)", value = 0.0, format = "%.2f")
#input4 = st.sidebar.number_input("Enter Input 3 (old balanceOrg)", value = 0.0, format = "%.2f")


#3. Main page
#st.title("Short 3D ML Visualizer")
st.header("Real-Time Fraud Detection 3D App")

if st.sidebar.button('Run Prediction', type="primary", use_container_width=True):
    input_data = pd.DataFrame({
        'step':[step],
        'type':[typeval],
        'amount':[amount],
        'oldbalanceOrg':[oldbalanceorg],
        'newbalanceOrig':[newbalanceorig],
        'oldbalanceDest':[oldbalancedest],
        'newbalanceDest' :[newbalancedest],
        'isFlaggedFraud': [0.0]
    })
    try:
        prediction = model.predict(input_data)[0]
        col1,col2 = st.columns([0.4,0.6])
        with col1:
            st.success(f'Predicted value: {prediction:.2f}')
            features_for_plot = [step, amount, prediction]
       # st.success(f'Predicted value: {prediction:.2f}')
      #  features_for_plot = [step, amount, prediction]

    except Exception as e:
        st.error(f"Error occurred during prediction: {e}")

#if st.sidebar.button('Run Prediction', type="primary", use_container_width=True):
#    try:

        #features = np.array([[input1, input2]])
        #prediction = model.predict(features)[0]
        #st.success(f'Predicted value: {prediction:.2f}')
    #except Exception as e:
        #st.error(f"Error:{e}")

#4 Create 3D Scatter Plot (for the single point)
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    #x=[input1], y=[input2],z=['prediction'], mode='markers', marker = dict(
    x=[step], y=[amount],z=[0], mode='markers', marker = dict(
        size = 12, color = 0, colorscale = [[0, 'blue'],[1, 'red']], showscale = True,
        colorbar = dict(
            title = "Fraud detection")),
    hoverinfo= 'x+y+z'
))

# Configure Plot Layout
fig.update_layout(title="Your New Prediction in 3D Space",scene=dict(xaxis =dict(title='Step (X-axis)'),
                                                                     yaxis = dict(title='Amount (Y-axis)'),
                                                                     zaxis = dict(title='Prediction (Z-axis)'),

),
                  margin=dict(l=0,r=0,b=0,t=0))

# Display the plot
try:
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Error occurred: {e}")

