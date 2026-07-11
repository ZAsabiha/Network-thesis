import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import os

st.set_page_config(page_title="SDN Traffic Monitor", layout="wide")

st.title("🌐 SDN Traffic Monitoring Dashboard")

# Sidebar
with st.sidebar:
    st.header("Controls")
    refresh = st.button("🔄 Refresh Data")
    auto_refresh = st.checkbox("Auto Refresh (5s)", value=True)
    
    st.divider()
    
    st.subheader("Traffic Generator")
    traffic_type = st.selectbox("Traffic Type", ["TCP", "UDP", "Mixed"])
    bandwidth = st.selectbox("Bandwidth", ["5M", "10M", "20M", "50M", "100M"])
    duration = st.number_input("Duration (seconds)", min_value=5, max_value=300, value=30)
    
    if st.button("▶ Generate Traffic", type="primary"):
        st.success(f"Generating {traffic_type} traffic at {bandwidth} for {duration}s")
        # Add traffic generation logic here

# Main content
col1, col2, col3, col4 = st.columns(4)

try:
    df = pd.read_csv('traffic_dataset.csv')
    
    # Stats
    with col1:
        st.metric("Current TX Speed", f"{df['tx_kbps'].iloc[-1]:.0f} Kbps" if len(df) > 0 else "0 Kbps")
    with col2:
        st.metric("Current RX Speed", f"{df['rx_kbps'].iloc[-1]:.0f} Kbps" if len(df) > 0 else "0 Kbps")
    with col3:
        congestion_rate = (df['is_congested'].sum() / len(df)) * 100 if len(df) > 0 else 0
        st.metric("Congestion Rate", f"{congestion_rate:.1f}%")
    with col4:
        st.metric("Total Samples", len(df))
    
    # Traffic Chart
    st.subheader("📊 Traffic Over Time")
    fig = px.line(df.tail(100), x='timestamp', y=['tx_kbps', 'rx_kbps'],
                  title="TX/RX Speed Over Time")
    fig.update_layout(legend_title="Speed")
    st.plotly_chart(fig, use_container_width=True)
    
    # Congestion Chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔴 Congestion Status")
        fig = px.pie(df.tail(100), names=df['is_congested'].map({0: 'Normal', 1: 'Congested'}),
                     title="Congestion Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Switch Statistics")
        switch_stats = df.groupby('switch_id').agg({
            'tx_kbps': 'mean',
            'rx_kbps': 'mean',
            'is_congested': 'sum'
        }).reset_index()
        switch_stats.columns = ['Switch', 'Avg TX', 'Avg RX', 'Congestion Count']
        st.dataframe(switch_stats, use_container_width=True)
    
    # Raw data
    with st.expander("📋 Raw Data"):
        st.dataframe(df.tail(20))
        
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Make sure traffic_dataset.csv exists. Generate traffic first!")

# Auto refresh
if auto_refresh:
    time.sleep(5)
    st.rerun()