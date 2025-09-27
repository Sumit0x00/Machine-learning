import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(layout='wide', page_title='Regression')

url = 'https://raw.githubusercontent.com/campusx-official/100-days-of-machine-learning/refs/heads/main/day49-regression-metrics/placement.csv'
df = pd.read_csv(url)
x = df.iloc[:,0:1]
y = df.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
lr = LinearRegression()
lr.fit(x, y)
y_pred = lr.predict(x_test)

st.title('Linear Regression - CGPA to Package Prediction')
st.sidebar.title('CGPA-Package Regression')

# Input for CGPA
cgpa_input = st.sidebar.number_input(
    'Enter CGPA:', 
    min_value=0.0, 
    max_value=10.0, 
    value=7.5, 
    step=0.1,
    format="%.2f"
)

# Metrics selection
options = st.sidebar.selectbox('Select Metrics', ['MAE', 'MSE', 'RMSE', 'R2SCORE'])

# Make prediction if CGPA is provided
predicted_package = None
if cgpa_input:
    predicted_package = lr.predict([[cgpa_input]])[0]
    st.sidebar.success(f"Predicted Package: ₹{predicted_package:.2f} LPA")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Performance & Prediction")
    
    # Show prediction
    if predicted_package is not None:
        st.metric(
            label="Predicted Package", 
            value=f"{predicted_package:.2f} LPA",
            help=f"For CGPA: {cgpa_input}"
        )
    else:
        st.metric("Predicted Package", "Enter CGPA", help="Enter CGPA in sidebar")
    
    # Show selected metric
    if options == 'MAE':
        mae_value = mean_absolute_error(y_test, y_pred)
        st.metric('Mean Absolute Error (MAE)', f"{mae_value:.4f}")
        st.info("MAE: Average absolute difference between actual and predicted values")
        
    elif options == 'MSE':
        mse_value = mean_squared_error(y_test, y_pred)
        st.metric('Mean Squared Error (MSE)', f"{mse_value:.4f}")
        st.info("MSE: Average squared difference between actual and predicted values")
        
    elif options == 'RMSE':
        rmse_value = np.sqrt(mean_squared_error(y_test, y_pred))
        st.metric('Root Mean Squared Error (RMSE)', f"{rmse_value:.4f}")
        st.info("RMSE: Square root of MSE, in same units as target variable")
        
    elif options == 'R2SCORE':
        r2_value = r2_score(y_test, y_pred)
        st.metric('R² Score', f"{r2_value:.4f}")
        st.info("R²: Proportion of variance explained by the model (closer to 1 is better)")
    

    
    # Additional metrics summary
    with st.expander("All Metrics Summary"):
        st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred):.4f}")
        st.write(f"**MSE:** {mean_squared_error(y_test, y_pred):.4f}")
        st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
        st.write(f"**R² Score:** {r2_score(y_test, y_pred):.4f}")

with col2:
    st.subheader("Data Visualization")
    
    # Create scatter plot
    fig = px.scatter(
        df,
        x="cgpa",
        y="package",
        hover_data=["cgpa", "package"],
        title="CGPA vs Package Distribution"
    )
    
    # Add regression line
    x_range = np.linspace(df['cgpa'].min(), df['cgpa'].max(), 100)
    y_line = lr.predict(x_range.reshape(-1, 1))
    
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_line,
            mode='lines',
            name='Regression Line',
            line=dict(color='red', width=3)
        )
    )
    
    # Highlight the prediction point if CGPA is entered
    if predicted_package is not None:
        fig.add_trace(
            go.Scatter(
                x=[cgpa_input],
                y=[predicted_package],
                mode='markers',
                name=f'Prediction (CGPA: {cgpa_input})',
                marker=dict(color='green', size=15, symbol='star'),
                hovertemplate=f'<b>Prediction</b><br>CGPA: {cgpa_input}<br>Package: {predicted_package:.2f} LPA<extra></extra>'
            )
        )
    
    fig.update_layout(
        xaxis_title="CGPA",
        yaxis_title="Package (LPA)",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data summary
    st.subheader("Dataset Summary")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Total Records", len(df))
        st.metric("Avg CGPA", f"{df['cgpa'].mean():.2f}")
    with col_b:
        st.metric("Avg Package", f"{df['package'].mean():.2f} LPA")
        st.metric("CGPA Range", f"{df['cgpa'].min():.1f} - {df['cgpa'].max():.1f}")

# Additional prediction examples
st.subheader("Quick Prediction Examples")
example_col1, example_col2, example_col3 = st.columns(3)

with example_col1:
    if st.button("CGPA: 6.0"):
        pred = lr.predict([[6.0]])[0]
        st.write(f"Package: ₹{pred:.2f} LPA")

with example_col2:
    if st.button("CGPA: 8.0"):
        pred = lr.predict([[8.0]])[0]
        st.write(f"Package: ₹{pred:.2f} LPA")

with example_col3:
    if st.button("CGPA: 9.5"):
        pred = lr.predict([[9.5]])[0]
        st.write(f"Package: ₹{pred:.2f} LPA")