import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import datetime

# Load dataset
def load_data():
    df = pd.read_parquet("partc.parquet")  # Ensure this file is uploaded
    st.write("Dataset Columns:", df.columns.tolist())  # Debugging output
    return df

# Forecasting function
def forecast_stock(df, company):
    company_data = df[df['name'] == company]
    
    X = company_data[['EMA_10', 'MACD', 'ATR_14', 'Williams_%R']]  # Features
    y = company_data['close']  # Using 'close' price as target variable
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = ExtraTreesRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    error = mean_absolute_error(y_test, predictions)
    accuracy = r2_score(y_test, predictions) * 100
    
    # Forecast for next 10 days
    last_known_date = pd.to_datetime(company_data['date'].iloc[-1])
    future_dates = [last_known_date + datetime.timedelta(days=i) for i in range(1, 11)]
    future_predictions = model.predict(X.tail(10))
    
    return predictions, y_test, error, accuracy, future_dates, future_predictions

# Streamlit App
st.title("Stock Forecasting with Extra Trees Regressor")

# Load data
df = load_data()
if 'name' not in df.columns:
    st.error("Error: 'name' column not found in the dataset. Please check the column names.")
    st.stop()

companies = df['name'].unique()
selected_company = st.selectbox("Select a Company:", companies)

if selected_company:
    predictions, actual, error, accuracy, future_dates, future_predictions = forecast_stock(df, selected_company)
    
    st.write(f"Mean Absolute Error: {error:.2f}")
    st.write(f"Accuracy Score: {accuracy:.2f}%")
    
    # Interactive Graphs
    fig1 = px.line(x=range(len(predictions)), y=predictions, title="Predicted Stock Prices", labels={'x': 'Days', 'y': 'Price'})
    st.plotly_chart(fig1)
    
    fig2 = px.line(x=range(len(actual)), y=actual, title="Actual Stock Prices", labels={'x': 'Days', 'y': 'Price'})
    st.plotly_chart(fig2)
    
    combined_df = pd.DataFrame({"Actual": actual.values, "Predicted": predictions})
    fig3 = px.line(combined_df, title="Actual vs Predicted Prices")
    st.plotly_chart(fig3)
    
    fig4 = px.scatter(x=actual, y=predictions, title="Actual vs Predicted Scatter Plot", labels={'x': 'Actual', 'y': 'Predicted'})
    st.plotly_chart(fig4)
    
    future_df = pd.DataFrame({"Date": future_dates, "Forecasted Price": future_predictions})
    fig5 = px.line(future_df, x='Date', y='Forecasted Price', title="10-Day Forecasted Stock Prices")
    st.plotly_chart(fig5)
