# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# import pickle
# from datetime import timedelta

# st.set_page_config(page_title="Demand Forecasting - Model Comparison", layout="wide")

# # === File paths ===
# DATASET_2025_PATH = r'C:/Users/COMTECH COMPUTER/Desktop/AI Based Load Forecasting/prediction_data_12_2024.csv'

# MODEL_PATHS = {
#     "Linear Regression": r"C:/Users/COMTECH COMPUTER/Desktop/AI Based Load Forecasting/LR.pkl",
#     "Ridge Regression": r"C:/Users/COMTECH COMPUTER/Desktop/AI Based Load Forecasting/Ridge.pkl",
#     "Lasso Regression": r"C:/Users/COMTECH COMPUTER/Desktop/AI Based Load Forecasting/Lasso.pkl",
#     "ElasticNet": r"C:/Users/COMTECH COMPUTER/Desktop/AI Based Load Forecasting/ElasticNet.pkl",
#     "K-Nearest Neighbors": r"C:/Users/COMTECH COMPUTER/Desktop/AI Based Load Forecasting/KNN.pkl",
#     "Random Forest": r"C:/Users/COMTECH COMPUTER/Desktop/AI Based Load Forecasting/RF01.pkl",
#     "AdaBoost": r"C:/Users/COMTECH COMPUTER/Desktop/AI Based Load Forecasting/AdaBoost.pkl",
#     "Gradient Boosting": r"C:/Users/COMTECH COMPUTER/Desktop/AI Based Load Forecasting/GradientBoost.pkl",
#     "XGBoost": r"C:/Users/COMTECH COMPUTER/Desktop/AI Based Load Forecasting/rs_xgb.pkl",
#     "LightGBM": r"C:/Users/COMTECH COMPUTER/Desktop/AI Based Load Forecasting/LightGradientBoost.pkl",
#     "CatBoost": r"C:/Users/COMTECH COMPUTER/Desktop/AI Based Load Forecasting/CatGradientBoost.pkl",
#     "Voting": r"C:/Users/COMTECH COMPUTER/Desktop/AI Based Load Forecasting/Voting.pkl",
#     "Stacking": r"C:/Users/COMTECH COMPUTER/Desktop/AI Based Load Forecasting/Stacking.pkl",
#     "Blanding": r"C:/Users/COMTECH COMPUTER/Desktop/AI Based Load Forecasting/Blending.pkl"
# }

# MODEL_PATHS["Stacking"] = {
#     "base": r"C:/Users/COMTECH COMPUTER/Desktop/AI Based Load Forecasting/Base_Models_for_Stacking.pkl",
#     "meta": r"C:/Users/COMTECH COMPUTER/Desktop/AI Based Load Forecasting/Meta_Model_for_Stacking.pkl"
# }

# FEATURE_COLUMNS = [
#     'embedded_wind_generation', 'embedded_solar_generation', 'wind + solar generation',
#     'year', 'weekend', 'hour_of_day_x', 'hour_of_day_y', 'week_x', 'week_y',
#     'month_x', 'month_y', 'demand_diff', 'wind_diff', 'solar_diff',
#     'rolling_mean_24', 'rolling_std_24', 'rolling_mean_48', 'rolling_std_48',
#     'load_lag_48', 'timeofday_Evening', 'timeofday_Morning',
#     'timeofday_Night', 'season_Spring', 'season_Summer', 'season_Winter'
# ]
# TARGET_COLUMN = 'nd'

# @st.cache_data
# def load_data():
#     df = pd.read_csv(DATASET_2025_PATH)
#     df['settlement_date'] = pd.to_datetime(df['settlement_date'])
#     df.set_index('settlement_date', inplace=True)
#     df[TARGET_COLUMN].fillna(df[TARGET_COLUMN].mean(), inplace=True)
#     df.dropna(subset=FEATURE_COLUMNS, inplace=True)
#     return df

# @st.cache_resource
# def load_model(path):
#     with open(path, 'rb') as f:
#         return pickle.load(f)

# def calculate_metrics(y_true, y_pred):
#     mae = np.mean(np.abs(y_true - y_pred))
#     rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
#     r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
#     return mae, rmse, r2

# def plot_demand_forecast(y_true, y_pred, model_name, start, end):
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=y_true.index, y=y_true, mode='lines+markers', name='Actual',
#                              line=dict(color='royalblue', width=2)))
#     fig.add_trace(go.Scatter(x=y_pred.index, y=y_pred, mode='lines+markers', name='Predicted',
#                              line=dict(color='tomato', width=2, dash='dash')))

#     fig.update_layout(
#         title=f"{model_name} Forecast: {start.date()} to {end.date()}",
#         xaxis_title="Time",
#         yaxis_title="Electricity Demand (MW)",
#         legend=dict(x=0.01, y=0.99),
#         template="plotly_white",
#         margin=dict(t=60, b=40),
#         height=500
#     )
#     return fig

# def predict_with_stacking(X_input):
#     # Load base and meta models
#     with open(MODEL_PATHS["Stacking"]["base"], 'rb') as f:
#         base_estimators = pickle.load(f)
#     with open(MODEL_PATHS["Stacking"]["meta"], 'rb') as f:
#         meta_model = pickle.load(f)

#     # Generate predictions from each base model
#     meta_features = np.column_stack([
#         model.predict(X_input) for name, model in base_estimators
#     ])

#     # Final prediction from meta-model
#     final_preds = meta_model.predict(meta_features)
#     return final_preds


# def main():
#     st.title("⚡ Electricity Demand Forecasting")
#     st.markdown("Visualize and compare the performance of multiple forecasting models using real-world electricity demand data.")
#     st.markdown("---")

#     data = load_data()
#     X = data[FEATURE_COLUMNS]
#     y = data[TARGET_COLUMN]

#     st.sidebar.header("⚙️ Options")
#     selected_model_name = st.sidebar.selectbox("Select Model", list(MODEL_PATHS.keys()))
#     model_path = MODEL_PATHS[selected_model_name]

#     st.sidebar.markdown("---")
#     st.sidebar.header("📅 Select Date Range")
#     min_date = data.index.min().date()
#     max_date = data.index.max().date()

#     date_range = st.sidebar.date_input(
#         "Forecast Date Range:",
#         value=[min_date, min_date + timedelta(days=1)],
#         min_value=min_date,
#         max_value=max_date
#     )

#     if not date_range or len(date_range) != 2:
#         st.info("ℹ️ Please select a valid start and end date.")
#         return

#     start_time = pd.Timestamp(date_range[0])
#     end_time = pd.Timestamp(date_range[1]) + pd.Timedelta(hours=23, minutes=30)

#     X_test = X.loc[start_time:end_time]
#     y_test = y.loc[start_time:end_time]

#     if X_test.empty or y_test.empty:
#         st.error("❌ No data available for selected range.")
#         return
    


#     from sklearn.preprocessing import StandardScaler

#     if selected_model_name == "K-Nearest Neighbors":
#         # Scale both features and target
#         scaler_X = StandardScaler()
#         scaler_y = StandardScaler()

#         X_scaled = scaler_X.fit_transform(X_test)
#         y_scaled = scaler_y.fit_transform(y_test.values.reshape(-1, 1))

#         # Load model and predict
#         model = load_model(model_path)
#         y_pred_scaled = model.predict(X_scaled)

#         # Inverse scale the predictions
#         y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
#         predictions = pd.Series(y_pred, index=y_test.index)

#     elif selected_model_name == "Stacking":
#         predictions = predict_with_stacking(X_test)


#     # elif selected_model_name == "Blending":


#     else:
#         model = load_model(model_path)
#         y_pred = model.predict(X_test)
#         predictions = pd.Series(y_pred.flatten(), index=y_test.index)


    
#     pred_series = pd.Series(np.array(predictions).flatten(), index=X_test.index)

#     y_true = y_test.loc[pred_series.index]
#     y_pred = pred_series.loc[y_test.index]

#     mae, rmse, r2 = calculate_metrics(y_true.values, y_pred.values)

#     st.subheader(f"📉 Model Performance: {selected_model_name}")
#     col1, col2, col3 = st.columns(3)
#     col1.metric("🔹 MAE", f"{mae:.2f}")
#     col2.metric("🔸 RMSE", f"{rmse:.2f}")
#     col3.metric("📈 R²", f"{r2:.4f}")

#     with st.expander("🔍 Forecast Visualization", expanded=True):
#         fig = plot_demand_forecast(y_true, y_pred, selected_model_name, start_time, end_time)
#         st.plotly_chart(fig, use_container_width=True)

#     st.markdown("---")
#     st.caption("Developed as part of Final Year Project | AI-Based Load Forecasting")

# if __name__ == "__main__":
#     main()





import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle
from datetime import timedelta
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Demand Forecasting - Model Comparison", layout="wide")

# === File paths ===
DATASET_2025_PATH = r'C:/Users/COMTECH COMPUTER/Desktop/AI Based Load Forecasting/prediction_data_12_2024.csv'

MODEL_PATHS = {
    "Linear Regression": r"C:/Users/COMTECH COMPUTER/Desktop/AI Based Load Forecasting/LR.pkl",
    "Ridge Regression": r"C:/Users/COMTECH COMPUTER/Desktop/AI Based Load Forecasting/Ridge.pkl",
    "Lasso Regression": r"C:/Users/COMTECH COMPUTER/Desktop/AI Based Load Forecasting/Lasso.pkl",
    "ElasticNet": r"C:/Users/COMTECH COMPUTER/Desktop/AI Based Load Forecasting/ElasticNet.pkl",
    "K-Nearest Neighbors": r"C:/Users/COMTECH COMPUTER/Desktop/AI Based Load Forecasting/KNN.pkl",
    "Random Forest": r"C:/Users/COMTECH COMPUTER/Desktop/AI Based Load Forecasting/RF01.pkl",
    "AdaBoost": r"C:/Users/COMTECH COMPUTER/Desktop/AI Based Load Forecasting/AdaBoost.pkl",
    "Gradient Boosting": r"C:/Users/COMTECH COMPUTER/Desktop/AI Based Load Forecasting/GradientBoost.pkl",
    "XGBoost": r"C:/Users/COMTECH COMPUTER/Desktop/AI Based Load Forecasting/rs_xgb.pkl",
    "LightGBM": r"C:/Users/COMTECH COMPUTER/Desktop/AI Based Load Forecasting/LightGradientBoost.pkl",
    "CatBoost": r"C:/Users/COMTECH COMPUTER/Desktop/AI Based Load Forecasting/CatGradientBoost.pkl",
    "Voting": r"C:/Users/COMTECH COMPUTER/Desktop/AI Based Load Forecasting/Voting.pkl",
    "Stacking": {
        "base": r"C:/Users/COMTECH COMPUTER/Desktop/AI Based Load Forecasting/Base_Models_for_Stacking.pkl",
        "meta": r"C:/Users/COMTECH COMPUTER/Desktop/AI Based Load Forecasting/Meta_Model_for_Stacking.pkl"
    },
    "Blending": {
        "base": r"C:/Users/COMTECH COMPUTER/Desktop/AI Based Load Forecasting/Base_Models_for_Stacking.pkl",
        "meta": r"C:/Users/COMTECH COMPUTER/Desktop/AI Based Load Forecasting/Blending.pkl"
    }
}

FEATURE_COLUMNS = [
    'embedded_wind_generation', 'embedded_solar_generation', 'wind + solar generation',
    'year', 'weekend', 'hour_of_day_x', 'hour_of_day_y', 'week_x', 'week_y',
    'month_x', 'month_y', 'demand_diff', 'wind_diff', 'solar_diff',
    'rolling_mean_24', 'rolling_std_24', 'rolling_mean_48', 'rolling_std_48',
    'load_lag_48', 'timeofday_Evening', 'timeofday_Morning',
    'timeofday_Night', 'season_Spring', 'season_Summer', 'season_Winter'
]
TARGET_COLUMN = 'nd'

@st.cache_data
def load_data():
    df = pd.read_csv(DATASET_2025_PATH)
    df['settlement_date'] = pd.to_datetime(df['settlement_date'])
    df.set_index('settlement_date', inplace=True)
    df[TARGET_COLUMN].fillna(df[TARGET_COLUMN].mean(), inplace=True)
    df.dropna(subset=FEATURE_COLUMNS, inplace=True)
    return df

@st.cache_resource
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def calculate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    return mae, rmse, r2

def plot_demand_forecast(y_true, y_pred, model_name, start, end):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true.index, y=y_true, mode='lines+markers', name='Actual',
                             line=dict(color='royalblue', width=2)))
    fig.add_trace(go.Scatter(x=y_pred.index, y=y_pred, mode='lines+markers', name='Predicted',
                             line=dict(color='tomato', width=2, dash='dash')))

    fig.update_layout(
        title=f"{model_name} Forecast: {start.date()} to {end.date()}",
        xaxis_title="Time",
        yaxis_title="Electricity Demand (MW)",
        legend=dict(x=0.01, y=0.99),
        template="plotly_white",
        margin=dict(t=60, b=40),
        height=500
    )
    return fig

def predict_with_stacking(X_input):
    base_estimators = load_model(MODEL_PATHS["Stacking"]["base"])
    meta_model = load_model(MODEL_PATHS["Stacking"]["meta"])
    meta_features = np.column_stack([model.predict(X_input) for name, model in base_estimators])
    return meta_model.predict(meta_features)

def predict_with_blending(X_input):
    base_estimators = load_model(MODEL_PATHS["Blending"]["base"])
    meta_model_blend = load_model(MODEL_PATHS["Blending"]["meta"])
    meta_features = np.column_stack([model.predict(X_input) for name, model in base_estimators])
    return meta_model_blend.predict(meta_features)

def main():
    st.title("⚡ Electricity Demand Forecasting")
    st.markdown("Visualize and compare the performance of multiple forecasting models using real-world electricity demand data.")
    st.markdown("---")

    data = load_data()
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    st.sidebar.header("⚙️ Options")
    selected_model_name = st.sidebar.selectbox("Select Model", list(MODEL_PATHS.keys()))
    model_path = MODEL_PATHS[selected_model_name]

    st.sidebar.markdown("---")
    st.sidebar.header("📅 Select Date Range")
    min_date = data.index.min().date()
    max_date = data.index.max().date()

    date_range = st.sidebar.date_input("Forecast Date Range:",
        value=[min_date, min_date + timedelta(days=1)],
        min_value=min_date,
        max_value=max_date)

    if not date_range or len(date_range) != 2:
        st.info("ℹ️ Please select a valid start and end date.")
        return

    start_time = pd.Timestamp(date_range[0])
    end_time = pd.Timestamp(date_range[1]) + pd.Timedelta(hours=23, minutes=30)

    X_test = X.loc[start_time:end_time]
    y_test = y.loc[start_time:end_time]

    if X_test.empty or y_test.empty:
        st.error("❌ No data available for selected range.")
        return

    from sklearn.preprocessing import StandardScaler

    if selected_model_name == "K-Nearest Neighbors":
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_test)
        y_scaled = scaler_y.fit_transform(y_test.values.reshape(-1, 1))
        model = load_model(model_path)
        y_pred_scaled = model.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        predictions = pd.Series(y_pred, index=y_test.index)

    elif selected_model_name == "Stacking":
        predictions = predict_with_stacking(X_test)

    elif selected_model_name == "Blending":
        predictions = predict_with_blending(X_test)

    else:
        model = load_model(model_path)
        y_pred = model.predict(X_test)
        predictions = pd.Series(y_pred.flatten(), index=y_test.index)

    pred_series = pd.Series(np.array(predictions).flatten(), index=X_test.index)
    y_true = y_test.loc[pred_series.index]
    y_pred = pred_series.loc[y_test.index]

    mae, rmse, r2 = calculate_metrics(y_true.values, y_pred.values)

    st.subheader(f"📉 Model Performance: {selected_model_name}")
    col1, col2, col3 = st.columns(3)
    col1.metric("🔹 MAE", f"{mae:.2f}")
    col2.metric("🔸 RMSE", f"{rmse:.2f}")
    col3.metric("📈 R²", f"{r2:.4f}")

    with st.expander("🔍 Forecast Visualization", expanded=True):
        fig = plot_demand_forecast(y_true, y_pred, selected_model_name, start_time, end_time)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.caption("Developed as part of Final Year Project | AI-Based Load Forecasting")

if __name__ == "__main__":
    main()

