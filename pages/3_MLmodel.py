import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#st.sidebar.title("Navigation")
st.sidebar.success("Select a page above.")
st.title("Machine Learning Model")

if 'data' not in st.session_state:
    st.session_state.data = pd.read_csv("dirty_cafe_sales.csv")  # Ensure file exists in correct path

def app():
    tab1, tab2 = st.tabs(["Linear Regression", "K-Nearest Neighbor"])

    with tab1:
        data = st.session_state.data  # Use stored dataset

        # Convert columns to numeric & handle missing values
        for col in ['Quantity', 'Price Per Unit', 'Total Spent']:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            data[col].fillna(data[col].mean(), inplace=True)

        feature_column = st.selectbox("Select Feature for Prediction (X)", ['Quantity', 'Price Per Unit'])
        target_column = 'Total Spent'

        X = data[[feature_column]].values
        Y = data[target_column].values

        # Train the Linear Regression Model
        model = LinearRegression()
        model.fit(X, Y)

        # Predictions
        Y_pred = model.predict(X)

        # Display Model Coefficients
        st.write(f"**Intercept:** {model.intercept_:.2f}")
        st.write(f"**Coefficient:** {model.coef_[0]:.2f}")

        # Plot Interactive Graph with Plotly (using original values)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X.flatten(), y=Y, mode='markers', name='Actual Data', marker=dict(color='blue')))
        fig.add_trace(go.Scatter(x=X.flatten(), y=Y_pred, mode='lines', name='Regression Line', line=dict(color='red')))
        fig.update_layout(
            title=f"Linear Regression {feature_column}: {target_column}",
            xaxis_title=feature_column,
            yaxis_title=target_column,
            showlegend=True,
            hovermode='x'
        )
        st.plotly_chart(fig, key="linear_regression_plot")

        # User Input Prediction for Linear Regression
        st.subheader(f"Predict {target_column} Based on {feature_column}")
        user_input = st.number_input(f"Enter {feature_column}:", min_value=0.0, step=0.1, key=f"input_{feature_column}_lr")

        # Round user input to two decimal places
        user_input_rounded = round(user_input, 2)

        if user_input_rounded > 0:
            user_input_array = np.array([[user_input_rounded]])  # Convert to 2D array
            user_prediction = model.predict(user_input_array)

            # Round the prediction to two decimal places
            user_prediction_rounded = round(user_prediction[0], 2)

            st.write(f"Predicted {target_column} for {feature_column} {user_input_rounded}: **${user_prediction_rounded:,.2f}**")

    with tab2:
        st.subheader("K-Nearest Neighbors Regression Model")
        
        feature_column_knn = st.selectbox("Select Feature for KNN Prediction (X)", ['Quantity', 'Price Per Unit'], key="knn_feature_reg")

        # Define X and Y
        X = data[[feature_column_knn]].values  # Features
        y = data["Total Spent"].values         # Target variable (Regression)

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Apply MinMaxScaler for KNN
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # User selects K
        n_neighbors = st.slider("Select Number of Neighbors (K)", min_value=1, max_value=20, value=5, step=1)

        # Train KNN Regression Model
        knn_model = KNeighborsRegressor(n_neighbors=n_neighbors)
        knn_model.fit(X_train, y_train)

        # Predictions
        y_pred = knn_model.predict(X_test)

        # Scatter Plot (Using original values for the plot)
        fig = go.Figure()

        # Plot actual points (using the real feature values)
        fig.add_trace(go.Scatter(x=X.flatten(), y=y, mode='markers', name="Actual Data", marker=dict(color='blue')))
        
        # Plot predicted points (using the real feature values)
        fig.add_trace(go.Scatter(x=X.flatten(), y=y_pred, mode='markers', name="Predicted Points", marker=dict(color='red', symbol='x')))

        fig.update_layout(
            title=f"KNN Regression for {feature_column_knn} vs. Total Spent",
            xaxis_title=feature_column_knn,
            yaxis_title="Total Spent",
            showlegend=True,
            hovermode='x'
        )

        st.plotly_chart(fig, key=f"knn_regression_{feature_column_knn}_plot")

if __name__ == "__main__":
    app()
