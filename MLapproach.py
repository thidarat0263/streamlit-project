import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go

# Set page config
st.set_page_config(page_title="Project IS")
st.sidebar.success("Select a page above.")
def app():
    st.title("DevelopMent Approach")
    st.subheader(":orange[Machine Learning]")
    tab1, tab2,tab3 = st.tabs(["Dataset and Preparation Process", "ML Algorithms", "Model Development Procedure"])
    with tab1:
        st.write("The dataset is from website calls [Kaggle](https://www.kaggle.com). It's the Dirty Cafe Sales dataset that contains 10,000 rows of synthetic data representing sales transactions in a cafe.")
        st.caption('The fact that it names "Dirty" data because it came with missing values and errors to practicing for data cleaning')

        st.link_button("Go to source", "https://www.kaggle.com/datasets/ahmedmohamed2003/cafe-sales-dirty-data-for-cleaning-training")
        st.markdown("""
                ### Feature Descriptions
                - **Quantity**: Number of units that sold in a transaction.
                - **Price Per Unit**: Cost of one unit of an item.
                - **Total Spent**: Total amount paid (calculated as Quantity × Price Per Unit).
                """)
        st.markdown("### Data Preparation Process")
        st.write("*1.* Started by downloading the dataset from the website above.")
        st.write("*2.* Write the code to check if there are errors from any features then Handling missing data by fill missing values with the mean of each column.")
        code = '''if 'data' in locals():
                data['Quantity'] = pd.to_numeric(data['Quantity'], errors='coerce')
                data['Price Per Unit'] = pd.to_numeric(data['Price Per Unit'], errors='coerce')
                data['Total Spent'] = pd.to_numeric(data['Total Spent'], errors='coerce')

                data['Quantity'].fillna(data['Quantity'].mean(), inplace=True)
                data['Price Per Unit'].fillna(data['Price Per Unit'].mean(), inplace=True)
                data['Total Spent'].fillna(data['Total Spent'].mean(), inplace=True)

                nan_rows = data[data.isna().any(axis=1)]
                if not nan_rows.empty:
                        st.write("Rows with missing data after conversion:", nan_rows)

                data.fillna(0, inplace=True)'''
        st.code(code, language="python")
        st.write("*3.* You can see the raw data and missing values name '**ERRORS**' in the features and There is '**Preprocessed Data Preview**' which is the data that already been handled.")

        st.markdown("### Raw CSV File")  
        st.write("See the preview of the data below:")

        try:
                data = pd.read_csv("dirty_cafe_sales.csv")
                st.write("**Raw Data Preview:**", data)  
        except Exception as e:
                st.error(f"⚠️ Error loading CSV file: {e}")  


        st.markdown("### Preprocess the data: Handle missing values (NaN)")
        st.write("- **Handling missing data**")
                
        if 'data' in locals():
                data['Quantity'] = pd.to_numeric(data['Quantity'], errors='coerce')
                data['Price Per Unit'] = pd.to_numeric(data['Price Per Unit'], errors='coerce')
                data['Total Spent'] = pd.to_numeric(data['Total Spent'], errors='coerce')

        # Fill missing values with the mean of each column
                data['Quantity'].fillna(data['Quantity'].mean(), inplace=True)
                data['Price Per Unit'].fillna(data['Price Per Unit'].mean(), inplace=True)
                data['Total Spent'].fillna(data['Total Spent'].mean(), inplace=True)

                nan_rows = data[data.isna().any(axis=1)]
                if not nan_rows.empty:
                        st.write("Rows with missing data after conversion:", nan_rows)

        
                data.fillna(0, inplace=True)
        # Display preprocessed data
                st.write("Preprocessed Data Preview:", data)
        else:
                st.warning("Please ensure the CSV file is loaded correctly.")
        with tab2:
             st.markdown("### Algorithm")
             st.write("**Two algorithms have been chosen for Machine Learning model.** ")
             st.markdown('''- ### :gray[**Linear Regression Algorithm**] ''')
             st.write(" **Linear regression** is a type of supervised machine-learning algorithm that learns from the labelled datasets and maps the data points with most optimized linear functions which can be used for prediction on new datasets. It computes the linear relationship between the dependent variable and one or more independent features by fitting a linear equation with observed data.")
             st.write("In this case, with the dataset we have key features: Quantity, Price Per Unit and Total spent. If we want to predicted Total spent we consider 2 factor which is Quantity and Price Per Unit, linear regression uses all these parameter to predict Total Spent as it consider a linear relation between all these features and Total Spent.")
             st.markdown('''- ### :gray[**K-Nearest Neighbors (Regression) Algorithm**] ''')
             st.write("KNN regression is a non-parametric method used for predicting continuous values. The core idea is to predict the target value for a new data point by averaging the target values of the K nearest neighbors in the feature space.")
             st.write("**How KNN Regression Works**")
             st.write(" - Choosing the number of neighbors (K): The initial step involves selecting the number of neighbors. This choice greatly affects the model's performance. A smaller value of K makes the model more prone to noise, whereas a larger value of K results in smoother predictions.")
             st.write(" - Calculating distances: For a new data point, calculate the distance between this point and all points in the training set.")
             st.write("- Finding K nearest neighbors: Identify the K points in the training set that are closest to the new data point.")
             st.write("- Predicting the target value: Compute the average of the target values of the K nearest neighbors and use this as the predicted value for the new data point.")

        with tab3:
              st.markdown("### Development Procedure")
              st.write('''- ### :gray[Linear Regression] ###''')
              st.write("These are the steps of how i implemented the model.")
              st.markdown(" **Step 1:** Aside from import all the important libraries i started from the dataset by ensure the dataset  in the correct path.")
              code = '''if 'data' not in st.session_state: 
                            st.session_state.data = pd.read_csv('dirty_cafe_sales.csv')'''
              st.code(code, language="python")
              st.markdown(" **Step 2:** Use the stored dataset and convert columns to numeric and handling missing values. ")
              code = '''
              for col in ['Quantity', 'Price Per Unit', 'Total Spent']:
                  data[col] = pd.to_numeric(data[col], errors='coerce')
                  data[col].fillna(data[col].mean(), inplace=True)
              feature_column = st.selectbox("Select Feature for Prediction (X)", ['Quantity', 'Price Per Unit'])
              target_column = 'Total Spent'

              X = data[[feature_column]].values
              Y = data[target_column].values'''
              st.code(code, language="python")
              st.markdown(" **Step 3:** Then i started to trained the Linear Regression Model and  set predictions and display model intercept and coefficient.")
              code = '''
              model = LinearRegression()
        model.fit(X, Y)

        Y_pred = model.predict(X)

        st.write(f"**Intercept:** {model.intercept_:.2f}")
        st.write(f"**Coefficient:** {model.coef_[0]:.2f}")'''
              st.code(code, language="python")
              st.markdown(" **Step 4:** The next step is plot your Interactive Graph. I plot my graph with Plotly and also using original values. (Note:set the blue color for Actual Data and red color for Regression Line).")
              code = '''
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
        st.plotly_chart(fig, key="linear_regression_plot")'''
              st.code(code, language="python")
              st.markdown(" **Step 5:** User can be able to see the prediction numbers of Linear Regression by this code.")
              code = ''' 
        st.subheader(f"Predict {target_column} Based on {feature_column}")
        user_input = st.number_input(f"Enter {feature_column}:", min_value=0.0, step=0.1, key=f"input_{feature_column}_lr")

        user_input_rounded = round(user_input, 2)

        if user_input_rounded > 0:
            user_input_array = np.array([[user_input_rounded]])  # Convert to 2D array
            user_prediction = model.predict(user_input_array)

            user_prediction_rounded = round(user_prediction[0], 2)
            st.write(f"Predicted {target_column} for {feature_column} {user_input_rounded}: **${user_prediction_rounded:,.2f}**")'''
              st.code(code, language="python")

              st.write('''- ### :gray[K-Nearest Neighbors Regression] ###''')
              st.write("These are the steps of how i implemented the model.")
              st.markdown(" **Step 1:** Started from implemented selectbox for users to choose feature they want to see for prediction.")
              code = '''feature_column_knn = st.selectbox("Select Feature for KNN Prediction (X)", ['Quantity', 'Price Per Unit'], key="knn_feature_reg")'''
              st.code(code, language="python")
              st.markdown(" **Step 2:** Define X and Y for what are features/Target, Train-Test split and then apply MinMaxScaler for KNN.")
              code = '''
        X = data[[feature_column_knn]].values  # Features
        y = data["Total Spent"].values         # Target variable (Regression)

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)'''
              st.code(code, language="python")
              st.markdown(" **Step 3:** Put a slider for user to selects K between numbers 1 to 20, Then start to train the KNN Regression Model and set Predictions. ")
              code = '''
        n_neighbors = st.slider("Select Number of Neighbors (K)", min_value=1, max_value=20, value=5, step=1)

        knn_model = KNeighborsRegressor(n_neighbors=n_neighbors)
        knn_model.fit(X_train, y_train)

        y_pred = knn_model.predict(X_test)'''
              st.code(code, language="python")
              st.markdown(" **step 4:** Scatter Plot the graph by using original values to plot actual points and predicted points.")
              code = '''
              fig = go.Figure()

        fig.add_trace(go.Scatter(x=X.flatten(), y=y, mode='markers', name="Actual Data", marker=dict(color='blue')))
        
        fig.add_trace(go.Scatter(x=X.flatten(), y=y_pred, mode='markers', name="Predicted Points", marker=dict(color='red', symbol='x')))

        fig.update_layout(
            title=f"KNN Regression for {feature_column_knn} vs. Total Spent",
            xaxis_title=feature_column_knn,
            yaxis_title="Total Spent",
            showlegend=True,
            hovermode='x'
        )

        st.plotly_chart(fig, key=f"knn_regression_{feature_column_knn}_plot")'''
              st.code(code, language="python")
if __name__ == "__main__":
    app()
