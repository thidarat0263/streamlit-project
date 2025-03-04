import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Set up the Streamlit app title and description
#st.sidebar.title("Navigation")
st.sidebar.success("Select a page above.")
st.title("Development Approach")
st.subheader(":blue[Neural Network]")

tab1, tab2,tab3 = st.tabs(["Dataset and Preparation Process", "NN Algorithms", "Model Development Procedure"])
with tab1:
    st.write("The dataset is from website calls [Kaggle](https://www.kaggle.com). It's the Iris dirty dataset that contains measurements of different Iris species.")
    st.caption('The fact that it names "Dirty" data because it came with inconsistencies, missing values to practicing for data cleaning')

    st.link_button("Go to source", "https://www.kaggle.com/datasets/jaskarandhillon1609/iris-dirty-dataset?select=iris_dirty.csv")

    st.markdown(""" 
    ### Feature Descriptions
    - **Sepal Length:** The length of the sepal in centimeters.
    - **Sepal Width:** The width of the sepal in centimeters.
    - **Petal Length:** The length of the petal in centimeters.
    - **Petal Width:** The width of the petal in centimeters.
    - **Species:** The class label (0 = Setosa, 1 = Versicolor, 2 = Virginica).""")

    st.markdown("### Data Preparation Process")
    st.write("*1.* Started by downloading the dataset from the website above.")
    st.write("*2.* Define path to dataset and then Load the dataset from the predefined path.")
    code = '''
    dataset_path = r"C:\\Username\\Downloads\\iris_dirty.csv"
    try:
        df = pd.read_csv(dataset_path)
        # st.write("Dataset loaded successfully from the path.")
    except Exception as e:
        st.error(f"Error loading dataset: ")'''
    st.code(code, language="python")
    # Define the path to your dataset 


# Upload dataset using Streamlit's file uploader
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

    if uploaded_file is not None:
        try:
        # Read the CSV file directly from the uploaded file
            df = pd.read_csv(uploaded_file)
            st.write("Dataset loaded successfully!")
            st.write(df.head(20))  # Show the first 20 rows of the dataframe
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
    else:
        st.write("Please upload a CSV file.")
    code = '''
    st.subheader("Dataset Preview")
    st.write(df.head(20))'''
    st.code(code, language="python")

    # Show dataset preview
    st.subheader("Dataset Preview")
    st.write(df.head(20))

    st.write("**4.** Write code checkbox for Check for missing values and if it checked then fill missing values with the mean of numeric columns.")
    code = '''
    st.subheader("Check for Missing Values")
    missing_values = df.isnull().sum()
    st.write(missing_values)

    fill_missing = st.checkbox("Fill missing values with mean", value=True)
    if fill_missing:
        numeric_columns = df.select_dtypes(include=['number']).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        st.write("Missing values filled with column mean for numeric columns.")
    else:
        st.write("No missing value filling applied.")'''
    st.code(code, language="python")

    st.subheader("Check for Missing Values")
    missing_values = df.isnull().sum()
    st.write(missing_values)

    # Fill missing values with the mean of numeric columns if the checkbox is checked
    fill_missing = st.checkbox("Fill missing values with mean", value=True)
    if fill_missing:
        # Select only numeric columns for filling missing values
        numeric_columns = df.select_dtypes(include=['number']).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        st.write("Missing values filled with column mean for numeric columns.")
    else:
        st.write("No missing value filling applied.")

    st.write(" **5.** Show Data Types and Normalized the Features (this is also the option).")
    code = '''
    st.subheader("Data Types")
    data_types = df.dtypes
    st.write(data_types)

    st.subheader("Normalize Features")
    normalize = st.checkbox("Normalize Features", value=True)

    if normalize:
        # Select only numeric columns 
        numeric_columns = df.select_dtypes(include=['number']).columns
        X = df[numeric_columns]  # Only numeric columns should be scaled
        
        # Standardize the numeric columns
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create a DataFrame from the scaled values
        df_scaled = pd.DataFrame(X_scaled, columns=numeric_columns)
        
        # Keep the non-numeric columns in the final DataFrame
        if 'species' in df.columns:
            df_scaled['species'] = df['species']
        
        st.write("Features normalized using StandardScaler.")
        st.write(df_scaled.head())
    else:
        st.write("Features not normalized.")
        st.write(df.head())'''
    st.code(code, language="python")

    st.subheader("Data Types")
    data_types = df.dtypes
    st.write(data_types)

    # Normalize the Features
    st.subheader("Normalize Features")
    normalize = st.checkbox("Normalize Features", value=True)

    if normalize:
        # Select only numeric columns (exclude 'species' or other non-numeric columns)
        numeric_columns = df.select_dtypes(include=['number']).columns
        X = df[numeric_columns]  # Only numeric columns should be scaled
        
        # Standardize the numeric columns
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create a DataFrame from the scaled values
        df_scaled = pd.DataFrame(X_scaled, columns=numeric_columns)
        
        # Keep the non-numeric columns (like 'species') in the final DataFrame
        if 'species' in df.columns:
            df_scaled['species'] = df['species']
        
        st.write("Features normalized using StandardScaler.")
        st.write(df_scaled.head())
    else:
        st.write("Features not normalized.")
        st.write(df.head())

    st.write("**6.** Show the Cleaned Dataset with limit preview only 20 rows.")
    code = '''
    st.subheader("Cleaned Dataset Preview")
    st.write(df.head(20))'''
    st.code(code,language="python")
    # Show cleaned dataset
    st.subheader("Cleaned Dataset Preview")
    st.write(df.head(20))

    st.write("Additional: Code for option to download the cleaned dataset.")
    code = '''
    st.download_button(
        label="Download Cleaned Dataset",
        data=df.to_csv(index=False).encode(),
        file_name="cleaned_iris.csv",
        mime="text/csv"
    )'''
    st.code(code,language="python")
    st.download_button(
        label="Download Cleaned Dataset",
        data=df.to_csv(index=False).encode(),
        file_name="cleaned_iris.csv",
        mime="text/csv"
    )
    st.write("**7.** Convert the 'species' column to string if it's not already and ensure that all columns have consistent data types. then display the cleaned dataset (again).")
    code = '''
    if 'species' in df.columns:
        df['species'] = df['species'].astype(str)

    df = df.convert_dtypes()

    st.write("Cleaned DataFrame for Display")
    st.write(df)'''
    st.code(code, language="python")
    # Ensure proper data types before displaying or downloading
    if 'species' in df.columns:
        df['species'] = df['species'].astype(str)

    # Ensure all columns have consistent data types
    df = df.convert_dtypes()

    #Display the cleaned dataset
    st.write("Cleaned DataFrame for Display")
    st.write(df)
with tab2:
    st.markdown("### Algorithm")
    st.write("**Feedforward Neural Network algorithm have been chosen for Neural Network model.** ")
    st.markdown('''- ### :gray[**Feedforward Neural Network Algorithm**] ''')
    st.write(" **Feedforward Neural Network (FFNN)** is a type of artificial neural network where connections between the nodes do not form cycles.The network consists of an input layer, one or more hidden layers, and an output layer. Information flows in one direction—from input to output—hence the name 'feedforward.'")
    st.write("""**Structure of a Feedforward Neural Network**
- **Input Layer:** The input layer consists of neurons that receive the input data. Each neuron in the input layer represents a feature of the input data.
- **Hidden Layers:** One or more hidden layers are placed between the input and output layers. These layers are responsible for learning the complex patterns in the data. Each neuron in a hidden layer applies a weighted sum of inputs followed by a non-linear activation function.
- **Output Layer:** The output layer provides the final output of the network. The number of neurons in this layer corresponds to the number of classes in a classification problem or the number of outputs in a regression problem.""")
    st.write(""" **Some Keywords:** 
    - **Epoch** is a common keyword in deep learning, which means a single pass through the training set. If the training set has 60,000 samples, one epoch leads to 60 gradient descent steps. Then it will start over and take another pass through the training set. It means one more decision to make, the optimal number of epochs. It is decided by looking at the trends of performance metrics on a holdout set of training data. 
    - **Learning rate** is a hyperparameter that governs how much a machine learning model adjusts its parameters at each step of its optimization algorithm. The learning rate can determine whether a model delivers optimal performance or fails to learn during the training process. """)

with tab3:
    st.markdown("### Development Procedure")
    st.write('''- ### :gray[Feedforward Neural Network] ###''')
    st.write("These are the steps of how i implemented the model.")
    st.markdown(" **Step 1:** Aside from import all the important libraries, Start by loading the Iris dataset and then split the dataset into training and test sets.")
    code = '''
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)'''
    st.code(code, language="python")
    st.markdown(" **Step 2:** Normalize the data using StandardScaler.")
    code = '''
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)'''
    st.code(code, language="python")
    st.markdown(" **Step 3:** Created FFNN class to defines a simple neural network with one hidden layer and an output layer. ")
    code = '''
    class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First hidden layer
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Use ReLU activation
        x = self.fc2(x)
        return x'''
    st.code(code, language="python")
    st.markdown(" **Step 4:** Convert data to tensors and get input and output dimensions." )
    code = '''
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

input_dim = X_train.shape[1]
output_dim = len(set(y_train))'''
    st.code(code, language="python")
    st.markdown(" **Step 5:** Build widgets for interactive parameters of hidden layer, number of epochs and Learning rate. Then create the model with dynamic parameters." )
    code = '''
hidden_dim = st.slider('Hidden Layer Size', 16, 128, 64)  
num_epochs = st.slider('Number of Epochs', 50, 200, 100)  
learning_rate = st.slider('Learning Rate', 0.0001, 0.01, 0.001, step=0.0001)  

model = FFNN(input_dim, hidden_dim, output_dim)'''
    st.code(code, language="python")
    st.markdown(" **Step 6:** Call the Loss function and optimizer then initialize lists to track metrics." )
    code = '''

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loss_values, val_loss_values = [], []
train_acc_values, val_acc_values = [], []'''
    st.code(code, language="python")
    st.markdown(" **Step 7:** Start training the model. " )
    code = '''
    for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Zero out the gradients

    # Forward pass and calculate loss
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Compute training accuracy
    _, predicted_train = torch.max(outputs, 1)
    train_accuracy = (predicted_train == y_train_tensor).sum().item() / y_train_tensor.size(0)

    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    # Store training accuracy
    train_loss_values.append(loss.item())
    train_acc_values.append(train_accuracy)'''
    st.code(code, language="python")
    st.markdown(" **Step 8:** Start the Validation step by evaluating model, compute validation accuracy and store validation accuracy." )
    code  = '''
    model.eval()  
    with torch.no_grad():  
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor)

        # Compute validation accuracy
        _, predicted_val = torch.max(val_outputs, 1)
        val_accuracy = (predicted_val == y_test_tensor).sum().item() / y_test_tensor.size(0)

        # Store validation metrics
        val_loss_values.append(val_loss.item())
        val_acc_values.append(val_accuracy)'''
    st.code(code, language="python")
    st.markdown(" **Step 9:** Display accuracy after every 10 epochs." )
    code = '''
    if (epoch + 1) % 10 == 0:
        st.text(f"Epoch [{epoch+1}/{num_epochs}] - "
                f"Train Accuracy: {train_accuracy * 100:.2f}% | "
                f"Validation Accuracy: {val_accuracy * 100:.2f}%")'''
    st.code(code, language="python")
    st.markdown(" **Step 10:** Display the final accuracy after training." )
    code = '''
st.subheader("Final Accuracy after Training.")
st.write(f"Training Accuracy: {train_acc_values[-1] * 100:.2f}%")
st.write(f"Validation Accuracy: {val_acc_values[-1] * 100:.2f}%")'''
    st.code(code, language="python")
    st.markdown(" **Step 11:** Create DataFrame for loss and accuracy metrics." )
    code = '''
metrics_df = pd.DataFrame({
    "Epoch": list(range(1, num_epochs + 1)),
    "Training Loss": train_loss_values,
    "Validation Loss": val_loss_values,
    "Training Accuracy": train_acc_values,
    "Validation Accuracy": val_acc_values
})'''
    st.code(code, language="python")
    st.markdown(" **Step 12:** Plot Loss and Accuracy Graph. " )
    code = '''
fig = px.line(metrics_df, x="Epoch", 
              y=["Training Loss", "Validation Loss", "Training Accuracy", "Validation Accuracy"],
              title="Training & Validation Loss and Accuracy",
              labels={"value": "Value", "variable": "Metric"},
              markers=True)
st.plotly_chart(fig)'''
    st.code(code, language="python")



