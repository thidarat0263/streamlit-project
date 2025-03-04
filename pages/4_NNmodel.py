import torch
import torch.nn as nn
import torch.optim as optim
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#st.sidebar.title("Navigation")
st.sidebar.success("Select a page above.")
# Load Iris dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

st.subheader("Training and Evaluation of FFNN on Iris Dataset")

class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  
        x = self.fc2(x)
        return x

# Convert data to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Get input and output dimensions
input_dim = X_train.shape[1]
output_dim = len(set(y_train))

# Streamlit widgets
hidden_dim = st.slider('Hidden Layer Size', 16, 128, 64)
num_epochs = st.slider('Number of Epochs', 50, 200, 100)
learning_rate = st.slider('Learning Rate', 0.0001, 0.01, 0.001, step=0.0001)

# Create the model
model = FFNN(input_dim, hidden_dim, output_dim)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Lists to track metrics
train_loss_values, val_loss_values = [], []
train_acc_values, val_acc_values = [], []

# Training Loop
for epoch in range(num_epochs):
    model.train()  
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Compute training accuracy
    _, predicted_train = torch.max(outputs, 1)
    train_accuracy = (predicted_train == y_train_tensor).sum().item() / y_train_tensor.size(0)

    # Backward pass
    loss.backward()
    optimizer.step()

    # Store training metrics
    train_loss_values.append(loss.item())
    train_acc_values.append(train_accuracy)

    # Validation Step
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor)

        # Compute validation accuracy
        _, predicted_val = torch.max(val_outputs, 1)
        val_accuracy = (predicted_val == y_test_tensor).sum().item() / y_test_tensor.size(0)

        # Store validation metrics
        val_loss_values.append(val_loss.item())
        val_acc_values.append(val_accuracy)

    if (epoch + 1) % 10 == 0:
        st.text(f"Epoch [{epoch+1}/{num_epochs}] - "
                f"Train Accuracy: {train_accuracy * 100:.2f}% | "
                f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Display the final accuracy after training
st.subheader("Final Accuracy after Training")
st.write(f"Training Accuracy: {train_acc_values[-1] * 100:.2f}%")
st.write(f"Validation Accuracy: {val_acc_values[-1] * 100:.2f}%")

# Create DataFrame for loss and accuracy metrics
metrics_df = pd.DataFrame({
    "Epoch": list(range(1, num_epochs + 1)),
    "Training Loss": train_loss_values,
    "Validation Loss": val_loss_values,
    "Training Accuracy": train_acc_values,
    "Validation Accuracy": val_acc_values
})

# Plot Loss and Accuracy Graph
fig = px.line(metrics_df, x="Epoch", 
              y=["Training Loss", "Validation Loss", "Training Accuracy", "Validation Accuracy"],
              title="Training & Validation Loss and Accuracy",
              labels={"value": "Value", "variable": "Metric"},
              markers=True)
st.plotly_chart(fig)
