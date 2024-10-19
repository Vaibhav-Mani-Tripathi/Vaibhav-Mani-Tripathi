import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('Disease_symptom_and_patient_profile_dataset.csv')

# Preprocess the data
df = df.dropna()  # Remove missing values
df = df.astype({'Age': int, 'Outcome_Variable': float})  # Convert data types


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('Disease_symptom_and_patient_profile_dataset.csv')

# Preprocess the data
df = df.dropna()  # Remove missing values
df = df.astype({'Age': int, 'Outcome_Variable': float})  # Convert data types

# Convert categorical features to numerical using Label Encoding
for column in df.select_dtypes(include=['object']).columns:
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('Outcome_Variable', axis=1), df['Outcome_Variable'], test_size=0.2, random_state=42)

# Train a random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate the model
accuracy = rf.score(X_test, y_test)
print(f'Accuracy: {accuracy:.3f}')



import torch
import torch.nn as nn
import torch.optim as optim

# Define the model architecture
class InfectionModel(nn.Module):
    def _init_(self):
        super(InfectionModel, self)._init_()
        self.fc1 = nn.Linear(9, 128)  # Input layer - Changed input size to 9 to match X_train_tensor
        self.fc2 = nn.Linear(128, 128)  # Hidden layer
        self.fc3 = nn.Linear(128, 1)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
model = InfectionModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(100):
    optimizer.zero_grad()
    # Assuming X_train and y_train are tensors:
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32) # Convert X_train to tensor
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32) # Convert y_train to tensor
    
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')



    # Make predictions using the machine learning model
predictions = rf.predict(X_test)

# Make predictions using the deep learning model
# Make predictions using the machine learning model
predictions = rf.predict(X_test)

# Make predictions using the deep learning model
# Convert X_test to a PyTorch tensor before passing it to the model
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)  
predictions_dl = model(X_test_tensor)


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



# Create a figure with multiple subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Infection Rate Distribution
sns.histplot(df['Age'], ax=axs[0, 0])
axs[0, 0].set_title('Age')

# Plot 2: Age vs. Infection Rate
sns.scatterplot(x='Age', y='Fever', data=df, ax=axs[0, 1])
axs[0, 1].set_title('Age vs. Fever')

# Plot 3: Infection Rate by Department
sns.barplot(x='Age', y='Cough', data=df, ax=axs[1, 0])
axs[1, 0].set_title('Age vs. Cough')

# Plot 4: Infection Rate Over Time
sns.lineplot(x='Age', y='Fatigue', data=df, ax=axs[1, 1])
axs[1, 1].set_title('Age vs. Fatigue')

# Layout so plots do not overlap
fig.tight_layout()

# Display the plot
plt.show()