import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# Load the Dataset

data = fetch_california_housing()
X = data.data[:, [0, 1, 2]] 
y = data.target.reshape(-1, 1)

# 2. Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Define and Train the Model
model = nn.Linear(3, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print("Training model...")
for epoch in range(500):
    inputs = torch.FloatTensor(X_scaled)
    labels = torch.FloatTensor(y)
    
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 4. Generate the JavaScript Code
weights = model.weight.data.numpy().flatten().tolist()
bias = model.bias.data.item()
means = scaler.mean_.tolist()
scales = scaler.scale_.tolist()

print("\n" + "="*30)
print("COPY AND PASTE THIS INTO YOUR INDEX.HTML")
print("="*30)
print(f"const weights = {weights};")
print(f"const bias = {bias};")
print(f"const means = {means};")
print(f"const scales = {scales};")
print("="*30)
