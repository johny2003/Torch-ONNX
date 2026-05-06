import torch
import torch.nn as nn

# This matches the model you've been trying to export
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(3, 1)

model = RegressionModel()


print(f"W1, W2, W3: {model.linear.weight.data.tolist()[0]}")
print(f"Bias: {model.linear.bias.data.item()}")