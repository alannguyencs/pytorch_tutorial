import torch
import torch.nn as nn

"""
Problem: Given two sets of inputs and targets, build a linear regression model to fit them
Inputs: 10 samples of 3D vector
Targets: 10 samples of 1D vector
Linear model: targets = inputs * weight + bias (like Y = X * W + b)
           --> optimize weight and bias
reference: https://mccormickml.com/2014/03/04/gradient-descent-derivation/
"""

inputs = torch.randn(10, 3)
targets = torch.randn(10, 1)

#===== Setup model =========
linear = nn.Linear(3, 1)   # 10x1 = 10x3 * 3x1
print ('weight: ', linear.weight)
print ('bias: ', linear.bias)

"""
Setup loss function and optimizer
pred_0 = inputs * weight_0 + bias_0
Loss:
    L1 loss = |targets - pred_0|
    MSELoss = (targets - pred_0)^2

gradient descent: linear_new = linear_old - lr * gradient
"""
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

#===== Model training =========
# Forward pass.
pred = linear(inputs)

# Backward pass = compute loss + compute gradients + update weights
loss = criterion(pred, targets)
print('loss: ', loss.item())

optimizer.zero_grad()   # set gradients to zero
loss.backward()         # compute gradients
print ('dL/dw: ', linear.weight.grad)
print ('dL/db: ', linear.bias.grad)

optimizer.step()   # update weights

