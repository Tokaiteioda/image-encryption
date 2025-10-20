import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
X_tensor = X_tensor.to()
Y_tensor = Y_tensor.to(device)
model = GRUChaos().to(device)
