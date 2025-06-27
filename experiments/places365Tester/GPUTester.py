import torch

if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("Using GPU")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)