import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Initialize the model and optimizer
model_time = Net()
optimizer = optim.Adam(model_time.parameters(), lr=0.01)

# Define the time decay scheduler
scheduler = ExponentialLR(optimizer, gamma=0.99)

# Define the loss function
criterion = nn.MSELoss()

# Define the reinforcement learning parameters
num_episodes = 1000
alpha_values = alpha[:, 0]  # replace this with the historical alpha values
l2_distances = alpha[:, 1] # replace this with the historical l2 values corresponding to the alpha value

# Train the model
for episode in range(num_episodes):
    # Select a random alpha value
    alpha_index = torch.randint(0, 100, (1,))
    alpha_value = alpha_values[alpha_index]
    l2_distance = l2_distances[alpha_index]
    
    # Pass the alpha value through the model
    alpha_pred = model_time(alpha_value)
    
    # Calculate the loss
    loss = criterion(alpha_pred, l2_distance)
    
    # Zero the gradients
    optimizer.zero_grad()
    
    # Perform backpropagation
    loss.backward()
    
    # Update the model parameters
    optimizer.step()
    
    scheduler.step()

# Test the model
with torch.no_grad():
    test_time_alpha = torch.rand((1)) * (177-2) + 2 
    test_time_pred = model_time(test_time_alpha)
    print("Predicted L2 distance for alpha = ", test_time_alpha.item(), ": ", test_time_pred.item())