import torch
import torch.nn as nn


# ---- Dummy reward network ----
class DummyRewardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                 # (B, 3, 224, 224) -> (B, 150528)
            nn.Linear(1000, 128), # 3 *224 * 224
            nn.ReLU(),
            nn.Linear(128, 1)             # output scalar reward
        )

    def forward(self, x):
        return self.net(x)


# Create model
model = DummyRewardNet()
model.eval()

# Example input (NCHW!)
example_input = torch.randn(1, 1000)

# Export using TorchScript tracing
scripted_model = torch.jit.trace(model, example_input)

# Save
scripted_model.save("dummy_reward_model.pt")

print("Model exported.")