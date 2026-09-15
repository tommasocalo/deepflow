import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear = torch.nn.Linear(in_features=128, out_features=64, bias=True)
        self.Linear_2 = torch.nn.Linear(in_features=64, out_features=32, bias=True)
        self.Linear_3 = torch.nn.Linear(in_features=32, out_features=16, bias=True)

    def forward(self, x):
        t1 = self.Linear(x)
        t2 = self.Linear_2(t1)
        t3 = self.Linear_3(t2)
        return t3


# -----------------------------------------------------------------------
# Auto-generated training script. Edit the data loaders to match your task.
# -----------------------------------------------------------------------

import torch
from torch.utils.data import DataLoader


def build_dataloaders():
    """Replace this with real datasets — see Data graph in DeepFlow v2."""
    raise NotImplementedError("Wire up your datasets / dataloaders here.")


def train(num_epochs: int = 10, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)

    Adam_optimizer = torch.optim.Adam(model.parameters(), weight_decay=0, betas=(0.9, 0.999), eps=1e-8, lr=0.001)
    Cross_Entropy_loss = torch.nn.CrossEntropyLoss(reduction="mean", label_smoothing=0)

    train_loader, val_loader = build_dataloaders()

    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            x = batch["inputs"].to(device)
            y = batch.get("labels")
            if y is not None:
                y = y.to(device)
            Adam_optimizer.zero_grad()
            logits = model(x)
            loss_total = Cross_Entropy_loss(logits, y) * 1
            loss_total.backward()
            Adam_optimizer.step()
            global_step += 1
        # TODO: validation pass per epoch
        print(f"epoch {epoch} done — step={global_step}")


if __name__ == "__main__":
    train()
