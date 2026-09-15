import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.G_linear_1 = torch.nn.Linear(in_features=64, out_features=128, bias=True)
        self.ReLU = torch.nn.ReLU(inplace=False)
        self.G_linear_2 = torch.nn.Linear(in_features=128, out_features=784, bias=True)
        self.D_linear_1 = torch.nn.Linear(in_features=784, out_features=128, bias=True)
        self.LeakyReLU = torch.nn.LeakyReLU(negative_slope=0.2, inplace=False)
        self.D_linear_2 = torch.nn.Linear(in_features=128, out_features=1, bias=True)

    def forward(self, x):
        t1 = self.G_linear_1(x)
        t2 = self.ReLU(t1)
        t3 = self.G_linear_2(t2)
        t4 = self.D_linear_1(t3)
        t5 = self.LeakyReLU(t4)
        t6 = self.D_linear_2(t5)
        return t6


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

    G_optimizer = torch.optim.Adam(list(model.G_linear_1.parameters()) + list(model.G_linear_2.parameters()), weight_decay=0, betas=(0.5, 0.999), eps=1e-8, lr=0.0002)
    D_optimizer = torch.optim.Adam(list(model.D_linear_1.parameters()) + list(model.D_linear_2.parameters()), weight_decay=0, betas=(0.5, 0.999), eps=1e-8, lr=0.0002)
    G_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
    D_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")

    train_loader, val_loader = build_dataloaders()

    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            x = batch["inputs"].to(device)
            y = batch.get("labels")
            if y is not None:
                y = y.to(device)
            # G step
            G_optimizer.zero_grad()
            logits = model(x)
            loss_total = G_loss(logits, y) * 1
            loss_total.backward()
            G_optimizer.step()

            # D step
            D_optimizer.zero_grad()
            logits = model(x)
            loss_total = D_loss(logits, y) * 1
            loss_total.backward()
            D_optimizer.step()
            global_step += 1
        # TODO: validation pass per epoch
        print(f"epoch {epoch} done — step={global_step}")


if __name__ == "__main__":
    train()
