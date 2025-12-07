# wl_gcl/src/trainers/eval.py
import torch
from torch.optim import Adam

@torch.no_grad()
def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def evaluate_linear_probe(
    model: torch.nn.Module,
    data: torch.Tensor,
    num_classes: int,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-2,
) -> float:
    """
    Standard linear probe evaluation for node classification.

    Encoder is frozen. A single linear layer is trained on top
    of the learned embeddings using train_mask and evaluated
    on test_mask.

    Since only the linear layer is trained:

    -> Any improvement in classification accuracy comes solely from the **quality** of the embeddings.

    -> There is no help from task-specific nonlinear layers.
    """

    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)

    z = z.detach()

    clf = torch.nn.Linear(z.size(1), num_classes).to(device)
    optimizer = Adam(clf.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_mask, _, test_mask = data.train_mask, data.val_mask, data.test_mask

    for _ in range(epochs):
        clf.train()
        optimizer.zero_grad()

        logits = clf(z[train_mask])
        loss = loss_fn(logits, data.y[train_mask])

        loss.backward()
        optimizer.step()

    clf.eval()

    with torch.no_grad():
        test_logits = clf(z[test_mask])

    return compute_accuracy(test_logits, data.y[test_mask])