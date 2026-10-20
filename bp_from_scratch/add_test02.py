import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform = transforms.ToTensor()

train_ds = datasets.MNIST('../data', train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST('../data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_ds, batch_size=1000)

#   神经模型
def make_model(use_bn=False):
    layers = [nn.Flatten(), nn.Linear(784, 128)]
    if use_bn:
        layers.append(nn.BatchNorm1d(128))
    layers += [nn.ReLU(), nn.Linear(128, 10)]
    return nn.Sequential(*layers).to(device)

@torch.no_grad()
def evaluate(model):
    model.eval()
    correct = total = 0
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        correct += (model(X).argmax(1) == y).sum().item()
        total += len(y)
    return correct / total

def run(lr=0.1, batch_size=64, opt='sgd', momentum=0.0, use_bn=False, epochs=3, seed=0):
    """跑一次实验，返回每个 epoch 的 loss 和最终 test acc"""
    torch.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    model = make_model(use_bn)

    if opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            loss = loss_fn(model(X), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        losses.append(total_loss / len(train_ds))
    return losses, evaluate(model)

if __name__ == "__main__":

    # print('=== 实验 1：学习率（固定 batch=64, SGD, 3 epoch）===')
    # for lr in [0.001, 0.01, 0.1, 1.0]:
    #     losses, acc = run(lr=lr)
    #     print(f'lr={lr:<6}  loss={losses[-1]:.4f}  acc={acc:.4f}')

    """
    lr=0.001   loss=1.6617  acc=0.7560
    lr=0.01    loss=0.3935  acc=0.9014
    lr=0.1     loss=0.1698  acc=0.9564
    lr=1.0     loss=0.1278  acc=0.9614
    """

    # print('\n=== 实验 2：优化器 ===')
    # run_specs = [
    #     ('SGD',           dict(opt='sgd', lr=0.1)),
    #     ('SGD+momentum',  dict(opt='sgd', lr=0.1, momentum=0.9)),
    #     ('Adam',          dict(opt='adam', lr=0.001)),
    # ]
    # for name, kwargs in run_specs:
    #     losses, acc = run(**kwargs)
    #     print(f'{name:<14}  loss={losses[-1]:.4f}  acc={acc:.4f}')

    """
        === 实验 2：优化器 ===
    SGD             loss=0.1698  acc=0.9564
    SGD+momentum    loss=0.0811  acc=0.9673
    Adam            loss=0.1094  acc=0.9685
    """

    # print('\n=== 实验 3：batch_size（固定 lr=0.1, SGD, 3 epoch）===')
    # for batch_size in [8, 64, 512]:
    #     losses, acc = run(batch_size=batch_size)
    #     print(f'batch_size={batch_size:<4}  loss={losses[-1]:.4f}  acc={acc:.4f}')

    # """补充实验 A（2 分钟）：把学习率推到真的崩"""
    # print('\n=== 补充实验 A（2 分钟）：把学习率推到真的崩===')
    # for lr in [1.0, 3.0, 10.0, 30.0]:
    #     losses, acc = run(lr=lr, epochs=3)
    #     print(f'lr={lr:<5}  losses={[f"{x:.3f}" for x in losses]}  acc={acc:.4f}')

    """
        === 补充实验 A（2 分钟）：把学习率推到真的崩===
    lr=1.0    losses=['0.368', '0.156', '0.128']  acc=0.9614
    lr=3.0    losses=['2.383', '2.314', '2.316']  acc=0.1028
    lr=10.0   losses=['10.028', '2.384', '2.379']  acc=0.0980
    lr=30.0   losses=['26498.498', '6.243', '6.239']  acc=0.1009"""

    print('=== 补充实验：固定更新次数 ≈ 7500 次 ===')
    for bs, ep in [(8, 1), (64, 8), (512, 64)]:
        losses, acc = run(batch_size=bs, epochs=ep)
        print(f'batch={bs:<4} epochs={ep:<3} '
            f'更新={60000//bs*ep:<6} loss={losses[-1]:.4f} acc={acc:.4f}')