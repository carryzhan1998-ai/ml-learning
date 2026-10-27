import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


transform = transforms.ToTensor()
train_ds = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=1000)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.sequential = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),          nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(400, 120), nn.ReLU(),
            nn.Linear(120, 84),  nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.sequential(x)
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_epoch = 10                   # the number of training epoch
learning_rate = 1e-4         # learning rate
model_path = './model.ckpt'     # the path where the checkpoint will be saved

for learning_rate in [1e-5, 1e-4, 1e-3, 1e-2]:
    model = LeNet().to(device)
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0.0
    best_val_loss = float('inf')  # 新增：记录最小验证loss
    best_acc_epoch = -1
    best_loss_epoch = -1

    for epoch in range(num_epoch):
        train_acc = 0.0
        train_loss = 0.0
        test_acc = 0.0
        test_loss = 0.0
        
        # training
        model.train() # set the model to training mode
        for i, batch in enumerate(train_loader):
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad() 
            outputs = model(features) 
            
            loss = criterion(outputs, labels)
            loss.backward() 
            optimizer.step() 
            
            _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
            train_acc += (train_pred.detach() == labels.detach()).sum().item()
            train_loss += loss.item()
        
        # validation
        model.eval() # set the model to evaluation mode
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)
                
                loss = criterion(outputs, labels) 
                
                _, test_pred = torch.max(outputs, 1) 
                test_acc += (test_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
                test_loss += loss.item()

       # 计算本轮指标
        epoch_train_acc = train_acc / len(train_ds)
        epoch_train_loss = train_loss / len(train_loader)
        epoch_val_acc = test_acc / len(test_ds)
        epoch_val_loss = test_loss / len(test_loader)

        print(f'[{epoch+1:03d}/{num_epoch:03d}] Train Acc: {epoch_train_acc:3.5f} Loss: {epoch_train_loss:3.5f} | Val Acc: {epoch_val_acc:3.5f} loss: {epoch_val_loss:3.5f}')

        # 判断最优acc
        if test_acc > best_acc:
            best_acc = test_acc
            best_acc_epoch = epoch + 1
            torch.save(model.state_dict(), model_path)
            print(f'>>> saving model with acc {epoch_val_acc:.5f}')
        # 判断最小val loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_loss_epoch = epoch + 1

    # 这一组lr跑完，输出这一组的最高acc、最低val loss
    print("="*60)
    print(f'lr = {learning_rate:.0e} | 该组最高Val Acc = {best_acc/len(test_ds):.5f} (epoch {best_acc_epoch})')
    print(f'lr = {learning_rate:.0e} | 该组最低Val Loss = {best_val_loss:.5f} (epoch {best_loss_epoch})')
    print("="*60 + "\n")

#   对比不同神经网络的参数数量
"""def count_params(model):
    return sum(p.numel() for p in model.parameters())

mlp = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
lenet = LeNet()

print(f'MLP   参数量: {count_params(mlp):,}')
print(f'LeNet 参数量: {count_params(lenet):,}')"""

