import torch
from torchvision import datasets, transforms
import torchvision
from tqdm import tqdm

device_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 可用GPU
BATCH_SIZE = 64

transform = transforms.Compose([transforms.ToTensor()])
data_train = datasets.MNIST(root="./data/",
                            transform=transform,
                            train=True,
                            download=True)
data_test = datasets.MNIST(root="./data/",
                           transform=transform,
                           train=False)

data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                # 单卡batch size * 卡数
                                                batch_size=BATCH_SIZE * len(device_ids),
                                                shuffle=True,
                                                num_workers=2)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size=BATCH_SIZE * len(device_ids),
                                               shuffle=True,
                                               num_workers=2)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2),
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(14 * 14 * 128, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14 * 14 * 128)
        x = self.dense(x)
        return x


model = Model()
# 指定要用到的设备
model = torch.nn.DataParallel(model, device_ids=device_ids)
# 模型加载到设备0
model = model.cuda(device=device_ids[0])

cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
from time import sleep

n_epochs = 50
for epoch in range(n_epochs):
    running_loss = 0.0
    running_correct = 0
    print("Epoch {}/{}".format(epoch, n_epochs))
    print("-" * 10)
    for data in tqdm(data_loader_train):
        X_train, y_train = data
        # 指定设备0
        X_train, y_train = X_train.cuda(device=device_ids[0]), y_train.cuda(device=device_ids[0])
        outputs = model(X_train)
        _, pred = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        loss = cost(outputs, y_train)

        loss.backward()
        optimizer.step()
        running_loss += loss.data.item()
        running_correct += torch.sum(pred == y_train.data)
    testing_correct = 0
    for data in data_loader_test:
        X_test, y_test = data
        # 指定设备1
        X_test, y_test = X_test.cuda(device=device_ids[0]), y_test.cuda(device=device_ids[0])
        outputs = model(X_test)
        _, pred = torch.max(outputs.data, 1)
        testing_correct += torch.sum(pred == y_test.data)
    print("Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(
        torch.true_divide(running_loss, len(data_train)),
        torch.true_divide(100 * running_correct, len(data_train)),
        torch.true_divide(100 * testing_correct, len(data_test))))
torch.save(model.state_dict(), "model_parameter.pkl")