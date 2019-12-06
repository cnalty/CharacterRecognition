import torch.utils.data
from torch import nn
from torch.autograd import Variable
from Dataset import Mnist_Dataset
from model import CharNet
import torch.optim.lr_scheduler

BATCH_SIZE = 2048
NUM_EPOCHS = 30
LR_DECAY = 10


def main():
    # Load model, dataset and set up gradient decent
    model = CharNet(10)
    model.cuda()
    print(model)


    optimizer = torch.optim.SGD(
        model.parameters(),
        lr = .01,
        momentum = 0.9,
        weight_decay=1e-5
    )

    dataset = Mnist_Dataset('train.csv')

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )



    criterion = nn.CrossEntropyLoss()
    criterion.cuda()

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, LR_DECAY, gamma=0.1)

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        train(model, optimizer, train_loader, criterion)
        scheduler.step()

    # Save the model
    torch.save(model.state_dict(), "wieghts.pth.tar")



def train(model, optimizer, train_loader, criterion):
    model.train()

    for batch_num, (data, label) in enumerate(train_loader):
        data = data.cuda()
        label = label.cuda()

        output = model(data)

        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (batch_num + 1) % 10 == 0:
            print("loss is " + str(loss.item()))






if __name__ == "__main__":
    main()
