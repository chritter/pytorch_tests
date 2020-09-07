
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np

# this allows to run torch.utils.bottleneck main.py with workers>0
torch.multiprocessing.set_start_method('spawn', force=True)

torch.set_default_tensor_type('torch.cuda.FloatTensor')



class Net(nn.Module):
    def __init__(self, batchnorm=False):
        super(Net, self).__init__()

        units = 100
        num_classes = 10
        self.fc1 = nn.Linear(784, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.fc4 = nn.Linear(units, num_classes)

        self.bn= batchnorm

        if self.bn:
            self.bn1 = nn.BatchNorm1d(units)
            self.bn2 = nn.BatchNorm1d(units)
            self.bn3 = nn.BatchNorm1d(units)


        self.save_activation_output = []

    def forward(self, x):
        
        x = torch.flatten(x, 1)

        # layer 1
        x = self.fc1(x)
        if self.bn: x = self.bn1(x)
        x = F.sigmoid(x)

        # layer 2
        x = self.fc2(x)
        if self.bn: x = self.bn2(x)
        x = F.sigmoid(x)

        # layer 3
        x = self.fc3(x)
        if self.bn: x = self.bn3(x)
        x = F.sigmoid(x)
        # take value from (non-leaf) node parameter 

        q = np.array([0, 0.5, 1])

        self.save_activation_output.append(np.quantile(x[:,0].detach().cpu().numpy(), [0, 0.5, 1]))


        # final layer
        x = self.fc4(x)

        output = F.log_softmax(x, dim=1)

        return output



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # set zero: {'eps': 1e-06, 'initial_lr': 1.0, 'lr': 1.0, 'params': model_params, 'rho': 0.9, 'weight_decay': 0}]
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 2,
                       'pin_memory': False,
                       'shuffle': True},
                     )
        print('use codea')

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)


    model = Net(batchnorm=False).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Decays the learning rate of each parameter group by gamma every epoch
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        scheduler.step()

    model_bn = Net(batchnorm=True).to(device)
    optimizer = optim.Adadelta(model_bn.parameters(), lr=args.lr)

    # Decays the learning rate of each parameter group by gamma every epoch
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model_bn, device, train_loader, optimizer, epoch)
        scheduler.step()



    print('test')
    import matplotlib.pyplot as plt
    plt.plot(range(len(model.save_activation_output)),model.save_activation_output, label='without BN', marker='x', markevery=64)
    plt.plot(range(len(model_bn.save_activation_output)),model_bn.save_activation_output, label='with BN', marker = 'o', markevery=64)
    plt.title("15, 50, 85% percentile")
    plt.legend()
    plt.show()

    a  =3 


    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()

