import torch
import torchvision
import torchvision.transforms as transforms

def trainloader(batch_size = 64):
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./', train=True,
                                            download=False, transform=transform_train)
    
    return  torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

def testloader(batch_size = 64):
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./', train=False,
                                           download=False, transform=transform_test)
    
    return torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)