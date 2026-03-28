import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets

resize = (32,32)

transform_train_list = transforms.Compose([
    transforms.Resize(resize,interpolation=InterpolationMode.BICUBIC),
    transforms.RandomCrop(32),
    transforms.ToTensor()
])

transform_test_list = transforms.Compose([
    transforms.Resize(resize,interpolation=InterpolationMode.BICUBIC),
    transforms.RandomCrop(32),
    transforms.ToTensor()
])

mnist_train = torchvision.datasets.MNIST(root='./mnist_data/',train=True,download=True,transform=transform_train_list)
mnist_test = torchvision.datasets.MNIST(root='./mnist_data/',train=False,download=True,transform=transform_test_list)

mnist_dataloaders = {
    'train': torch.utils.data.DataLoader(mnist_train,batch_size=128,shuffle=True),
    'test': torch.utils.data.DataLoader(mnist_test,batch_size=128,shuffle=False)
}

mnist_size = {
    'train': len(mnist_train),
    'test': len(mnist_test)
}

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1,stride=1,bias=False)
        self.conv2 = nn.Conv2d(32,64,3,1,1,bias=False)
        self.conv3 = nn.Conv2d(64,128,3,1,1,bias=False)
        self.batch1 = nn.BatchNorm2d(32)
        self.batch2 = nn.BatchNorm2d(64)
        self.batch3 = nn.BatchNorm2d(128)
        self.fc = nn.LazyLinear(10)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.batch2(x)
        x = F.relu(x)
        
        x = self.conv3(x)
        x = self.batch3(x)
        x = F.relu(x)
        
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        
        return x
    
mnist_model = CNN().cuda()

mnist_optimizer = optim.SGD(mnist_model.parameters(),lr=0.01)
mnist_scheduler = CosineAnnealingLR(mnist_optimizer, T_max=100)
criterion = nn.CrossEntropyLoss()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model,dataloaders,train_epoch,loss_fn,optimizer,scheduler,size):
    loss_li = {'train':[],'test':[]}
    acc_li = {'train':[],'test':[]}
    
    for epoch in range(train_epoch):
        for phase in ['train','test']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)
                
            running_loss = 0.0
            running_corrects = 0.0   
            
            
                
            for data in dataloaders[phase]:
                inputs,label = data
                inputs,label = inputs.to(device), label.to(device)
                
                if phase == 'train':
                    optimizer.zero_grad()
                    
                model_output = model(inputs)
                _,preds = torch.max(model_output.data,1)
                loss = loss_fn(model_output,label)
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                running_loss += loss.item()
                running_corrects += torch.sum(preds == label.data).item()
                
            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects / size[phase]
            print(f'epoch :{epoch}')
            print(f'{phase} Loss:{epoch_loss:.6f} // Acc:{100*epoch_acc:.4f}%\n') 
            loss_li[phase].append(epoch_loss)
            acc_li[phase].append(100*epoch_acc)
        scheduler.step()
        
    return loss_li,acc_li

train(mnist_model,mnist_dataloaders,3,criterion,mnist_optimizer,mnist_scheduler,mnist_size)