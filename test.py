import os
import numpy as np
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from torchvision import datasets

device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
attack 함수 정의
"""


def fgsm_targeted(model,x,target,eps=0.3):
    
    x_adv = x.detach().clone().requires_grad_(True)
    loss_fn = nn.CrossEntropyLoss()
    
    logits = model(x_adv)
    loss = loss_fn(logits,target)
    model.zero_grad()
    loss.backward()
    
    with torch.no_grad():
        x_adv = x_adv - eps * torch.sign(x_adv.grad)
        x_adv = torch.clamp(x_adv,min=0,max=1)
        
    return x_adv

def fgsm_untargeted(model,x,label,eps=0.3):
    x_adv = x.detach().clone().requires_grad_(True)
    loss_fn = nn.CrossEntropyLoss()
    
    logits = model(x_adv)
    loss = loss_fn(logits,label)
    model.zero_grad()
    loss.backward()
    
    with torch.no_grad():
        x_adv = x_adv + eps * torch.sign(x_adv.grad)
        x_adv = torch.clamp(x_adv,min=0,max=1)
    
    return x_adv

def pgd_targeted(model,x,label,k=40,eps_step=0.01,eps=0.3):
    x_adv = x.detach().clone()
    
    for i in range(k):
        x_adv.requires_grad_(True)
        x_adv = fgsm_targeted(model,x_adv,label,eps_step)
        
        with torch.no_grad():
            x_adv = torch.clamp(x_adv,x-eps,x+eps) #x 범위가 [x-ε,x+ε]사이에 있을 수 있게 clip
            x_adv = torch.clamp(x_adv,0,1)
        
    return x_adv

def pgd_untargeted(model,x,target,k=40,eps_step=0.01,eps=0.3):
    x_adv = x.detach().clone()
    
    for i in range(k):
        x_adv.requires_grad_(True)
        x_adv = fgsm_untargeted(model,x_adv,target,eps_step)
        
        with torch.no_grad():
            x_adv = torch.clamp(x_adv,x-eps,x+eps)
            x_adv = torch.clamp(x_adv,0,1)
        
    return x_adv

"""
모델 준비 및 train
mnist에는 Custom CNN
cifar10에는 convnext_Tiny를 사용
"""

resize = (32,32)

mnist_transform_train_list = transforms.Compose([
    transforms.Resize(resize,interpolation=InterpolationMode.BICUBIC),
    transforms.RandomCrop(32),
    transforms.ToTensor()
])

mnist_transform_test_list = transforms.Compose([
    transforms.Resize(resize,interpolation=InterpolationMode.BICUBIC),
    transforms.RandomCrop(32),
    transforms.ToTensor()
])

cifar10_transform_train_list = transforms.Compose([
    transforms.Resize(resize,interpolation=InterpolationMode.BICUBIC),
    transforms.RandomCrop(32),
    transforms.ToTensor()
])

cifar10_transform_test_list = transforms.Compose([
    transforms.Resize(resize,interpolation=InterpolationMode.BICUBIC),
    transforms.RandomCrop(32),
    transforms.ToTensor()
])

mnist_train = torchvision.datasets.MNIST(root='./mnist_data/',train=True,download=True,transform=mnist_transform_train_list)
mnist_test = torchvision.datasets.MNIST(root='./mnist_data/',train=False,download=True,transform=mnist_transform_test_list)

cifar10_train = torchvision.datasets.CIFAR10(root='./cifar10_data/',train=True,download=True,transform=cifar10_transform_train_list)
cifar10_test = torchvision.datasets.CIFAR10(root='./cifar10_data/',train=False,download=True,transform=cifar10_transform_test_list)

mnist_dataloaders = {
    'train': torch.utils.data.DataLoader(mnist_train,batch_size=128,shuffle=True),
    'test': torch.utils.data.DataLoader(mnist_test,batch_size=128,shuffle=False)
}

mnist_size = {
    'train': len(mnist_train),
    'test': len(mnist_test)
}

cifar10_dataloaders = {
    'train': torch.utils.data.DataLoader(cifar10_train,batch_size=128,shuffle=True),
    'test': torch.utils.data.DataLoader(cifar10_test,batch_size=128,shuffle=False)
}

cifar10_size = {
    'train': len(cifar10_train),
    'test': len(cifar10_test)
}

#Cutsom CNN정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1,stride=1,bias=False)
        self.conv2 = nn.Conv2d(32,64,3,1,1,bias=False)
        self.conv3 = nn.Conv2d(64,128,3,1,1,bias=False)
        self.batch1 = nn.BatchNorm2d(32)
        self.batch2 = nn.BatchNorm2d(64)
        self.batch3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2,2)
        self.fc = nn.LazyLinear(10)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.batch2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.batch3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        
        return x

mnist_model = CNN().cuda()

cifar10_model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
cifar10_model.classifier[2] = nn.Linear(cifar10_model.classifier[2].in_features,10)
cifar10_model = cifar10_model.cuda()

mnist_optimizer = optim.SGD(mnist_model.parameters(),lr=0.01)
mnist_scheduler = CosineAnnealingLR(mnist_optimizer, T_max=100)
criterion = nn.CrossEntropyLoss()

cifar10_optimizer = optim.SGD(cifar10_model.parameters(),lr=0.01)
cifar10_scheduler = CosineAnnealingLR(cifar10_optimizer, T_max=100)
criterion = nn.CrossEntropyLoss()


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
train(cifar10_model,cifar10_dataloaders,5,criterion,cifar10_optimizer,cifar10_scheduler,cifar10_size)

"""
attack simulation 함수 정의
"""

def attack_simulation(model,dataloaders,attack,dataset,eps,device,sample_size=100):
    model.eval()
    success = 0 #라벨을 맞춘 데이터의 수
    tot = 0 #총 데이터의 수
    image_created = False #이미지 생성여부
    
    for data in dataloaders['test']:
        inputs,label = data
        inputs,label = inputs.to(device), label.to(device)
        target = torch.full_like(label,7).to(device) #target을 7로 설정
        target[label == 7] = 0 # label이 7인 경우 target을 0으로 변경
        
        #orignal 데이터에 대한 예측
        with torch.no_grad():
            ori_output = model(inputs)
            _,ori_preds = torch.max(ori_output.data,1)
        
        if attack == 'fgsm_targeted':
            x_adv = fgsm_targeted(model,inputs,target,eps)
            #cls 1은 targeted 0은 untargeted
            cls = 1
        
        elif attack == 'fgsm_untargeted':
            x_adv = fgsm_untargeted(model,inputs,label,eps)
            cls = 0
        
        elif attack == 'pgd_targeted':
            x_adv = pgd_targeted(model,inputs,target,k=5,eps=eps)
            cls = 1
        
        elif attack == 'pgd_untargeted':
            x_adv = pgd_untargeted(model,inputs,label,k=5,eps=eps)
            cls = 0
        
        #attack을 적용한 데이터에 대한 예측
        with torch.no_grad():
            model_output = model(x_adv)
            _,preds = torch.max(model_output.data,1)
        
            if cls == 0:
                success += (preds != label).sum().item()
            else:
                success += (preds == target).sum().item()
            
            tot += label.size(0)
            
        #이미지를 한 번만 생성할 수 있게
        if not image_created:
            #이미지 시각화 및 저장
            for i in range(5):
                #이미지에 맞게 차원을 변경
                ori = inputs[i].cpu().permute(1,2,0).numpy()
                adv = x_adv[i].cpu().permute(1,2,0).numpy()
                pert = np.clip(np.abs(ori-adv)*10,0,1)
            
                fig = plt.figure()
                fig.tight_layout()
            
                ax1 = fig.add_subplot(131)
                ax1.set_title('Original')
                ax1.set_xlabel(f'label:{label[i].item()}/pred:{ori_preds[i].item()}')
                plt.imshow(ori)
            
                ax2 = fig.add_subplot(132)
                ax2.set_title('Adversarial')
                ax2.set_xlabel(f'label:{label[i].item()}/pred:{preds[i].item()}')
                plt.imshow(adv)
            
                ax3 = fig.add_subplot(133)
                ax3.set_title('Perturbation')
                plt.imshow(pert)
                
                save = f'results/{dataset}/{attack}_{i}th_sample.png'
                plt.savefig(save)
                plt.close()
                
            image_created = True
            
        #100개의 샘플에 대해서만 실행
        if tot >= sample_size:
            break    
        
    return 100*success / tot

os.makedirs('results',exist_ok=True)
os.makedirs('results/cifar10',exist_ok=True)
os.makedirs('results/mnist',exist_ok=True)

"""
결과표시
"""
attack_type = ['fgsm_targeted','fgsm_untargeted','pgd_targeted','pgd_untargeted']

mnist_result = {}
cifar10_result = {}

for attack in attack_type:
    result = attack_simulation(mnist_model,mnist_dataloaders,attack,'mnist',0.3,device)
    mnist_result[attack] = result
    
for attack in attack_type:
    result = attack_simulation(cifar10_model,cifar10_dataloaders,attack,'cifar10',0.3,device)
    cifar10_result[attack] = result   
    
print("results")
for attack in attack_type:
    print(f"attack type : {attack}")
    print(f"mnist results: {mnist_result[attack]:.2f},   cifar10 results: {cifar10_result[attack]:.2f}\n\n")