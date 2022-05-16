from __future__ import print_function
import argparse
import ct101
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import network
from dataloader import get_dataloader
from torch.utils.data import Dataset
import torch.utils.data as Data
import random
import numpy as np
import os
import PIL
from PIL import Image
import copy
import random
from torch.utils.data import ConcatDataset
import time
import torch.nn.utils.prune as prune

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if False and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader, cur_epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    #print('Epoch {} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        #cur_epoch, test_loss, correct, len(test_loader.dataset),
        #100. * correct / len(test_loader.dataset)))
    return correct/len(test_loader.dataset)
   
def masked_test(args,model,device,v,loader):
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for data,target in loader:
            noised_data=copy.deepcopy(data)
            noised_data=noised_data.to(torch.device("cpu"))
            noised_data=noised_data+torch.normal(0,v,data.shape)
            noised_data,target=noised_data.to(device),target.to(device)
            output=model(noised_data)
            test_loss+=F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(loader.dataset)
    return correct/len(loader.dataset)
 
 
def get_model(args):
    if args.model.lower()=='lenet5':
        return network.lenet.LeNet5()
    elif args.model.lower()=='resnet34':
        return  torchvision.models.resnet34(num_classes=args.num_classes, pretrained=args.pretrained)
    elif args.model.lower()=='resnet34_8x':
        return network.resnet_8x.ResNet34_8x(num_classes=args.num_classes)

# Parsing arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 1000)')
parser.add_argument('--data_root', type=str, default='data')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dataset', type=str, default='cifar100', choices=['mnist', 'cifar100', 'svhn', 'cifar10', 'caltech101', 'nyuv2'],
                        help='dataset name (default: mnist)')
parser.add_argument('--model', type=str, default='resnet34_8x', choices=['lenet5', 'resnet34', 'resnet34_8x'],
                        help='model name (default: mnist)') 
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
parser.add_argument('--step_size', type=int, default=50, metavar='S',
                        help='random seed (default: 1)')
parser.add_argument('--ckpt', type=str, default="checkpoint/teacher/cifar100-resnet34_8x.pt")
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
parser.add_argument('--test_only', action='store_true', default=False)
parser.add_argument('--download', action='store_true', default=False)
parser.add_argument('--pretrained', action='store_true', default=False)
parser.add_argument('--scheduler', action='store_true', default=False)
parser.add_argument('--verbose', action='store_true', default=False)

# Initializing.
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

#torch.manual_seed(args.seed)
#torch.cuda.manual_seed(args.seed)
#np.random.seed(args.seed)
#random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Preparing triggers (I).
def gen_I(scheme,N):
    ls=np.random.randint(0,args.num_classes,size=N)
    us=torch.rand(N,256,1,1).to(device)
    if scheme==1:
        Is=10*torch.rand(N,3,32,32).to(device)
        Is=Is.detach()
        Is.requires_grad=False
    if scheme==2:
        Is=50*generator2(us)
        #Is=20*generator2(us)
        Is=Is.detach()
        Is.requires_grad=False
    if scheme==3:
        Is=generator(us)
        Is=Is.detach()
        Is.requires_grad=False
    if scheme==4:
        Is=torch.rand(N,3,32,32).to(device)
        Is=Is.detach()
        Is.requires_grad=False
        for n in range(N):
            r=random.randint(0,1023)
            cc=random.randint(0,2)
            a=int(r/32)
            b=int(r%32)
            Is[n][cc][a][b]=2000
            #Is[n][0][a][b]=15
    return Is,ls

class Itriggers(Dataset):
    def __init__(self,Is_,ls_):
        self.I=Is_
        self.l=ls_
    def __getitem__(self,index):
        img=self.I[index]
        label=self.l[index]
        return img,label
    def __len__(self):
        return len(self.I)

def no_data(model,E,t_loader,verbose):
    #E=1800
    local_model=copy.deepcopy(model)
    local_model=local_model.to(device)
    local_optimizer=optim.Adam(local_model.parameters(),lr=0.000002)
    hm=[]
    ht=[]
    for epoch in range(0,E):
        train(args,local_model,device,t_loader,local_optimizer,epoch)
        if epoch%(int(E/50))==0 and verbose:
            mnist_acc=test(args,local_model,device,test_loader,epoch)
            trigger_acc=test(args,local_model,device,t_loader,epoch)
            print("Test branch 1/6, epoch=%i, primary_acc=%f,trigger_acc=%f"%(epoch,mnist_acc,trigger_acc))
            hm.append(mnist_acc)
            ht.append(trigger_acc)
    print("ms.append(",end="")
    print(hm,end="")
    print(")")
    print("ts.append(",end="")
    print(ht,end="")
    print(")")
    return local_model

def mnist(model,E,t_loader,verbose):
    #E=3
    local_model=copy.deepcopy(model)
    local_model=local_model.to(device)
    local_optimizer=optim.Adam(local_model.parameters(),lr=0.000005)
    hm=[]
    ht=[]
    for epoch in range(0,E):
        for batch_idx, (data, target) in enumerate(train_loader):
            #train_loader is global
            data,target=data.to(device),target.to(device)
            local_optimizer.zero_grad()
            output=local_model(data)
            loss=F.cross_entropy(output,target)
            loss.backward()
            local_optimizer.step()
            for i in range(1):
                train(args,local_model,device,t_loader,local_optimizer,epoch)
            if batch_idx%100==0 and verbose:
                mnist_acc=test(args,local_model,device,test_loader,epoch)
                trigger_acc=test(args,local_model,device,t_loader,epoch)
                print("Test branch 2/6, epoch=%i, primary_acc=%f,trigger_acc=%f"%(epoch,mnist_acc,trigger_acc))
                hm.append(mnist_acc)
                ht.append(trigger_acc)
    print("ms.append(",end="")
    print(hm,end="")
    print(")")
    print("ts.append(",end="")
    print(ht,end="")
    print(")")
    return local_model

def DFD(model,generator,E,t_loader,t,verbose):
    #E=4000
    local_model=copy.deepcopy(model)
    local_model=local_model.to(device)
    #0.00002
    local_optimizer=optim.Adam(local_model.parameters(),lr=0.00002)
    hm=[]
    ht=[]
    S=5*N
    l2=6
    best_acc=test(args,local_model,device,test_loader,0)
    start=time.time()
    for epoch in range(0,E):
        local_optimizer.zero_grad()
        #z=torch.rand((10*N,100,1,1)).to(device)
        z=torch.rand((S,256,1,1)).to(device)
        fake=generator(z).detach()
        #generator is global
        y=model(fake).detach()
        #model is global
        ty=local_model(fake)
        loss=l2*F.l1_loss(y,ty)
        triggers_op=local_model(t.I)
        #print(torch.tensor(I.l).shape)
        loss=loss+F.cross_entropy(triggers_op,torch.tensor(t.l).to(device))
        loss.backward()
        local_optimizer.step()
        trigger_acc=test(args,local_model,device,t_loader,0)
        if trigger_acc>=0.9:
            end=time.time()
            elapse=end-start
            now_acc=test(args,local_model,device,test_loader,0)
            diff=best_acc-now_acc
            print("m.append(%f)"%(100*diff))
            print("tt.append(%f)"%elapse)
            break
        #if epoch%20==0:
            #train(args,test_model_3,device,I_loader,test_optimizer_3,epoch)
        if epoch%(int(E/20))==0 and verbose:
            mnist_acc=test(args,local_model,device,test_loader,epoch)
            trigger_acc=test(args,local_model,device,t_loader,epoch)
            print("Test branch 3/6, epoch=%i, primary_acc=%f,trigger_acc=%f"%(epoch,mnist_acc,trigger_acc))
            hm.append(mnist_acc)
            ht.append(trigger_acc)
    #print("ms.append(",end="")
    #print(hm,end="")
    #print(")")
    #print("ts.append(",end="")
    #print(ht,end="")
    #print(")")
    return local_model

def ItoT(I,Q,EE,EEE,lam,verbose=False):
    # I->T
    original_dataset=copy.deepcopy(I)
    current_dataset=copy.deepcopy(I)
    for q in range(Q):
        #if verbose and q%(int(Q/10))==0:
            #print("ItoT round %i in %i"%(q,Q))
        model_q=copy.deepcopy(model)
        optimizer_q=optim.Adam(model_q.parameters(),lr=0.00008)
        current_loader=Data.DataLoader(
            dataset=current_dataset,
            batch_size=30,
            shuffle=True)
        for epoch in range(EE):
            z=torch.rand((args.batch_size,256,1,1)).to(device)
            fake=generator(z).detach()
            y=model(fake).detach()
            ty=model_q(fake)
            loss=F.l1_loss(y,ty)
            loss.backward()
            optimizer_q.step()
            train(args,model_q,device,current_loader,optimizer_q,epoch)
        for param in model_q.parameters():
            param.requires_grad=False
        T=copy.deepcopy(current_dataset.I)
        T.requires_grad=True
        optimizer_T=optim.Adam([T],lr=0.00008)
        for epoch in range(EEE):
            op=model_q(T)
            loss=F.cross_entropy(op,torch.tensor(I.l).to(device))
            loss=loss+lam*F.l1_loss(T,I.I)
            loss.backward()
            optimizer_T.step()
        current_dataset.I=T
    print("Trigger set T launched.")

    T_loader=Data.DataLoader(
        dataset=current_dataset,
        batch_size=30,
        shuffle=True)
    return current_dataset,T_loader

scheme=3
N=50
test_branch=4
device=torch.device("cuda:3")

#device = torch.device("cuda" if use_cuda else "cpu")
#os.makedirs('checkpoint/teacher',exist_ok=True)
kwargs={'num_workers':1, 'pin_memory':True} if use_cuda else {}
print(args)
train_loader,test_loader=get_dataloader(args)
model=get_model(args)
if args.ckpt is not None:
    model.load_state_dict(torch.load(args.ckpt,map_location=device))    
model=model.to(device)
optimizer=optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
best_acc=0
if args.scheduler:
    scheduler=optim.lr_scheduler.StepLR(optimizer, args.step_size, 0.1)

# Model loaded.
generator=network.gan.GeneratorA(nz=256,nc=3,img_size=32)
generator.load_state_dict(torch.load("./checkpoint/student/cifar100-resnet18_8x-generator.pt"))
generator.eval()
generator=generator.to(device)
# Generator loaded.
# For scheme2.
generator2=network.gan.GeneratorA(nz=256,nc=3,img_size=32)
generator2.eval()
generator2=generator2.to(device)

Is,ls=gen_I(scheme,N)
I=Itriggers(Is,ls)
I_loader=Data.DataLoader(
    dataset=I,
    batch_size=30,
    shuffle=True)

# Triggers (I) generated.
baselineacc=test(args,model,device,test_loader,0)
print("--------------------Baseline:%f--------------------"%baselineacc)
print("Scheme=%i"%scheme)
print("Test branch=%i"%test_branch)
print("N=%i"%N)

#Main test process.
# 1:empty+I
# 2:empty+T
# 3:DFD+I
# 4:DFD+T
# 5:train+I
# 6:train+T

if test_branch==0:
    m=mnist_acc=test(args,model,device,test_loader,0)
    print(m)
if test_branch==1:
    no_data(model,2000,I_loader,True)
if test_branch==2:
    T,T_loader=ItoT(I,30,20,20,0.15,True)
    no_data(model,1000,T_loader,True)
if test_branch==3:
    #1,2 2000
    #3 300
    DFD(model,generator,200,I_loader,I,False)
if test_branch==4:
    Q=40
    start=time.time()
    T,T_loader=ItoT(I,Q,20,20,0.05,True)
    end=time.time()
    elapse=end-start
    print("Q=%i"%Q)
    print("tg.append(%f)"%elapse)    
    DFD(model,generator,200,T_loader,T,False)
if test_branch==5:
    mnist(model,30,I_loader,True)
if test_branch==6:
    T,T_loader=ItoT(I,30,20,20,0.05,True)
    mnist(model,30,T_loader,True)
