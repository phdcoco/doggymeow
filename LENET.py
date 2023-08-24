import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms #이미지 전처리 라이브러리
import torchvision.datasets as datasets
import torchvision.models as models

import matplotlib.pyplot as plt
import numpy as np
import copy
from collections import namedtuple #키 값으로 데이터 접근
import os #파일 경로 관련 라이브러리
import random
import time

import cv2
from torch.utils.data import DataLoader, Dataset
from PIL import Image

os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train' : transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5,1.0)), #resize는 224x224 크기로, scale은 50~100%만큼 면적을 무작위로 자름
                transforms.RandomHorizontalFlip(), #수평 반전 기본값 50%
                transforms.ToTensor(), #0~255사이 값을 가지는 픽셀 값이 0~1 사이가 되고 차원의 순서도 바뀜, 추후 연산에 용이하게 바꾸는 것.
                transforms.Normalize(mean, std) #채널별 평균과 표준편차, 이것은 쓰는 모델에 따라 다르다.
                ]),
            'val' : transforms.Compose([
                transforms.Resize(256), #이미지의 짧은 부분을 256픽셀 길이로 조정한다. 가로 세로 비율은 유지하면서!
                transforms.CenterCrop(resize), #가운데 부분을 중심으로 256픽셀을 제외하고 자른다. 즉 정사각형 사진을 만드는 것임.
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
                ])
            }
        
    def __call__(self, img, phase):
        return self.data_transform[phase](img) #call은 함수를 그냥 실행만 해도 저절로 출력까지 해주는 역할임

size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
batch_size = 32

cat_directory = r'C:/Users/dh-ry/Desktop/dogvscat/train/cat' #400개의 이미지
dog_directory = r'C:/Users/dh-ry/Desktop/dogvscat/train/dog/' #400개의 이미지
test_directory = r'C:/Users/dh-ry/Desktop/dogvscat/funny test/test1'

cat_images_filepaths = sorted([os.path.join(cat_directory, f) for f in os.listdir(cat_directory)]) #이미지 경로와 파일명을 붙여서 리스트로 반환
dog_images_filepaths = sorted([os.path.join(dog_directory, f) for f in os.listdir(dog_directory)])
test_images_filepaths = sorted([os.path.join(test_directory, f) for f in os.listdir(test_directory)])
images_filepaths = [*cat_images_filepaths, *dog_images_filepaths] #*은 ,로 나뉘어진 리스트 원소들을 하나하나 분리해준다. 즉 이렇게 하면 하나의 리스트에 두 개의 리스트 원소들을 합칠 수 있음.
test_filepaths = [*test_images_filepaths]
correct_images_filepaths = [i for i in images_filepaths if cv2.imread(i) is not None] #그림 파일을 더 이상 불러올 수 없을 때까지 불러옴.
test_images_filepaths = [j for j in test_filepaths if cv2.imread(j) is not None]

random.seed(159)
random.shuffle(correct_images_filepaths)
random.shuffle(test_images_filepaths)
train_images_filepaths = correct_images_filepaths[:600]
val_images_filepaths = correct_images_filepaths[:-600]


print(len(train_images_filepaths), len(val_images_filepaths), len(test_images_filepaths))

class DogvsCatDataset(Dataset):
    def __init__(self, file_list, transform, phase):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)#이미지 데이터를 가져옴
        img_transformed = self.transform(img, self.phase)
        label = img_path.split('/')[-1].split('.')[0]#img_path는 파일 전체 경로, /기준으로 -1만큼 버리기, .기준으로 0의 값 -> 이미지 이름인 dog or cat 추출
        if label == 'dog':
            label = 1
        elif label == 'cat':
            label = 0
        return img_transformed, label #변환된 이미지와 레이블 값을 반환

train_dataset = DogvsCatDataset(train_images_filepaths, transform=ImageTransform(size, mean, std), phase='train')
val_dataset = DogvsCatDataset(val_images_filepaths, transform=ImageTransform(size, mean, std), phase='val')

index = 0 #첫 번째 이미지
print(train_dataset.__getitem__(index)[0].size()) #return의[0]값인 변환된 이미지의 사이즈를 출력
print(train_dataset.__getitem__(index)[1]) #return의[1]값인 label을 출력

train_iterator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_iterator = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
dataloader_dict = {'train': train_iterator, 'val': valid_iterator} #훈련 검증 데이터셋을 합침

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0) 
        self.relu1 = nn.ReLU() 
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) 
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0) 
        self.relu2 = nn.ReLU() # activation
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)         
        self.fc1 = nn.Linear(32*53*53, 512) 
        self.relu5 = nn.ReLU()         
        self.fc2 = nn.Linear(512, 2) 
        self.output = nn.Softmax(dim=1)        
    
    def forward(self, x):
        out = self.cnn1(x) 
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out) 
        out = self.relu2(out) 
        out = self.maxpool2(out) 
        out = out.view(out.size(0), -1) 
        out = self.fc1(out) 
        out = self.fc2(out)                    
        out = self.output(out)
        return out
    
model = LeNet()
    
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)

def train_model(model, dataloader_dict, criterion, optimizer, num_epoch):    
    since = time.time()
    best_acc = 0.0
    
    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch + 1, num_epoch))
        print('-'*20)
        
        for phase in ['train', 'val']:           
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            epoch_loss = 0.0
            epoch_corrects = 0
            
            for inputs, labels in tqdm(dataloader_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)
                    
            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloader_dict[phase].dataset)
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return model

import time

num_epoch = 10
model = train_model(model, dataloader_dict, criterion, optimizer, num_epoch)

import pandas as pd
id_list = []
pred_list = []
_id=0
with torch.no_grad():
    for test_path in tqdm(test_images_filepaths):
        img = Image.open(test_path)
        _id =test_path.split('/')[-1].split('.')[1]
        transform = ImageTransform(size, mean, std)
        img = transform(img, phase='val')
        img = img.unsqueeze(0)
        img = img.to(device)

        model.eval()
        outputs = model(img)
        preds = F.softmax(outputs, dim=1)[:, 1].tolist()        
        id_list.append(_id)
        pred_list.append(preds[0])
       
res = pd.DataFrame({
    'id': id_list,
    'label': pred_list
})

res.sort_values(by='id', inplace=True)
res.reset_index(drop=True, inplace=True)

res.to_csv('C:/Users/dh-ry/Desktop/dogvscat/answer.csv', index=False)

class_ = classes = {0:'cat', 1:'dog'}
def display_image_grid(images_filepaths, predicted_labels=(), cols=5):
    rows = len(images_filepaths) // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i, image_filepath in enumerate(images_filepaths):
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        a = random.choice(res['id'].values)    
        label = res.loc[res['id'] == a, 'label'].values[0]
        if label > 0.5:
            label = 1
        else:
            label = 0
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_title(class_[label])
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()

display_image_grid(test_images_filepaths) 