## QUESTİON 4-) PCA / Feature Extraction

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets 
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu" #select the device
model_save_path = ""

# Prepare data
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



train_dataset = #prepare data to DataLoader and use data_transform 
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


val_dataset = #prepare data to DataLoader and use data_transform 
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)


model = models.resnet50(pretrained=True) # Load pre-trained ResNet-50 model


"""Freeze everything except the last 3 layers"""
for param in #resnet layers params except the last3: 

    param.requires_grad = False


""" You have to change resnet fc layer according to your class size"""
model.fc = 


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.to(device)

def val(model, val_loader):# Training loop
   
   model.valid()
   val_loss = 0.0

   with experiment.test():

      for inputs, labels in val_loader:

         with torch.no_grad():
             """
            Test model here 

            Save the best model on validation set
         """




def train(model, train_loader, epoch):# Training loop
   
   model.train()
   train_loss = 0.0
   #add necesseri variable
   for inputs, labels in train_loader:
      
      input = inputs.to(device)
      label = labels.to(device)

      """
         Train model here 
      """




for epoch in epochs:
   print(epoch)
   train(model,train_loader,epoch)
   val(model, val_loader, epoch)


