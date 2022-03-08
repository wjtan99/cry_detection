#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time, os, sys, copy, argparse
import multiprocessing

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

import numpy as np


# In[2]:


import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchsummary import summary


# In[3]:


#Set the train and validation directory paths
#dataset_file = "dataset/dataset_8000_1.05_None_64_0_None_256_128.pk"
dataset_file = "dataset/trainset_8000_4.1_None_64_0_None_512_512.pk"
#dataset_file = "dataset/dataset_8000_5_None_64_0_None_512_256.pk"
#dataset_file = "dataset/dataset_8000_5_None_64_0_None_256_128.pk"
#dataset_file = "dataset/dataset_8000_5_None_64_50_None_256_128.pk"


# In[4]:


from model import BlazeNet 
from dataset import AudioDataset


# In[5]:


# Batch size
bs = 32 
# Number of epochs
num_epochs = 10
# Number of classes
num_classes = 2
# Number of workers
num_cpu = multiprocessing.cpu_count()

# Applying transforms to the data
image_transforms = { 
    'train': transforms.Compose([
        #transforms.Resize(size=128),
        #transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        #transforms.Resize(size=128),
        #transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        #transforms.Resize(size=128),
        #transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

}


# **train/** - folder containing the training files, with each top-level folder representing a subject  
# **train_labels.csv** - file containing the target MGMT_value for each subject in the training data (e.g. the presence of MGMT promoter methylation)   
# **test/** - the test files, which use the same structure as train/; your task is to predict the MGMT_value for each subject in the test data. NOTE: the total size of the rerun test set (Public and Private) is ~5x the size of the Public test set   
# **sample_submission.csv** - a sample submission file in the correct format

# In[8]:


# Load data from folders
dataset = {
    'train': AudioDataset(dataset_file, subset="train", transform=image_transforms['train']),
    'valid': AudioDataset(dataset_file, subset="valid", transform=image_transforms['valid']),
    'test': AudioDataset(dataset_file, subset="test", transform=image_transforms['test'])
}
 
# Size of train and validation data
dataset_sizes = {
    'train':len(dataset['train']),
    'valid':len(dataset['valid']),
    'test':len(dataset['test'])
}

# Create iterators for data loading
dataloaders = {
    'train':data.DataLoader(dataset['train'], batch_size=bs, shuffle=True,
                            num_workers=4, pin_memory=True, drop_last=True),
    'valid':data.DataLoader(dataset['valid'], batch_size=bs, shuffle=False,
                            num_workers=4, pin_memory=True, drop_last=True),
    'test':data.DataLoader(dataset['test'], batch_size=bs, shuffle=False,
                            num_workers=4, pin_memory=True, drop_last=True)

}

# Class names or target labels
class_names = dataset['train'].classes
class_to_idx = dataset['train'].class_to_idx
print("Classes:", class_names)
print("Class_to_idx:", class_to_idx)
 
# Print the train and validation data sizes
print("Training-set size:",dataset_sizes['train'],
      "\nValidation-set size:", dataset_sizes['valid'],
      "\nTest-set size:", dataset_sizes['test'],
     )


# In[9]:


# Set default device as gpu, if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[13]:


# Instantiate a neural network model 
model_ft = BlazeNet(back_model=2)

# Print model summary
print('Model Summary:-\n')
for num, (name, param) in enumerate(model_ft.named_parameters()):
    print(num, name, param.requires_grad )

model_ft = model_ft.to(device)
summary(model_ft, input_size=(3, 64, 64))
print(model_ft)


# In[14]:


# Loss function
weight = torch.tensor([1.0, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=weight)
#criterion = nn.BCELoss(weight=weight)


# In[15]:


# Model training routine 
print("\nTraining:-\n")
def train_model(model, criterion, optimizer, scheduler, num_epochs=30):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Tensorboard summary
    writer = SummaryWriter()
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            for inputs, labels, srcs, inds, vocals in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # Record training loss and accuracy for each phase
            if phase == 'train':
                writer.add_scalar('Train/Loss', epoch_loss, epoch)
                writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
                writer.flush()
            else:
                writer.add_scalar('Valid/Loss', epoch_loss, epoch)
                writer.add_scalar('Valid/Accuracy', epoch_acc, epoch)
                writer.flush()

            # deep copy the model
            if phase == 'valid' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# In[16]:


# Optimizer 
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)
# Train the model
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=60)
# Save the entire model
print("\nSaving the model...")
model_file = "checkpoints/blazenet_{}.pth".format(dataset_file.split('/')[-1]) 
torch.save(model_ft, model_file)


# In[16]:


# Class label names
class_names=['cry','nocry']

# Initialize the prediction and label lists
predlist=torch.zeros(0,dtype=torch.long, device='cpu')
lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

# Evaluate the model accuracy on the dataset
correct = 0
total = 0
with torch.no_grad():
    for images, labels, srcs, inds, vocals in dataloaders["test"]:
        images, labels = images.to(device), labels.to(device)
        outputs = model_ft(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        #print(labels,predicted)        
        correct += (predicted == labels).sum().item()

        predlist=torch.cat([predlist,predicted.view(-1).cpu()])
        lbllist=torch.cat([lbllist,labels.view(-1).cpu()])

# Overall accuracy
overall_accuracy=100 * correct / total
print('Accuracy of the network on the {:d} test images: {:.2f}%'.format(dataset_sizes['test'], 
    overall_accuracy))

# Confusion matrix
conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
print('Confusion Matrix')
print('-'*16)
print(conf_mat,'\n')

# Per-class accuracy
class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
print('Per class accuracy')
print('-'*18)
for label,accuracy in zip(dataset['test'].classes, class_accuracy):
     class_name=label
     print('Accuracy of class %8s : %0.2f %%'%(class_name, accuracy))


# In[17]:


testset_file = 'dataset/testset_8000_4.1_None_64_0_None_512_512.pk'
test_dataset2 = AudioDataset(
    testset_file,    
    subset="test",
    mode = "RGB",
    transform = image_transforms["test"]
)
test_loader2 = data.DataLoader(
    test_dataset2,
    batch_size=32,
    shuffle=False,
    num_workers=4,
)
for i, (img,label,src,ind,vocal) in enumerate(test_loader2):
    print(i)
    print(img.shape)
    print(label)
    print(src)
    print(ind)
    print(vocal)
    break  


# In[25]:


predictions = [] 

with torch.no_grad():
    for images, labels, srcs, inds, vocals in test_loader2:
        images, labels = images.to(device), labels.to(device)
        outputs = model_ft(images)
        
        _, predicted = torch.max(outputs.data, 1)
        
        probs = torch.softmax(outputs.data, 1)[:,0]
                
        probs = probs.cpu().detach().numpy()
        inds = inds.detach().numpy()
        vocals = vocals.detach().numpy()


        print(srcs)
        print(probs)
        print(inds)
        print(vocals)        
        #print(predicted)

        total += labels.size(0)        
        pred = predicted.tolist()       
        
        for k in range(len(srcs)):
            predictions.append((srcs[k],probs[k],inds[k],vocals[k])) # 1 is cry 


# In[27]:


predictions_by_audio = {} 

cry_thresh = 0.5 

audio_files = [x[0] for x in predictions]
audio_files = list(set(audio_files))

for au in audio_files:
    predictions_by_audio[au] = [] 
    
for p in predictions: 
    predictions_by_audio[p[0]].append((p[1],p[2],p[3],p[1]>cry_thresh))
    
for au in predictions_by_audio:
    print(au)
    print(predictions_by_audio[au])


# In[20]:


print(len(predictions_by_audio))


# In[ ]:




