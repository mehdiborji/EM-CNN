# M step taking list of indices of patches for training

# round_0 takes the pre_trained weights
# round_n takes the weights from last iteration
# external testing takes the slides not used during trainig

import numpy as np
import os
import torch
from torchvision import datasets, models, transforms as T
import trainer
import datetime
import argparse
import ast
import pathlib


def CNN_train_round_0(output_dir, it_n, data_dir, train_list, val_list, test_list, num_classes, num_epochs):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(device)
    #print(torch.cuda.device_count())
    #print(torch.cuda.get_device_name(0))
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True) 

    full_dataset = datasets.ImageFolder(data_dir)
    classes=full_dataset.classes
    samples=full_dataset.samples


    T_Normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    data_transforms = {'train': T.Compose([ T.RandomResizedCrop(224), T.RandomVerticalFlip(), T.RandomHorizontalFlip(), T.ToTensor(),  T_Normalize ]),
                        'val': T.Compose([ T.Resize(224), T.ToTensor(), T_Normalize ]),
                        'test': T.Compose([ T.Resize(224), T.ToTensor(), T_Normalize ])}

    data_subsets = {'train': train_list, 'val': val_list, 'test': test_list}
    image_datasets = {x: torch.utils.data.Subset(datasets.ImageFolder(data_dir, data_transforms[x]), data_subsets[x]) for x in ['train', 'val', 'test']}

    training_dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
    testing_dataloader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=128, shuffle=False, num_workers=4)

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    print(dataset_sizes)

  
    model_ft = models.resnet18(pretrained=True)
    
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)
    model_ft = model_ft.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)    
    model_ft, loss, accuracy = trainer.train_model(training_dataloaders, model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs, device, dataset_sizes)
    torch.save(model_ft, output_dir + 'model_round_' + str(it_n) + '.pth' )
    
    model_ft.eval()
    probabilities = []
    with torch.no_grad():
        for data, target in testing_dataloader:
            
            data = data.to(device)
            output = torch.nn.functional.softmax(model_ft(data), dim=1)
            probabilities.extend(output.cpu().numpy())
     
    
    np.save(output_dir + 'prbs_round_' + str(it_n) + '.npy',np.array( probabilities ) )
    np.save(output_dir + 'imgs_round_' + str(it_n) + '.npy',np.array( [full_dataset.samples[i] for i in test_list] ) )

    
    
def CNN_train_round_n(output_dir, it_n, data_dir, train_list, val_list, test_list, model, num_epochs):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(device)
    #print(torch.cuda.device_count())
    #print(torch.cuda.get_device_name(0))
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    full_dataset = datasets.ImageFolder(data_dir)
    classes=full_dataset.classes
    samples=full_dataset.samples

    T_Normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    data_transforms = {'train': T.Compose([ T.RandomResizedCrop(224), T.RandomVerticalFlip(), T.RandomHorizontalFlip(), T.ToTensor(),  T_Normalize ]),
                        'val': T.Compose([ T.Resize(224), T.ToTensor(), T_Normalize ]),
                        'test': T.Compose([ T.Resize(224), T.ToTensor(), T_Normalize ])}

    data_subsets = {'train': train_list, 'val': val_list, 'test': test_list}
    image_datasets = {x: torch.utils.data.Subset(datasets.ImageFolder(data_dir, data_transforms[x]), data_subsets[x]) for x in ['train', 'val', 'test']}

    training_dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
    testing_dataloader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=128, shuffle=False, num_workers=4)

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    print(dataset_sizes)

    model_ft = torch.load(model)
    
    model_ft = model_ft.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)    
    model_ft, loss, accuracy = trainer.train_model(training_dataloaders, model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs, device, dataset_sizes)
    torch.save(model_ft, output_dir + 'model_round_' + str(it_n) + '.pth' )
    
    model_ft.eval()
    probabilities = []
    with torch.no_grad():
        for data, target in testing_dataloader:
            
            data = data.to(device)
            output = torch.nn.functional.softmax(model_ft(data), dim=1)
            probabilities.extend(output.cpu().numpy())
            
    np.save(output_dir + 'prbs_round_' + str(it_n) + '.npy',np.array( probabilities ) )
    np.save(output_dir + 'imgs_round_' + str(it_n) + '.npy',np.array( [full_dataset.samples[i] for i in test_list] ) )

def CNN_external_test(output_dir, data_dir, test_list, model):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    full_dataset = datasets.ImageFolder(data_dir)
    classes=full_dataset.classes
    samples=full_dataset.samples

    T_Normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    data_transform = T.Compose([ T.Resize(224), T.ToTensor(), T_Normalize ])
    image_dataset = torch.utils.data.Subset(datasets.ImageFolder(data_dir, data_transform), test_list)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=128, shuffle=False, num_workers=4)
    print( len(image_dataset) )
    model_ft = torch.load(model)
    model_ft = model_ft.to(device)
    model_ft.eval()
    probabilities = []
    with torch.no_grad():
        for data, target in dataloader:
            
            data = data.to(device)
            output = torch.nn.functional.softmax(model_ft(data), dim=1)
            probabilities.extend(output.cpu().numpy())
            
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)         
    np.save(output_dir + 'prbs' + '_' + 'external' + '.npy',np.array( probabilities ) )
    np.save(output_dir + 'imgs' + '_' + 'external' + '.npy',np.array( [full_dataset.samples[i] for i in test_list] ) )

if __name__ == '__main__':
    main()