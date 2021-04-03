# generation of training (random or topk) and testing patch ids

import os
import numpy as np
import datetime
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000000
from torchvision import datasets
import random

# given numpy files for list of validated patches (cases.npy) 
# and their probs (probs.npy) we find top k patch in gound truth map

def top_k(patch_size, k, cases, probs, patches):
   
    full_dataset = datasets.ImageFolder(patches)
    full_cases = np.array(full_dataset.samples)
    
    for i in range(len(full_cases)):
        full_cases[i][0]=full_cases[i][0].replace("\\","/")
    
    slide_dims = np.load('slide_dims.npy', allow_pickle=True)
    cases = np.load(cases)
    probs = np.load(probs)
    
    for i in range(len(cases)):
        cases[i][0]=cases[i][0].replace("\\","/")
    
    patches = []
    for i in range(len(cases)):
        patches.append(cases[i][0].split('/')[-2] + '/' + cases[i][0].split('/')[-1])
        
    slides = np.unique([f.split('_')[0] + '_' + f.split('_')[1] for f in patches])

    print(slides)
    
    top_train_ids = []
    top_val_ids = []

    for s in slides:
        for slide_dim in slide_dims:
            if s.split('/')[1] in slide_dim[0]:
                prob_map=np.zeros((slide_dim[1][0][1]//patch_size,slide_dim[1][0][0]//patch_size))
        #print(prob_map.shape)
        for p in range(len(patches)):
            if s in patches[p]:
                x0=int(patches[p].split('.')[0].split('_')[-3])//patch_size
                y0=int(patches[p].split('.')[0].split('_')[-1])//patch_size
                
                prob_map[y0,x0] = probs[p, int( cases[p][1] ) ]

        disc_mask=(prob_map>=np.sort(prob_map, axis=None)[-k])*1
        nodes=np.transpose(np.array(disc_mask.nonzero()))
        
        top_slide_ids = []
        for node in nodes:
            y0, x0 = node[0]*patch_size , node[1]*patch_size
            coords = 'x0_'+str(x0)+'_y0_'+str(y0)
            #print(coords)
            for q in range(len(full_cases)):
                if (s in full_cases[q][0]) and ( coords in full_cases[q][0]):
                    top_slide_ids.append(q)
        
        n_val =  int( 0.1 * k )
        slide_val_sample = random.sample( top_slide_ids, n_val)
        slide_train_sample = list ( set(top_slide_ids) - set(slide_val_sample) )
        
        top_val_ids.extend(slide_val_sample)
        top_train_ids.extend(slide_train_sample)
        
    return (top_train_ids, top_val_ids)


def test(test_ids, patches):
   
    full_dataset = datasets.ImageFolder(patches)
    classes=full_dataset.classes
    full_cases = np.array(full_dataset.samples)
    
    for i in range(len(full_cases)):
        full_cases[i][0]=full_cases[i][0].replace("\\","/")
    
    patches = []
    for i in range(len(full_cases)):
        patches.append(full_cases[i][0].split('/')[-2] + '/' + full_cases[i][0].split('/')[-1])
        
    slides = np.unique([f.split('_')[0] + '_' + f.split('_')[1] for f in patches])

    test_slides = slides[test_ids]

    print(test_slides)

    test_list = []

    for i, p in enumerate(patches):
        if any(t in p for t in test_slides):
            test_list.append(i)
                
    return (test_list)

def random_k(samples_size, train_ids, patches):
   
    full_dataset = datasets.ImageFolder(patches)
    classes=full_dataset.classes
    full_cases = np.array(full_dataset.samples)
    
    for i in range(len(full_cases)):
        full_cases[i][0]=full_cases[i][0].replace("\\","/")
    
    patches = []
    for i in range(len(full_cases)):
        patches.append(full_cases[i][0].split('/')[-2] + '/' + full_cases[i][0].split('/')[-1])
        
    slides = np.unique([f.split('_')[0] + '_' + f.split('_')[1] for f in patches])

    train_slides = slides[train_ids]

    print(train_slides)

    train_list = []
    val_list = []


    for c in classes:
        ids=[]
        for i, p in enumerate(patches):
            if c in p and any(t in p for t in train_slides):
                ids.append(i)
        
        class_samples = random.sample(ids, samples_size)
        n_val =  int( 0.1 * samples_size )
        val_list.extend(class_samples[0:n_val])
        train_list.extend(class_samples[n_val:])

    return (train_list, val_list)

