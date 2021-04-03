import os
import numpy as np
import datetime
from PIL import Image
import cv2
Image.MAX_IMAGE_PIXELS = 1000000000000

slide_dims = np.load('slide_dims.npy', allow_pickle=True)

def _extractor(data_dir):

    probs_files = []
    cases_files = []
    for path, subdirs, files in os.walk(data_dir):
        for name in files:
            if name.split("_")[0] == 'prbs':
                probs_files.append(os.path.join(path, name))
            elif name.split("_")[0] == 'imgs':
                cases_files.append(os.path.join(path, name))

    cases = np.empty((0,2), 'int')
    for file in cases_files:
        imgs = np.load(file)
        #print(imgs.shape)
        cases=np.append(cases, imgs, axis=0)
    for i in range(len(cases)):
        cases[i][0]=cases[i][0].replace("\\","/")

    patches = []
    for i in range(len(cases)):
        patches.append(cases[i][0].split('/')[-2] + '/' + cases[i][0].split('/')[-1])

    slides = np.unique([f.split('_')[0] + '_' + f.split('_')[1] for f in patches])
    
    """probs = np.empty((0,3), 'float64')
    for file in probs_files:
        prbs = np.load(file)
        #print(prbs.shape)
        probs=np.append(probs, prbs, axis=0)"""
    
    probs = np.empty((0,3), 'int')
    if len(probs_files)>1:
        for s in slides:
            count=0
            for file in probs_files:
                if s.split('/')[1] in file:
                    #print(file)
                    prbs = np.load(file)
                    if count==0:
                        prbs_sum = prbs
                    else:
                        prbs_sum += prbs
                    count+=1
            prbs_sum /= count
            probs=np.append(probs, prbs_sum, axis=0)
    elif len(probs_files)==1:
        probs = np.load(probs_files[0])
    else:
        print('probs_files has length zero')

    return (slides, patches, probs, cases)

def _map(patch_size, data_dir):
    
    slides, patches, probs, cases = _extractor(data_dir)

    for s in slides:
        for slide_dim in slide_dims:
            if s.split('/')[1] in slide_dim[0]:
                prob_map=np.zeros((slide_dim[1][0][1]//patch_size,slide_dim[1][0][0]//patch_size), 'uint8')
        for p in range(len(patches)):
            if s in patches[p]:
                x0=int(patches[p].split('.')[0].split('_')[-3])//patch_size
                y0=int(patches[p].split('.')[0].split('_')[-1])//patch_size
                prob_map[y0,x0] = int( 255.9*probs[p,int(cases[p][1])] )
        im = Image.fromarray(prob_map)
        image_name = os.path.basename(os.path.dirname(data_dir))+ '_' + s.replace('/','_')+'_map.png'
        im.save( data_dir + image_name )

        
def _map_color(patch_size, data_dir):
    
    slides, patches, probs, cases = _extractor(data_dir)

    for s in slides:
        for slide_dim in slide_dims:
            if s.split('/')[1] in slide_dim[0]:
                prob_map=np.zeros((slide_dim[1][0][1]//patch_size,slide_dim[1][0][0]//patch_size, 3), 'uint8')
        for p in range(len(patches)):
            if s in patches[p]:
                x0=int(patches[p].split('.')[0].split('_')[-3])//patch_size
                y0=int(patches[p].split('.')[0].split('_')[-1])//patch_size
                
                prob_map[y0,x0, 0] = int( 255.9*probs[p,2] )
                prob_map[y0,x0, 1] = int( 255.9*probs[p,0] )
                prob_map[y0,x0, 2] = int( 255.9*probs[p,1] )
        im = Image.fromarray(prob_map)            
        image_name = os.path.basename(os.path.dirname(data_dir))+ '_' + s.replace('/','_')+'_map_color.png'
        im.save( data_dir + image_name )

        
def _map_overlay(patch_size, data_dir):
    
    slides, patches, probs, cases = _extractor(data_dir)

    downsamples = './downsamples/'
    level_1_files = []
    for path, subdirs, files in os.walk(downsamples):
        for name in files:
            if 'level1.tif' in name:
                level_1_files.append(os.path.join(path, name))
            
    for s in slides:
        
        print(datetime.datetime.now())
        print('---------------%s----------------' %(s))
        for slide_dim in slide_dims:
            if s.split('/')[1] in slide_dim[0]:
                prob_map=np.zeros((slide_dim[1][1][1],slide_dim[1][1][0], 3), 'uint8')
        for p in range(len(patches)):
            if s in patches[p]:
                step = patch_size//4
                x0=int(patches[p].split('.')[0].split('_')[-3])//4
                y0=int(patches[p].split('.')[0].split('_')[-1])//4
                prob_map[y0:y0+step,x0:x0+step, 0].fill(int(255.9*probs[p, 2]))  # Red = late
                prob_map[y0:y0+step,x0:x0+step, 1].fill(int(255.9*probs[p, 0]))  # Green = pre
                prob_map[y0:y0+step,x0:x0+step, 2].fill(int(255.9*probs[p, 1]))  # Blue = mid
        #im = Image.fromarray(prob_map)            
        #map_resized = im.resize((dims[0][0]//4, dims[0][1]//4))
        cv2_map = cv2.cvtColor(prob_map, cv2.COLOR_RGB2BGR)
        for level_1 in level_1_files:
            if s.split('/')[1] in level_1:
                level_1_wsi = Image.open(level_1)
                cv2_wsi = cv2.cvtColor(np.array(level_1_wsi), cv2.COLOR_RGB2BGR)
        #print('---------------adding----------------')
        
        added_image = cv2.addWeighted(cv2_wsi, 0.7, cv2_map, 0.3, 0)
        #print('---------------saving----------------')
        print(datetime.datetime.now())
        image_name = os.path.basename(os.path.dirname(data_dir))+ '_' + s.replace('/','_')+'_overlayed_lv1.png'
        cv2.imwrite( data_dir + image_name, added_image)
        
def _table_count(data_dir):
    
    slides, patches, probs, cases = _extractor(data_dir)
    filename = os.path.basename(os.path.dirname(data_dir)) + '_' + 'table_count.txt'
    with open( data_dir + filename, 'w') as text_file:
        text_file.write( '% 23s % 11s % 11s \n' % ('0_pre', '1_mid', '2_lte') )
        for s in slides:
            c1, c2, c3 = 0, 0, 0
            for p in range(len(patches)):
                if s in patches[p]:
                    c = np.argmax(probs[p])
                    if c == 0:
                        c1 += 1
                    elif c == 1:
                        c2 += 1
                    else:
                        c3 += 1
            cf= 100/(c1 + c2 + c3)
            text_file.write( '% 11s % 11f % 11f % 11f \n' % (s, c1*cf, c2*cf, c3*cf))
    
    
def _table_prob(data_dir):
    
    slides, patches, probs, cases = _extractor(data_dir)
    filename = os.path.basename(os.path.dirname(data_dir)) + '_' + 'table_prob.txt'
    with open( data_dir + filename, 'w') as text_file:
        text_file.write( '% 23s % 11s % 11s \n' % ('0_pre', '1_mid', '2_lte') )
        for s in slides:
            c1, c2, c3 = 0, 0, 0
            for p in range(len(patches)):
                if s in patches[p]:
                    c1 += probs[p,0]
                    c2 += probs[p,1]
                    c3 += probs[p,2]
            cf= 100/(c1 + c2 + c3)
            text_file.write( '% 11s % 11f % 11f % 11f \n' % (s, c1*cf, c2*cf, c3*cf))
