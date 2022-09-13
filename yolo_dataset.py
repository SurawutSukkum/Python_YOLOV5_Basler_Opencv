import os
from glob import glob 
import pandas as pd 
from xml.etree import ElementTree as et 
from functools import reduce
from shutil import move,copy

import warnings
warnings.filterwarnings('ignore')

def xml_to_namelist(path):
    
    xmlfiles = glob(path)
    
    replace_text = lambda x: x.replace('\\','/')
    xmlfiles = list(map(replace_text,xmlfiles))
    return xmlfiles

def readxml_to_df(xmlfiles):
    def extract_text(filename):
        tree = et.parse(filename)
        root = tree.getroot()

        
        image_name = root.find('filename').text
        
        width = root.find('size').find('width').text
        height = root.find('size').find('height').text
        objs = root.findall('object')
        parser = []
        for obj in objs:
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = bndbox.find('xmin').text
            xmax = bndbox.find('xmax').text
            ymin = bndbox.find('ymin').text
            ymax = bndbox.find('ymax').text
            parser.append([image_name, width, height, name,xmin,xmax,ymin,ymax])
            
        return parser

    parser_all = list(map(extract_text,xmlfiles))
    data = reduce(lambda x, y : x+y,parser_all)
    df = pd.DataFrame(data,columns = ['filename','width','height','name','xmin','xmax','ymin','ymax'])
    
    cols = ['width','height','xmin','xmax','ymin','ymax']
    df[cols] = df[cols].astype(int)
    
    df['center_x'] = ((df['xmax']+df['xmin'])/2)/df['width']
    df['center_y'] = ((df['ymax']+df['ymin'])/2)/df['height']
    
    df['w'] = (df['xmax']-df['xmin'])/df['width']
     
    df['h'] = (df['ymax']-df['ymin'])/df['height']
    
    return df

def make_fd_train_test(df,ratio_train,obj_name,main_folder,for_train_folder):
    images = df['filename'].unique()

    img_df = pd.DataFrame(images,columns=['filename'])
    img_train = tuple(img_df.sample(frac=ratio_train)['filename']) # shuffle and pick ratio_train% of images
    img_test = tuple(img_df.query(f'filename not in {img_train}')['filename']) # take rest 1-ratio_train% images
    
    train_df = df.query(f'filename in {img_train}')
    test_df = df.query(f'filename in {img_test}')

    def label_encoding(x):
        labels = obj_name
        return labels[x]

    train_df['id'] = train_df['name'].apply(label_encoding)
    test_df['id'] = test_df['name'].apply(label_encoding)
   
    train_folder = for_train_folder + '/train'
    test_folder = for_train_folder + '/valid'
    os.mkdir(train_folder)
    os.mkdir(test_folder)

    cols = ['filename','id','center_x','center_y', 'w', 'h']
    groupby_obj_train = train_df[cols].groupby('filename')
    groupby_obj_test = test_df[cols].groupby('filename')

    def save_data(filename, folder_path, group_obj,main_folder):
        src = os.path.join(main_folder,filename)
        dst = os.path.join(folder_path,filename)
        copy(src,dst) 
        
        text_filename = os.path.join(folder_path,
                                     os.path.splitext(filename)[0]+'.txt')
        group_obj.get_group(filename).set_index('filename').to_csv(text_filename,sep=' ',index=False,header=False)

    
    filename_series = pd.Series(list(groupby_obj_train.groups.keys()))
    filename_series.apply(save_data,args=(train_folder,groupby_obj_train,main_folder))

    filename_series_test = pd.Series(list(groupby_obj_test.groups.keys()))
    filename_series_test.apply(save_data,args=(test_folder,groupby_obj_test,main_folder))
