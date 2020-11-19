import pandas as pd
import os
import shutil
import random

def data_creator(covid_file_path, covid_image_path, target_dir, kaggle_file_path, target_normal_dir):
    '''
        this function is to create data from two different data sources
        
        inputs:
            covid_File_path : csv from which covid xray data to extract
            covid_image_path : img directory which contains covid xray image path
            target_dir : directory to save covid xray data from whole data
            kaggle_file_path : file path for normal xray data
            target_normal_dir : directory to save normal xray data from whole data

        output : 
            cretes final data directory
    '''
    df= pd.read_csv(covid_file_path)
    # df.head()

    cnt = 0

    for (i, row) in df.iterrows():
        if row["finding"]=="COVID-19" and row["view"]=="PA":
            filename =row["filename"]
            image_path = os.path.join(covid_image_path, filename)
            image_copy_path = os.path.join(target_dir, filename)
            shutil.copy2(image_path, image_copy_path)
            print("moving image ", cnt)
            cnt+=1


    kaggle_file_path = "chest-xray-pneumonia/chest_xray/train/NORMAL"
    target_normal_dir = "cleaned_dataset_for_covid_project/normal"

    image_name = os.listdir(kaggle_file_path)
    random.shuffle(image_name)

    for i in range(142):
    
        image_name =image_name[i] 
        image_normal_path = os.path.join(kaggle_file_path, image_name)
        target_normal_path = os.path.join(target_normal_dir, image_name)
        shutil.copy2(image_normal_path, target_normal_path)
        print(" copying image", i)

    
target_dir = "cleaned_dataset_for_covid_project/covid"
covid_file_path = "covid_chestxray_dataset_master/covid-chestxray-dataset-master/metadata.csv"
covid_image_path = "covid_chestxray_dataset_master/covid-chestxray-dataset-master/images"
kaggle_file_path = "chest-xray-pneumonia/chest_xray/train/NORMAL"
target_normal_dir = "cleaned_dataset_for_covid_project/normal"

data_creator(covid_file_path, covid_image_path, target_dir, kaggle_file_path, target_normal_dir)