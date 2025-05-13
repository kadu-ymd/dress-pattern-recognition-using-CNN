# Importing the Keras libraries and packages
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

import shutil

def details_vcount(list: list):
    print("*" * 40)

    for file in list:
        print(f"Lendo arquivo {file}:\n")

        df = pd.read_csv(f"{file}")
        print(df.Details.value_counts())
        print("*" * 40)

def main():
    file_list = "2_details_categories.csv, 3_details_categories.csv, 6_details_categories.csv".split(", ")

    df_list = pd.read_csv(file_list[0]).file_name.tolist()
    default_df_list = pd.read_csv("manipulated_data/initial_filtered_clothes.csv").file_name.tolist()

    test_clothes_list = []

    for img in os.listdir('images'):
        if img not in df_list and img in default_df_list:
            test_clothes_list.append(img)

    test_dataset = test_clothes_list[:len(df_list)//2]

    folder_name = "dataset_test"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

        print(f"-> Folder {folder_name} created successfully")
    else:
        print(f"-> Folder {folder_name} already exists")

    for img in test_dataset:
        src_path = os.path.join("images", img)
        dest_path = os.path.join(folder_name, img)

        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
        else:
            print(f'File not found: {img}')

    

    return 0

if __name__ == "__main__":
    main()