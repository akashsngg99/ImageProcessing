
import os
import tensorflow as tf

datasets = 'datasets/actions/'
labels = ['boxing','handclapping','handwaving','jogging','running','walking']
file_label_dict = {}

listing = []


def gen_file_record():
    for index,label in enumerate(labels):
        for file_name in os.listdir(datasets + label):
            file_label_dict[file_name] = index
            print(file_name,file_label_dict[file_name])






if __name__ == "__main__":
    gen_file_record()