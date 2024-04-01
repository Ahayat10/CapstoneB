import os
import glob
import pandas as pd

IMAGE_WIDTH = 480
IMAGE_HEIGHT = 852

def txt_to_csv(path):
    txt_list = []
    for txt_file in glob.glob(path + '/*.txt'):
        with open(txt_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip().split()
                # Assuming the format of each line in the .txt file is: class x_center y_center width height
                class_label = line[0]
                x_center = float(line[1])
                y_center = float(line[2])
                width = float(line[3])
                height = float(line[4])
                # Convert YOLO format to Pascal VOC format (xmin, ymin, xmax, ymax)
                xmin = int((x_center - width / 2) * IMAGE_WIDTH)
                ymin = int((y_center - height / 2) * IMAGE_HEIGHT)
                xmax = int((x_center + width / 2) * IMAGE_WIDTH)
                ymax = int((y_center + height / 2) * IMAGE_HEIGHT)
                value = (os.path.basename(txt_file.replace('.txt', '.jpg')), IMAGE_WIDTH, IMAGE_HEIGHT, class_label, xmin, ymin, xmax, ymax)
                txt_list.append(value)
                
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    txt_df = pd.DataFrame(txt_list, columns=column_name)
    return txt_df

def main():
    # Update the paths for train, test, and valid folders
    for folder in ['train', 'test', 'valid']:
        image_path = os.path.join('/content/drive/MyDrive/CapstoneB/yolov8', folder)
        # Assuming all images have the same width and height
        txt_df = txt_to_csv(image_path)
        txt_df.to_csv(os.path.join(image_path, folder + '_labels.csv'), index=None)
        print(f'Successfully converted txt to csv for {folder} set.')

main()
