import os
import io
import pandas as pd
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util
import argparse

def create_tf_example(group, path, labels):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for _, row in group.iterrows():  # Use iterrows instead of group.object.iterrows()
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(int(labels.index(row['class']) + 1))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main():
    parser = argparse.ArgumentParser(description='Create TFRecord files')
    parser.add_argument('--csv_input', required=True, help='Path to the CSV input')
    parser.add_argument('--labelmap', required=True, help='Path to the labelmap file')
    parser.add_argument('--image_dir', required=True, help='Path to the image directory')
    parser.add_argument('--output_path', required=True, help='Path to output TFRecord')
    args = parser.parse_args()

    writer = tf.io.TFRecordWriter(args.output_path)
    path = args.image_dir
    examples = pd.read_csv(args.csv_input)
    
    labels = []
    with open(args.labelmap, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Iterating over groups directly
    for _, group in examples.groupby('filename'):
        tf_example = create_tf_example(group, path, labels)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print('Successfully created the TFRecords: {}'.format(args.output_path))

    # Create labelmap.pbtxt file
    with open('labelmap.pbtxt', 'w') as f:
        for i, label in enumerate(labels):
            f.write('item {\n' +
                    '  id: %d\n' % (i + 1) +
                    '  name: \'%s\'\n' % label +
                    '}\n' +
                    '\n')

if __name__ == '__main__':
    main()
