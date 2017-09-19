import glob
from psd_tools import PSDImage
from matplotlib import pyplot as plt
import numpy as np
import time
import cv2
import xml.etree.cElementTree as ET
import tensorflow as tf
from object_detection.utils import dataset_util
import random
import numpy

flags = tf.app.flags
flags.DEFINE_string('output_path', './dl_data/outfile.records', 'Path to output TFRecord')
FLAGS = flags.FLAGS

writer = tf.python_io.TFRecordWriter('./dl_data/train_dataset.records')

class ImageCoder(object):
        """Helper class that provides TensorFlow image coding utilities."""

        def __init__(self):
                # Create a single Session to run all image coding calls.
                self._sess = tf.Session()

                # Initializes function that converts PNG to JPEG data.
                self._png_data = tf.placeholder(dtype=tf.string)
                image = tf.image.decode_png(self._png_data, channels=3)
                self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

                # Initializes function that decodes RGB JPEG data.
                self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
                self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

        def png_to_jpeg(self, image_data):
                return self._sess.run(self._png_to_jpeg, feed_dict={self._png_data: image_data})

        def decode_jpeg(self, image_data):
                image = self._sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
                assert len(image.shape) == 3
                assert image.shape[2] == 3
                return image

def create_tf_example(example):
        # TODO(user): Populate the following variables from your example.
        height = example["height"] # Image height
        width = example["width"] # Image width
        filename = example["filename"] # Filename of the image. Empty if image is not from file
        encoded_image_data = example["encoded_image_data"] # Encoded image bytes
        image_format = example["image_format"] # b'jpeg' or b'png'
        
        xmins = [example["xmin"] / width] # List of normalized left x coordinates in bounding box (1 per box)
        xmaxs = [example["xmax"] / width] # List of normalized right x coordinates in bounding box
        # (1 per box)
        ymins = [example["ymin"] / height] # List of normalized top y coordinates in bounding box (1 per box)
        ymaxs = [example["ymax"] / height] # List of normalized bottom y coordinates in bounding box
        # (1 per box)
        classes_text = [example["classtext"]] # List of string class name of bounding box (1 per box)
        classes = [example["classnum"]] # List of integer class id of bounding box (1 per box)
        
        tf_example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': dataset_util.int64_feature(height),
                'image/width': dataset_util.int64_feature(width),
                'image/filename': dataset_util.bytes_feature(filename),
                'image/source_id': dataset_util.bytes_feature(filename),
                'image/encoded': dataset_util.bytes_feature(encoded_image_data),
                'image/format': dataset_util.bytes_feature(image_format),
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        return tf_example

def is_png(filename):
        """Determine if a file contains a PNG format image.
        Args:
        filename: string, path of the image file.
        Returns:
        boolean indicating if the image is a PNG.
        """
        return '.png' in filename

def process_image(filename, coder):
        """Process a single image file.
        Args:
        filename: string, path to an image file e.g., '/path/to/example.JPG'.
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
        Returns:
        image_buffer: string, JPEG encoding of RGB image.
        height: integer, image height in pixels.
        width: integer, image width in pixels.
        """
        # Read the image file.
        with tf.gfile.FastGFile(filename, 'rb') as f:
                image_data = f.read()

        # # Convert any PNG to JPEG's for consistency.
        # if is_png(filename):
        #         print('Converting PNG to JPEG for %s' % filename)
        #         image_data = coder.png_to_jpeg(image_data)

        # Decode the RGB JPEG.
        image = coder.decode_jpeg(image_data)
        
        # Check that image converted to RGB
        assert len(image.shape) == 3
        height = image.shape[0]
        width = image.shape[1]
        assert image.shape[2] == 3

        return image_data, height, width

psdfiles = glob.glob('../*.psd')

featuresArray = np.array([])
labellingArray = np.array([])

index = 0

train_num = 62
var_num = len(psdfiles)
randomList = random.sample(range(len(psdfiles)), train_num)

for i in randomList:
    psdfile = psdfiles[i]
    print psdfile
    psd = PSDImage.load(psdfile)
    label = psd.layers[0]
    base = psd.layers[1]
    labelImage = label.as_PIL()
    baseImage = base.as_PIL()
    
    labelArray = np.asarray(labelImage, dtype=np.uint8)/255
    baseArray = np.asarray(baseImage, dtype=np.uint8)
    
    filename = psdfile.split('/')[1].split('.')[0]
    
    labelSlice = labelArray[:,:,3]
    imageWidth = labelSlice.shape[0]
    imageHeight = labelSlice.shape[1]
        
    # plt.ion()
    # plt.figure('test')
    # plt.imshow(labelSlice)

    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(labelSlice)
    
    # if (len(stats) - 1) == 0:
    #         break
    flatlabels = labels.ravel()
        
    annotation_file = '/home/jinzeyu/Downloads/demo2.xml'
        
    tree = ET.ElementTree(file=annotation_file)
    root = tree.getroot()

    tree = ET.parse(annotation_file)
    tree.find('folder').text = 'touthdisea'
    tree.find('path').text = psdfile.split('/')[1]
        
    tree.find('filename').text = filename + '.png'
    size = tree.find('size')
    size.find('width').text = str(imageHeight)
    size.find('height').text = str(imageWidth)
    
    index = 0
    
    for componentIndex in range(len(stats) - 1):
            
        index = index + 1
        # if index > 1:
        #     continue
        boolCurrentPart = (flatlabels == (componentIndex + 1))
        allIndices = np.arange(imageWidth * imageHeight)
        indicesCurrentPart = allIndices[boolCurrentPart]
        xIndices = indicesCurrentPart % imageHeight
        yIndices = indicesCurrentPart / imageHeight
        
        xMin, xMax, yMin, yMax = min(xIndices), max(xIndices), min(yIndices), max(yIndices)

        # from IPython import embed; embed()

        example = {}
        
        example["xmin"] = xMin
        example["xmax"] = xMax
        example["ymin"] = yMin
        example["ymax"] = yMax
        
        example["height"] = imageWidth
        example["width"] = imageHeight
        example["filename"] = filename + '.png' # Filename of the image. Empty if image is not from file
        # encoded_image_data =  # Encoded image bytes
        example["image_format"] = b'jpg' # b'jpeg' or b'png'

        imageFileName = '/home/jinzeyu/Downloads/touthdiseaImage/' + filename + '.jpg'
                
        cv2.imwrite(imageFileName, baseArray[:,:,::-1])

        coder = ImageCoder()
        image_buffer, height, width = process_image(imageFileName, coder)
        example["encoded_image_data"] = image_buffer
        
        example["classtext"] = "DecayedTooth" # List of string class name of bounding box (1 per box)
        example["classnum"] = 21 # List of integer class id of bounding box (1 per box)
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())


    # tree.write('/home/jinzeyu/Downloads/touthdisea/' + filename + '.xml')
writer.close()

print "build val dataset......"

writer = tf.python_io.TFRecordWriter('./dl_data/val_dataset.records')

for i in numpy.delete(range(len(psdfiles)), randomList):
    psdfile = psdfiles[i]
    print psdfile
    psd = PSDImage.load(psdfile)
    label = psd.layers[0]
    base = psd.layers[1]
    labelImage = label.as_PIL()
    baseImage = base.as_PIL()
    
    labelArray = np.asarray(labelImage, dtype=np.uint8)/255
    baseArray = np.asarray(baseImage, dtype=np.uint8)
    
    filename = psdfile.split('/')[1].split('.')[0]
    
    labelSlice = labelArray[:,:,3]
    imageWidth = labelSlice.shape[0]
    imageHeight = labelSlice.shape[1]
        
    # plt.ion()
    # plt.figure('test')
    # plt.imshow(labelSlice)

    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(labelSlice)
    
    # if (len(stats) - 1) == 0:
    #         break
    flatlabels = labels.ravel()
        
    annotation_file = '/home/jinzeyu/Downloads/demo2.xml'
        
    tree = ET.ElementTree(file=annotation_file)
    root = tree.getroot()

    tree = ET.parse(annotation_file)
    tree.find('folder').text = 'touthdisea'
    tree.find('path').text = psdfile.split('/')[1]
        
    tree.find('filename').text = filename + '.png'
    size = tree.find('size')
    size.find('width').text = str(imageHeight)
    size.find('height').text = str(imageWidth)
    
    index = 0
    
    for componentIndex in range(len(stats) - 1):
            
        index = index + 1
        # if index > 1:
        #     continue
        boolCurrentPart = (flatlabels == (componentIndex + 1))
        allIndices = np.arange(imageWidth * imageHeight)
        indicesCurrentPart = allIndices[boolCurrentPart]
        xIndices = indicesCurrentPart % imageHeight
        yIndices = indicesCurrentPart / imageHeight
        
        xMin, xMax, yMin, yMax = min(xIndices), max(xIndices), min(yIndices), max(yIndices)

        # from IPython import embed; embed()

        example = {}
        
        example["xmin"] = xMin
        example["xmax"] = xMax
        example["ymin"] = yMin
        example["ymax"] = yMax
        
        example["height"] = imageWidth
        example["width"] = imageHeight
        example["filename"] = filename + '.png' # Filename of the image. Empty if image is not from file
        # encoded_image_data =  # Encoded image bytes
        example["image_format"] = b'jpg' # b'jpeg' or b'png'

        imageFileName = '/home/jinzeyu/Downloads/touthdiseaImage/' + filename + '.jpg'
                
        cv2.imwrite(imageFileName, baseArray[:,:,::-1])

        coder = ImageCoder()
        image_buffer, height, width = process_image(imageFileName, coder)
        example["encoded_image_data"] = image_buffer
        
        example["classtext"] = "DecayedTooth" # List of string class name of bounding box (1 per box)
        example["classnum"] = 21 # List of integer class id of bounding box (1 per box)
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())


    # tree.write('/home/jinzeyu/Downloads/touthdisea/' + filename + '.xml')
writer.close()
