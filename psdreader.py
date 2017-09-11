import glob
from psd_tools import PSDImage
from matplotlib import pyplot as plt
import numpy as np
import time
import cv2
import xml.etree.cElementTree as ET

psdfiles = glob.glob('../*.psd')

featuresArray = np.array([])
labellingArray = np.array([])

print "len is joipfe"

index = 0

for psdfile in psdfiles:
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
        
        for componentIndex in range(len(stats) - 1):
                boolCurrentPart = (flatlabels == (componentIndex + 1))
                allIndices = np.arange(imageWidth * imageHeight)
                indicesCurrentPart = allIndices[boolCurrentPart]
                xIndices = indicesCurrentPart % imageHeight
                yIndices = indicesCurrentPart / imageHeight

                xMin, xMax, yMin, yMax = min(xIndices), max(xIndices), min(yIndices), max(yIndices)

                if componentIndex == 1:
                        object = tree.find('object')
                        object.find('name').text = 'DecayedTooth'
                        bndbox = object.find('bndbox')
                        bndbox.find('xmin').text = str(xMin)
                        bndbox.find('ymin').text = str(yMin)
                        bndbox.find('xmax').text = str(xMax)
                        bndbox.find('ymax').text = str(yMax)
                elif componentIndex > 1:
                        objectElement = ET.Element('object', {})
                        subElement = ET.Element('name', {})
                        subElement.text = "DecayedTooth"
                        subElement.tail = '\n\t'
                        objectElement.append(subElement)
                        subElement = ET.Element('pose', {})
                        subElement.text = "Unspecified"
                        subElement.tail = '\n\t'
                        objectElement.append(subElement)
                        subElement = ET.Element('truncated', {})
                        subElement.text = '0'
                        subElement.tail = '\n\t'
                        objectElement.append(subElement)
                        subElement = ET.Element('difficult', {})
                        subElement.text = '0'
                        subElement.tail = '\n\t'
                        objectElement.append(subElement)
                        subElement = ET.Element('bndbox', {})
                        subsubElement = ET.Element('xmin', {})
                        subsubElement.text = str(xMin)
                        subsubElement.tail = '\n\t'
                        subElement.append(subsubElement)
                        subsubElement = ET.Element('xmax', {})
                        subsubElement.text = str(xMax)
                        subsubElement.tail = '\n\t'
                        subElement.append(subsubElement)
                        subsubElement = ET.Element('ymin', {})
                        subsubElement.text = str(yMin)
                        subsubElement.tail = '\n\t'
                        subElement.append(subsubElement)
                        subsubElement = ET.Element('ymax', {})
                        subsubElement.text = str(yMax)
                        subsubElement.tail = '\n\t'
                        subElement.append(subsubElement)
                        objectElement.append(subElement)
                        objectElement.tail = '\n\t'
                        root.append(objectElement)

        imageFileName = '/home/jinzeyu/Downloads/touthdiseaImage/' + filename + '.png'
                
        cv2.imwrite(imageFileName, baseArray[:,:,::-1])


        tree.write('/home/jinzeyu/Downloads/touthdisea/' + filename + '.xml')

