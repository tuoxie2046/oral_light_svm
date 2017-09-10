import glob
from psd_tools import PSDImage
from matplotlib import pyplot as plt
import numpy as np
import time

psdfiles = glob.glob('*.psd')

featuresArray = np.array([])
labellingArray = np.array([])

index = 0

for psdfile in psdfiles:

    index = index + 1
    if index > 20:
        break
    psd = PSDImage.load(psdfile)
    label = psd.layers[0]
    base = psd.layers[1]
    labelImage = label.as_PIL()
    baseImage = base.as_PIL()

    labelArray = np.asarray(labelImage, dtype=np.uint8)/255
    baseArray = np.asarray(baseImage, dtype=np.uint8)

    labels = labelArray[:,:,3]
    imageWidth = labels.shape[0]
    imageHeight = labels.shape[1]
    
    plt.ion()
    plt.figure('test')
    plt.imshow(labelImage)

    

    
