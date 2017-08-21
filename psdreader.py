import glob
from psd_tools import PSDImage
from matplotlib import pyplot as plt
from sklearn import svm
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
    
    # plt.ion()
    # plt.figure('test')
    # plt.imshow(labelImage)

    clf = svm.SVC(probability=False,  # cache_size=200,
              kernel="rbf", C=2.8, gamma=.0073)

    dataArray = baseArray.reshape([-1, 3])
    if len(featuresArray) == 0:
        featuresArray = dataArray
    else:
        featuresArray = np.append(featuresArray, dataArray, axis=0)
    labelsArray = labels.reshape([-1, 1]).ravel()
    if len(labellingArray) == 0:
        labellingArray = labelsArray
    else:
        labellingArray = np.append(labellingArray, labelsArray,axis=0)

    # from IPython.terminal import embed; ipshell=embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())

start = time.time()
clf.fit(featuresArray, labellingArray)
end = time.time()

print "running time is {0}".format(end-start)

print(clf.predict(np.asarray([0,0,0]).reshape(1,-1)))

