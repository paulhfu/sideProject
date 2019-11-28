from easydict import EasyDict as edict

general = edict()
model = edict()

# Model params
model.freezeVgg = False
model.loadFromName = 'vgg3d_best1.pth'
model.saveToName = 'vgg3d_best1.pth'
# General
general.usePascalVOC = True
general.resume = True
general.useCuda = True
general.trainBatchSize = 1
general.testBatchSize = 1
general.numEpochs = 20
general.lossWeights = [1.0, 1.0]
general.noiseLevel = 0.2
general.augmentImgs = False
# Dset params
general.dataIdx = [1]
general.shuffleClasses = True
general.shots = 2
# env params
general.checkpointDir = './checkpoints/models'
general.checkpointSaveDir = './checkpoints/models'
general.logDir = './logs/vgg3d_best1'
general.dsetDir = './data/'
