from easydict import EasyDict as edict

general = edict()
model = edict()

# Model params
model.freezeVgg = False
model.loadFromName = 'fs_model_best1.pth'
model.saveToName = 'fs_model_best2.pth'
# General
general.usePascalVOC = True
general.resume = True
general.useCuda = True
general.trainBatchSize = 3
general.testBatchSize = 3
general.numEpochs = 50
general.lossWeights = [1.0, 1.5]
general.noiseLevel = 0.2
general.augmentImgs = False
# Dset params
general.dataIdx = [1]
general.shuffleClasses = True
general.shots = 3
# env params
general.checkpointDir = './checkpoints/models'
general.checkpointSaveDir = './checkpoints/models'
general.logDir = './logs/fs_model_best2_wInit'
general.dsetDir = './data/'
