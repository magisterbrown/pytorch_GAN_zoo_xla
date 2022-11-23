from .gan_trainer import GANTrainer
from ..pgan_xla import XLAProgressiveGAN
from .standard_configurations.pgan_config import _C
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import webdataset as wds
import torch
import numpy as np
import cv2
import pandas as pd

class PganXlaTrainer(GANTrainer):

    _defaultConfig = _C

    def getDefaultConfig(self):
        config = PganXlaTrainer._defaultConfig 

        config.maxIterAtScale = [
                      96000,
                      120000,
                      120000,
                      120000    ]
        return config

    def __init__(self, *args, **kwargs):

        GANTrainer.__init__(self, pathdb=None)

    def readTrainConfig(self, config):
        GANTrainer.readTrainConfig(self, config)
        self.scaleSanityCheck()

    def scaleSanityCheck(self):

        # Sanity check
        n_scales = min(len(self.modelConfig.depthScales),
                       len(self.modelConfig.maxIterAtScale),
                       len(self.modelConfig.iterAlphaJump),
                       len(self.modelConfig.alphaJumpVals))

        self.modelConfig.depthScales = self.modelConfig.depthScales[:n_scales]
        self.modelConfig.maxIterAtScale = self.modelConfig.maxIterAtScale[:n_scales]
        self.modelConfig.iterAlphaJump = self.modelConfig.iterAlphaJump[:n_scales]
        self.modelConfig.alphaJumpVals = self.modelConfig.alphaJumpVals[:n_scales]

        self.modelConfig.size_scales = [4]
        for scale in range(1, n_scales):
            self.modelConfig.size_scales.append(
                self.modelConfig.size_scales[-1] * 2)

        self.modelConfig.n_scales = n_scales



    def getDataset(self, scale, size=None):
        return list() 

    def initModel(self):
        config = {key: value for key, value in vars(self.modelConfig).items()}
        config["depthScale0"] = self.modelConfig.depthScales[0]
        self.model = XLAProgressiveGAN(**config)

    def train(self):
        n_scales = len(self.modelConfig.depthScales)
        flags = {
                'seed': 420
                }
        print(n_scales)
        print('XLA train')
        xmp.spawn(self.mpf, args=(flags), nprocs=8, start_method='fork')

    def mpf(self, index, flags):
        torch.manual_seed(420)
        xm.rendezvous('init')
        self.model.meme+=1
        device = xm.xla_device()  
        self.model.updateSolverDeviceTpu(device)
        n_scales = len(self.modelConfig.depthScales)
        store = list()#pd.DataFrame(columns=['scale', 'step', 'alpha'])

        alcal = AlphaCalc(self.modelConfig)

        for scale in range(self.startScale, n_scales):
            dbLoader = self.getDBLoader(scale)
            

            shiftIter = 0

            shiftAlpha = 0

            while not alcal.should_break(shiftIter, scale):

                #self.indexJumpAlpha = shiftAlpha
                status, sizeDB, store = self.trainOnEpoch(dbLoader, scale, shiftIter, self.modelConfig.maxIterAtScale[scale], store, alcal)

                if xm.is_master_ordinal():
                    print(sizeDB)
                if not status:
                    return False

                shiftIter = sizeDB
                #while shiftAlpha < len(self.modelConfig.iterAlphaJump[scale]) and \
                #        self.modelConfig.iterAlphaJump[scale][shiftAlpha] < shiftIter:
                #    shiftAlpha += 1

            if scale == n_scales - 1:
                break

            #self.model.addScale(self.modelConfig.depthScales[scale + 1])
        if xm.is_master_ordinal():
            store = pd.DataFrame(store)
            store.to_csv('data/alphalog.csv')

    def trainOnEpoch(self, dbLoader, scale, shiftIter, maxIter, store, alcal):
        steps = 0
        i = shiftIter
        bs = 4
        datas = [[np.ones((bs,)), 0]]*(6249//bs)
        #b0 = next(iter(dbLoader))
        #bs = b0[0].shape[0]
        #datas = [b0]*(6249//bs)
        #for item, data in enumerate(dbLoader, 0):
        for item, data in enumerate(datas, 0):
            incr=(data[0].shape[0]*8)//16
            i+=incr
            inputs_real, labels= data
            store.append({'step':i,'alpha':self.model.alpha, 'scale': scale})

            inputs_real = self.inScaleUpdate(inputs_real, alcal.get_alpha(i, scale))
            allLosses = self.model.optimizeParameters(inputs_real,
                                                          inputLabels=labels)
            if xm.is_master_ordinal():
                print(f'Step {i} scale {scale} alpha {self.model.alpha}')
            if alcal.should_break(i, scale):
               return True, i, store
        return True, i, store
    
    def getDBLoader(self, scale):
        size = pow(2,scale+1)

        shard = f'gs://monet-cool-gan/cifar_fortpu/b_{xm.get_ordinal()}.tar'
        proc = Allproc('npy','lable',size)
        ds = wds.WebDataset(shard).decode().shuffle(12).map(proc.proc)
        loader = torch.utils.data.DataLoader(ds,num_workers=1, batch_size=4, drop_last=True)

        return loader

    def inScaleUpdate(self, input_real, alpha):

        #if self.indexJumpAlpha < len(self.modelConfig.iterAlphaJump[scale]):
        #    if iter == self.modelConfig.iterAlphaJump[scale][self.indexJumpAlpha]:
        #        alpha = self.modelConfig.alphaJumpVals[scale][self.indexJumpAlpha]
        #        self.indexJumpAlpha += 1
        self.model.updateAlpha(alpha)

        #if alpha > 0:
        #    low_res_real = F.avg_pool2d(input_real, (2, 2))
        #    low_res_real = F.upsample(
        #        low_res_real, scale_factor=2, mode='nearest')

        #    alpha = self.model.config.alpha
        #    input_real = alpha * low_res_real + (1-alpha) * input_real

        return input_real

class AlphaCalc:
    def __init__(self, config):
        self.maxIterAtScale = config.maxIterAtScale
        self.alphaNJumps = config.alphaNJumps
        self.alphaSizeJumps = config.alphaSizeJumps

    def get_alpha(self, step, scale):
        if scale==0:
            return 0

        steps = self.alphaNJumps[scale]*self.alphaSizeJumps[scale]
        if step>=steps:
            return 0

        return 1+step*(-1/steps)

    def should_break(self, step, scale):
        return step > self.maxIterAtScale[scale]


class Allproc:
    def __init__(self, key: str, lable: str, side: int):
        self.key = key
        self.side = side        
        self.lable = lable

        self.lab_num = ['airplane',  
                        'automobile',  
                        'bird',  
                        'cat',  
                        'deer',  
                        'dog',  
                        'frog',  
                        'horse',  
                        'ship',  
                        'truck']

    def proc(self, el):
        lable = el[self.lable]
        el = el[self.key]
        el = cv2.resize(el, (self.side,self.side))
        el = np.moveaxis(el,2,0).astype(np.float32)
        el = (el/127.5)-1
        return el, self.lab_num.index(lable.decode())
