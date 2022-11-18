from .gan_trainer import GANTrainer
from ..pgan_xla import XLAProgressiveGAN
from .standard_configurations.pgan_config import _C
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import webdataset as wds
import torch
import numpy as np
import cv2

class PganXlaTrainer(GANTrainer):

    _defaultConfig = _C

    def getDefaultConfig(self):
        return PganXlaTrainer._defaultConfig
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

        for scale in range(self.startScale, n_scales):
            dbLoader = self.getDBLoader(scale)
            

            shiftIter = 0

            shiftAlpha = 0

            while shiftIter < self.modelConfig.maxIterAtScale[scale]:

                self.indexJumpAlpha = shiftAlpha
                status, sizeDB = self.trainOnEpoch(dbLoader, scale, shiftIter, self.modelConfig.maxIterAtScale[scale])

                if xm.is_master_ordinal():
                    print(sizeDB)
                if not status:
                    return False

                shiftIter = sizeDB
                while shiftAlpha < len(self.modelConfig.iterAlphaJump[scale]) and \
                        self.modelConfig.iterAlphaJump[scale][shiftAlpha] < shiftIter:
                    shiftAlpha += 1

            if scale == n_scales - 1:
                break

            #self.model.addScale(self.modelConfig.depthScales[scale + 1])

    def trainOnEpoch(self, dbLoader, scale, shiftIter, maxIter):
        steps = 0
        i = shiftIter
        bs = 4
        datas = [[np.ones((bs,)), 0]]*(6249//bs)
        #for item, data in enumerate(dbLoader, 0):
        for item, data in enumerate(datas, 0):
            incr=(data[0].shape[0]*8)//16
            i+=incr
            inputs_real, labels= data

            inputs_real = self.inScaleUpdate(i, scale, inputs_real)
            allLosses = self.model.optimizeParameters(inputs_real,
                                                          inputLabels=labels)
            if xm.is_master_ordinal():
                print(f'Step {i} alpha {self.model.alpha}')
            if i >= maxIter:
               return True, maxIter
            break
        return True, i
    
    def getDBLoader(self, scale):
        size = pow(2,scale+1)

        shard = f'gs://monet-cool-gan/cifar_fortpu/b_{xm.get_ordinal()}.tar'
        proc = Allproc('npy','lable',size)
        ds = wds.WebDataset(shard).decode().shuffle(12).map(proc.proc)
        loader = torch.utils.data.DataLoader(ds, batch_size=4, drop_last=True)

        return loader

    def inScaleUpdate(self, iter, scale, input_real):

        if self.indexJumpAlpha < len(self.modelConfig.iterAlphaJump[scale]):
            if iter == self.modelConfig.iterAlphaJump[scale][self.indexJumpAlpha]:
                alpha = self.modelConfig.alphaJumpVals[scale][self.indexJumpAlpha]
                self.model.updateAlpha(alpha)
                self.indexJumpAlpha += 1

        if self.model.config.alpha > 0:
            low_res_real = F.avg_pool2d(input_real, (2, 2))
            low_res_real = F.upsample(
                low_res_real, scale_factor=2, mode='nearest')

            alpha = self.model.config.alpha
            input_real = alpha * low_res_real + (1-alpha) * input_real

        return input_real


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
