from .gan_trainer import GANTrainer
from ..pgan_xla import XLAProgressiveGAN
from .standard_configurations.pgan_config import _C
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

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
        print(self.modelConfig.depthScales)
        print(n_scales)
        print('XLA train')
        xmp.spawn(self.mpf, args=(), nprocs=8, start_method='fork')

    def mpf(self, index):
        #print(index)
        self.model.meme+=1
        device = xm.xla_device()  
        self.model.updateSolverDeviceTpu(device)
        #print(self.model.meme)
        



















