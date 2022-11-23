from .base_GAN import BaseGAN
import torch.optim as optim
from .utils.config import BaseConfig
from .networks.progressive_conv_net import GNet, DNet
import torch_xla.distributed.xla_multiprocessing as xmp

class XLAProgressiveGAN(BaseGAN):
    def __init__(self, 
                dimLatentVector=512,
                 depthScale0=512,
                 initBiasToZero=True,
                 leakyness=0.2,
                 perChannelNormalization=True,
                 miniBatchStdDev=False,
                 equalizedlR=True,
             **kwargs):
        self.meme = 0
        self.alpha = 0
        if not 'config' in vars(self):
            self.config = BaseConfig()
        print(kwargs)

        self.config.depthScale0 = depthScale0
        self.config.initBiasToZero = initBiasToZero
        self.config.leakyReluLeak = leakyness
        self.config.depthOtherScales = []
        self.config.perChannelNormalization = perChannelNormalization
        self.config.alpha = 0
        self.config.miniBatchStdDev = miniBatchStdDev
        self.config.equalizedlR = equalizedlR

        BaseGAN.__init__(self, dimLatentVector, **kwargs)
        pass

    def updateAlpha(self, alpha):

       self.netG.setNewAlpha(alpha)
       self.netD.setNewAlpha(alpha)

       self.alpha = alpha

    def buildNoiseData(self, n_samples, inputLabels=None):
        return None, None
    
    def getNetG(self):

        gnet = GNet(self.config.latentVectorDim,
                    self.config.depthScale0,
                    initBiasToZero=self.config.initBiasToZero,
                    leakyReluLeak=self.config.leakyReluLeak,
                    normalization=self.config.perChannelNormalization,
                    generationActivation=self.lossCriterion.generationActivation,
                    dimOutput=self.config.dimOutput,
                    equalizedlR=self.config.equalizedlR)

        # Add scales if necessary
        for depth in self.config.depthOtherScales:
            gnet.addScale(depth)

        # If new scales are added, give the generator a blending layer
        if self.config.depthOtherScales:
            gnet.setNewAlpha(self.config.alpha)

        return xmp.MpModelWrapper(gnet)

    def getNetD(self):

        dnet = DNet(self.config.depthScale0,
                    initBiasToZero=self.config.initBiasToZero,
                    leakyReluLeak=self.config.leakyReluLeak,
                    sizeDecisionLayer=self.lossCriterion.sizeDecisionLayer +
                    self.config.categoryVectorDim,
                    miniBatchNormalization=self.config.miniBatchStdDev,
                    dimInput=self.config.dimOutput,
                    equalizedlR=self.config.equalizedlR)

        # Add scales if necessary
        for depth in self.config.depthOtherScales:
            dnet.addScale(depth)

        # If new scales are added, give the discriminator a blending layer
        if self.config.depthOtherScales:
            dnet.setNewAlpha(self.config.alpha)

        return xmp.MpModelWrapper(dnet)

    def getOptimizerD(self):
        return optim.Adam(filter(lambda p: p.requires_grad, self.netD.parameters()),
                          betas=[0, 0.99], lr=self.config.learningRate)

    def getOptimizerG(self):
        return optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()),
                          betas=[0, 0.99], lr=self.config.learningRate)

    def optimizeParameters(self, input_batch, inputLabels=None):
        n_samples = input_batch.size()[0]
        allLosses = dict()

        # Update the discriminator
        self.optimizerD.zero_grad()

        # #1 Real data
        predRealD = self.netD(input_batch, False)

        # Classification criterion
        allLosses["lossD_classif"] = \
            self.classificationPenalty(predRealD,
                                       self.realLabels,
                                       self.config.weightConditionD,
                                       backward=True)

        lossD = self.lossCriterion.getCriterion(predRealD, True)
        allLosses["lossD_real"] = lossD.item()

        # #2 Fake data
        inputLatent, targetRandCat = self.buildNoiseData(n_samples)
        predFakeG = self.netG(inputLatent).detach()
        predFakeD = self.netD(predFakeG, False)

        lossDFake = self.lossCriterion.getCriterion(predFakeD, False)
        allLosses["lossD_fake"] = lossDFake.item()
        lossD += lossDFake

        # #3 WGANGP gradient loss
        if self.config.lambdaGP > 0:
            allLosses["lossD_Grad"] = WGANGPGradientPenalty(self.real_input,
                                                            predFakeG,
                                                            self.netD,
                                                            self.config.lambdaGP,
                                                            backward=True)

        # #4 Epsilon loss
        if self.config.epsilonD > 0:
            lossEpsilon = (predRealD[:, 0] ** 2).sum() * self.config.epsilonD
            lossD += lossEpsilon
            allLosses["lossD_Epsilon"] = lossEpsilon.item()



        lossD.backward(retain_graph=True)
        finiteCheck(self.netD.parameters())
        self.optimizerD.step()

        return allLosses



    def updateSolverDeviceTpu(self, device):

        self.netD = self.netD.to(device)
        self.netG = self.netG.to(device)

        self.optimizerD = self.getOptimizerD()
        self.optimizerG = self.getOptimizerG()

        self.optimizerD.zero_grad()
        self.optimizerG.zero_grad()


    def updateSolversDevice(self, buildAvG=True):
        pass

