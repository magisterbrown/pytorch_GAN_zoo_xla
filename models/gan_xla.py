from .progressive_gan import ProgressiveGAN
from .utils.utils import  finiteCheck
import torch_xla.core.xla_model as xm
import torch
from .loss_criterions.gradient_losses import WGANGPGradientPenalty
import torch_xla.distributed.xla_multiprocessing as xmp

class XlaGan(ProgressiveGAN):
    def  __init__(self,gnet=None,dnet=None, *args, **kwargs):
        self.gnet=gnet
        self.dnet=dnet
        super().__init__(*args, **kwargs) 
    def optimizeParameters(self, input_batch, inputLabels=None):

        allLosses = dict()
        self.real_input = input_batch.to(self.device) 
        n_samples = self.real_input.size()[0]
        
        self.optimizerD.zero_grad()

        predRealD = self.netD(self.real_input, False)

        lossD = self.lossCriterion.getCriterion(predRealD, True)
        allLosses['lossD_real'] = lossD.item()
        # #2 Fake data
        inputLatent, targetRandCat = self.buildNoiseData(n_samples)
        predFakeG = self.netG(inputLatent).detach()
        predFakeD = self.netD(predFakeG, False)

        lossDFake = self.lossCriterion.getCriterion(predFakeD, False)
        lossD += lossDFake
        allLosses['lossDFake'] = lossDFake.item()

         # #3 WGANGP gradient loss
        if self.config.lambdaGP > 0:
            allLosses["lossD_Grad"] = WGANGPGradientPenalty(self.real_input,
                                                            predFakeG,
                                                            self.netD,
                                                            self.config.lambdaGP,
                                                            backward=True)

        # #4 Epsilon loss
        if self.config.epsilonD > 0:
            lossEpsilon = torch.mul(torch.sum(torch.pow(predRealD[:, 0],2)), self.config.epsilonD)
            lossD += lossEpsilon
            allLosses["lossD_Epsilon"] = lossEpsilon.item()


        lossD.backward(retain_graph=True)
        finiteCheck(self.getOriginalD().parameters())
        xm.optimizer_step(self.optimizerD)

        lossD = 0
        # Update the generator
        self.optimizerG.zero_grad()
        self.optimizerD.zero_grad()

        inputNoise, targetCatNoise = self.buildNoiseData(n_samples)
        predFakeG = self.netG(inputNoise)

        # #2 Status evaluation
        predFakeD = self.netD(predFakeG, False)

        # #3 GAN criterion
        lossGFake = self.lossCriterion.getCriterion(predFakeD, True)
        allLosses['lossGFake'] = lossGFake.item()
        lossGFake.backward(retain_graph=True)

        finiteCheck(self.getOriginalG().parameters())
        xm.optimizer_step(self.optimizerG)

        # Update the moving average if relevant
        for p, avg_p in zip(self.getOriginalG().parameters(),
                            self.getOriginalAvgG().parameters()):
            avg_p.mul_(0.999).add_(alpha=0.001, other=p.data)

        xm.mark_step()
        return allLosses

    def getNetG(self):
        if not self.gnet:
            return super().getNetG()
        return self.gnet
        #return xmp.MpModelWrapper(super().getNetG())

    def getNetD(self):
        if not self.dnet:
            return super().getNetD()
    #def getNetD(self):
    #    return xmp.MpModelWrapper(super().getNetD())























