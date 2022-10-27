from .progressive_gan import ProgressiveGAN
from .utils.utils import  finiteCheck
import torch_xla.core.xla_model as xm
import torch
from .loss_criterions.gradient_losses import WGANGPGradientPenalty

class XlaGan(ProgressiveGAN):
    def optimizeParameters(self, input_batch, inputLabels=None):

        allLosses = dict()
        self.real_input = input_batch.to(self.device) 
        n_samples = self.real_input.size()[0]
        
        self.optimizerD.zero_grad()

        predRealD = self.netD(self.real_input, False)

        allLosses['lossD_real'] = self.lossCriterion.getCriterion(predRealD, True)
        # #2 Fake data
        inputLatent, targetRandCat = self.buildNoiseData(n_samples)
        predFakeG = self.netG(inputLatent).detach()
        predFakeD = self.netD(predFakeG, False)

        allLosses['lossDFake'] = self.lossCriterion.getCriterion(predFakeD, False)
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
            lossEpsilon = torch.mul(torch.sum(torch.pow(predRealD[:, 0],2)), self.config.epsilonD)
            lossD += lossEpsilon
            allLosses["lossD_Epsilon"] = lossEpsilon


        lossD.backward(retain_graph=True)
        finiteCheck(self.getOriginalD().parameters())
        self.optimizerD.step()

        lossD = 0
        # Update the generator
        self.optimizerG.zero_grad()
        self.optimizerD.zero_grad()

        inputNoise, targetCatNoise = self.buildNoiseData(n_samples)
        predFakeG = self.netG(inputNoise)

        # #2 Status evaluation
        predFakeD = self.netD(predFakeG, False)

        # #3 GAN criterion
        allLosses['lossGFake'] = self.lossCriterion.getCriterion(predFakeD, True)
        lossGFake.backward(retain_graph=True)

        finiteCheck(self.getOriginalG().parameters())
        self.optimizerG.step()

        # Update the moving average if relevant
        for p, avg_p in zip(self.getOriginalG().parameters(),
                            self.getOriginalAvgG().parameters()):
            avg_p.mul_(0.999).add_(alpha=0.001, other=p.data)

        xm.mark_step()
        return allLosses
