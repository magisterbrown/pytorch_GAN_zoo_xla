from .progressive_gan import ProgressiveGAN
from .utils.utils import  finiteCheck
import torch_xla.core.xla_model as xm

class XlaGan(ProgressiveGAN):
    def optimizeParameters(self, input_batch, inputLabels=None):

        self.real_input = input_batch.to(self.device) 
        n_samples = self.real_input.size()[0]
        
        self.optimizerD.zero_grad()

        predRealD = self.netD(self.real_input, False)

        lossD = self.lossCriterion.getCriterion(predRealD, True)
        print(lossD)
        # #2 Fake data
        inputLatent, targetRandCat = self.buildNoiseData(n_samples)
        predFakeG = self.netG(inputLatent).detach()
        predFakeD = self.netD(predFakeG, False)

        lossDFake = self.lossCriterion.getCriterion(predFakeD, False)
        lossD += lossDFake
        print(lossD)
        #print(self.config.lambdaGP )
        #print(self.config.epsilonD )
        #print(self.config.logisticGradReal )
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
        lossGFake = self.lossCriterion.getCriterion(predFakeD, True)
        lossGFake.backward(retain_graph=True)
        print(lossGFake)

        finiteCheck(self.getOriginalG().parameters())
        self.optimizerG.step()

        # Update the moving average if relevant
        for p, avg_p in zip(self.getOriginalG().parameters(),
                            self.getOriginalAvgG().parameters()):
            avg_p.mul_(0.999).add_(alpha=0.001, other=p.data)

        xm.mark_step()
