from .gan_trainer import GANTrainer

from .standard_configurations.pgan_config import _C

class PganXlaTrainer(GANTrainer):

    _defaultConfig = _C
    def __init__(self, *kwargs):

        GANTrainer.__init__(self, pathDB=None)

    def train(self):
        print('XLA train')
