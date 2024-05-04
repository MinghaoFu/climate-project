import pytorch_lightning as pl
from pytorch_lightning import Trainer
import sys
sys.path.append('..')
from LiLY.modules.LatLinGau import LatentLinearGaussian
import argparse

from Caulimate.Data.SimLinGau import LinGauNoSuff
from Caulimate.Utils.Tools import load_yaml, dict_to_lass

def main(args):
    # Load dataset
    dataset = LinGauNoSuff(args.num, args.dim, args.degree, args.t_period, seed=args.seed)

    # Initialize model
    model = LatentLinearGaussian(args, dataset)

    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=args.epochs, gpus=args.gpus)

    # Train the model
    trainer.fit(model)

if __name__ == "__main__":

    args = load_yaml('/home/minghao.fu/workspace/climate/LiLY/configs/LatLinGau.yaml')

    main(args)