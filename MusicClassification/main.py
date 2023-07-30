# nsml: pytorch/pytorch:nightly-runtime-cuda9.2-cudnn7

from pprint import PrettyPrinter
import os
import yaml
import argparse
from trainer import Trainer
import nsml

class Initializer(object):
    def __init__(self, args):
        with open(args.config_file) as f:
            config = yaml.load(f)
            config['dataset_root_1'] = os.path.join(nsml.DATASET_PATH[0], 'train')
            config['dataset_root_2'] = os.path.join(nsml.DATASET_PATH[1], 'train')

        pp = PrettyPrinter(indent=4)
        pp.pprint(config)

        self.trainer = Trainer(config, args.mode)
        self.config = config

        self.bind_nsml()

        if args.pause:
            nsml.paused(scope=locals())

    def bind_nsml(self):
        import torch
        import json
        trainer = self.trainer
        config = self.config

        def save(model_dir):
            checkpoint = {
                'model0': trainer.model0.state_dict(),
                'model1': trainer.model1.state_dict(),
                'model2': trainer.model2.state_dict(),
                'model3': trainer.model3.state_dict(),
                'label_map': json.dumps(trainer.label_map),
                'config': json.dumps(config),
            }
            torch.save(checkpoint, os.path.join(model_dir, 'model'))

        def load(model_dir, **kwargs):
            fpath = os.path.join(model_dir, 'model')
            checkpoint = torch.load(fpath)
            trainer.model0.load_state_dict(checkpoint['model0'])
            trainer.model1.load_state_dict(checkpoint['model1'])
            trainer.model2.load_state_dict(checkpoint['model2'])
            trainer.model3.load_state_dict(checkpoint['model3'])
            trainer.label_map = json.loads(checkpoint['label_map'])
            self.config = json.loads(checkpoint['config'])
            print('Model loaded')

        def infer(test_dir, **kwargs):
            print('test_dir:',test_dir)
            return trainer.run_evaluation('1', test_dir)

        nsml.bind(save=save, load=load, infer=infer)


    def run(self):
        print('Starting Training')
        self.trainer.run()


if __name__ == '__main__':
    with open("test.dat.txt", "a") as f:
        print("writefile")
        f.write("new line\n")

    parser = argparse.ArgumentParser()
    # Needed for nsml submit
    parser.add_argument('--pause', type=int, default=0)
    parser.add_argument('--iteration', type=str, default=0, help='checkpoint loaded')
    parser.add_argument('--mode', type=str, default='train')

    # User argument
    parser.add_argument('--config_file', type=str, default='config.yaml')
    args = parser.parse_args()

    initializer = Initializer(args)
    if args.mode == 'train':
        initializer.run()
