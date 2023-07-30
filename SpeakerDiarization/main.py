"""Entry point code to evaluate speaker diarziation task on NSML."""
import argparse
import json
import os

import nsml

from nssd.speaker_diarization import SpeakerDiarization


def _build_config(args):
    # load configurations
    config = json.loads(open(args.config, 'r').read())
    return config


def bind_model(model):
    """Utility function to make the model run on NSML.

    Args:
        model: SpeakerDiarization instance.
    """

    # pylint: disable=W0613
    def load(dirname, **kwargs):
        # Workaround for escaping nsml error like below:
        # NSML Warning) The folder you are trying to save is empty.
        outfile = os.path.join(dirname, 'test.txt')
        with open(outfile, 'r') as outf:
            print(outf.read())

    def save(dirname, **kwargs):
        # Workaround for escaping nsml error like below:
        # NSML Warning) The folder you are trying to save is empty.
        outfile = os.path.join(dirname, 'test.txt')
        with open(outfile, 'w') as outf:
            outf.write('Hello World!')

    def infer(input_pcm):
        """Local inference function for NSML infer.

        Args:
            input_pcm: list of string, path of WAV file.

        Returns:
            list of 2-tuples of (dummy_prob, sel_tuple).
            sel_tuple is 3-tuples of (start_time, duration, speaker_id).
        """
        run_result = model.run([input_pcm])
        _, sel_tuples = run_result.export_nssd_data()
        sel_tuples = sel_tuples[0]
        return [(0.00, sel_tuples)]

    nsml.bind(load=load, save=save, infer=infer)


def main(args):
    """Run evaluation."""
    config = _build_config(args)
    speaker_diarization = SpeakerDiarization(**config)
    bind_model(speaker_diarization)

    if args.pause == 1:
        nsml.paused(scope=locals())

    nsml.save(args.checkpoint_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate speaker diarization task.")
    parser.add_argument('--config',
                        dest='config',
                        type=str,
                        help='configuration file for speaker diarization.',
                        default='configs/speaker_diarization.conf')
    parser.add_argument('--checkpoint_name',
                        dest='checkpoint_name',
                        type=str,
                        default='base')
    # arguments for nsml
    parser.add_argument('--model', type=str, default='train')
    parser.add_argument('--pause', type=int, default=0)
    main(parser.parse_args())
