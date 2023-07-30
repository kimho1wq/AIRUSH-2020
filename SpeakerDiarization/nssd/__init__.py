import os
import sys

cwd = os.path.dirname(os.path.abspath(__file__))

SPEAKER_NET_PATH = os.path.join(cwd, '../third_party/SpeakerNet')
DSCORE_PATH = os.path.join(cwd, '../third_party/dscore')

sys.path.append(SPEAKER_NET_PATH)
sys.path.append(DSCORE_PATH)
