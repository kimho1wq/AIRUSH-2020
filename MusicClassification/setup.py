#nsml: registry.navercorp.com/nsasr/cuda10.1-pytorch1.5
from setuptools import setup, find_packages

setup(
    name='NSSD',
    version='1.0',
    author='Jaesung Huh',
    author_email='jaesung.huh@navercorp.com',
    description='Naver Speech Speaker Diarization Toolkit',
    packages=find_packages(),
    install_requires=[
        'tabulate',
        'spectralcluster',
        'intervaltree',
        'webrtcvad'
        ,'cachetools==4.1.0'
        ,'fire==0.3.1'
        ,'h5py==2.7.1'
        ,'matplotlib==3.2.2'
        ,'scikit-learn==0.23.1'
        ,'numpy==1.19.0'
        ,'pandas==1.0.5'
        ,'pillow==7.1.2'
        ,'requests==2.24.0'
        ,'tqdm==4.46.1'
        ,'torch==1.5.0'
        ,'torchvision==0.5.0'
        ,'torchaudio'
        ,'adamp'
        ,'tqdm'
        ,'pyyaml'
    ]
)
