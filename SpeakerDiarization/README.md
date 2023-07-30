# Speaker diarization

> The process of partitioning an input audio stream into homogeneous segments according to the speaker identity. 
> It answers the question “who spoke when” in a multi-speaker environment. 

# Pipeline
1. Speech segmentation.
2. Audio embedding extraction.
3. Clustering.
4. Resegmentation.

# Baseline

제공되는 화자 분리 시스템 초기 모델의 구성은 아래와 같습니다.
* Speech segmentation: webrtcvad.
* Audio embedding extraction: SpeakerNet.
* Clustering: Agglomerative Hierarchical Clustering (AHC).
* Resegmentation: Speaker merging.

자세한 설정은 [SpeakerDiarization/configs/speaker_diarization.conf](SpeakerDiarization/configs/speaker_diarization.conf)에서 확인 및 수정 가능합니다.

# How to start?
NSML에서 동작을 위한 시작 스크립트는 [SpeakerDiarization/main.py](SpeakerDiarization/main.py)에 있습니다.

## Speech segmentation
관련 모듈은 [SpeakerDiarization/nssd/endpoint_detector.py](SpeakerDiarization/nssd/endpoint_detector.py)에 있습니다.

## Audio embedding extraction
SpeakerNet을 별도로 학습하고, 학습된 모델로 바꿔주면 됩니다. 
Weight 파일은 [SpeakerDiarization/third_party/SpeakerNet/models/weights/16k](SpeakerDiarization/third_party/SpeakerNet/models/weights/16k)에 있습니다.

### SpeakerNet 학습
```
cd SpeakerNetTrainer
./00-train_model.sh
```

### 학습 된 SpeakerNet 선택 및 다운로드
```
cd SpeakerNetTrainer
./01-get_model_list.sh <USER-ID> <SESSION_NUM> # <CHECKPOINT> 를 확인할 수 있음.
./02-download_model.sh <USER-ID> <SESSION_NUM> <CHECKPOINT>
```

## Clustering
관련 모듈은 [SpeakerDiarization/nssd/clustering.py](SpeakerDiarization/nssd/clustering.py)에 있습니다.

## Resegmentation
관련 코드는 [SpeakerDiarization/nssd/speaker_diarization.py#L191](https://oss.navercorp.com/NSASR3/AIRUSH2020_Baseline/blob/master/SpeakerDiarization/nssd/speaker_diarization.py#L191)에 있습니다.

# Data
## SpeakerNet
https://airush.nsml.navercorp.com/filesystem/vox
데이터 셋 이름: [vox](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
### 데이터 구성
* 음성 데이터: 약 6천명의 서로 다른 화자의 발성을 WAV 포맷으로 저장 
* 정답 스크립트: 각 발성 별로 화자 id 제공 (id0xxxx), vox/train/train_list.txt 참고

## SpeakerDiarization
* 1주차 데이터: rush8-1
* 2주차 데이터: rush8-2
* 3주차 데이터: rush8-3

### 데이터 구성
* 음성 데이터: 다수의 화자의 발성을 포함하는 10분 ~ 30분 정도 길이의 음성을 WAV 포맷으로 저장
* 정답 스크립트 (test_label): 각 음성 파일 내 어느 구간에서 어떤 화자가 발성하였는지 정보를 포함하는 [RTTM](https://github.com/nryant/dscore#rttm) 파일.

### Test label format
```
        {
            'subset1': {
                'sample1': <rttm-content>,
                'sample2': <rttm-content>,
            },
            'subset2': {
                'sample1': <rttm-content>,
                'sample2': <rttm-content>,
            },
        }
```

# Evaluation
```
cd SpeakerDiarization
./run_nsml.sh configs/speaker_diarization.conf <DATASET_NAME>
./submit_nsml.sh <USER-ID> <SESSION_NUM> <DATASET_NAME>
```

## Metric
Score = ([DER](https://github.com/nryant/dscore#diarization-error-rate) + [JER](https://github.com/nryant/dscore#jaccard-error-rate)) / 2

## Baseline
subset | score | evaluation (sec) | model load (sec)
-- | -- | -- | -- 
rush8-1  | 46.06 | 46.24 | 0.06 
rush8-2 | 45.50 | 95.99 | 0.06 
rush8-3 | 45.58 | 164.84 | 0.06 

소요 시간은 평가에 반영되지 않습니다 (참고용). 

# Appendix
* [NSML](https://n-clair.github.io/ai-docs/_build/html/ko_KR/index.html)
- [AI RUSH 2020](https://campaign.naver.com/airush)를 위한 화자 분리 시스템 초기 모델.
- 참고 논문: [https://arxiv.org/abs/1710.10468](https://arxiv.org/abs/1710.10468).


# 설정
JSON 포맷을 따르고, inference_config, diarization_config, epd_config 세 가지의 세부 설정으로 구성됩니다.
기본 설정은 [configs/speaker_diarization.conf](configs/speaker_diarization.conf)를 확인하시면 됩니다.
```python
configs/speaker_diarization.conf

{
    "inference_config": {
        "model_type": "ResNetSE_16k_150",
        "model_path": "third_party/SpeakerNet/models/weights/16k/ResNetSE_16k_150.model",
        "device" : "cuda",
        "batch_size": 512
    },
    "diarization_config": {
        "max_seg_ms": 1500,
        "shift_ms": 500,
        "method": "ahc",
        "num_cluster": "None",
        "normalize": true,
        "clustering_parameters": {
            "ahc_metric" : "cosine",
            "ahc_method": "complete",
            "ahc_criterion": "distance",
            "threshold": 0.95
        }
    },
    "epd_config": {
        "epd_mode": "webrtc",
        "resolution": 30,
        "voice_criteria": 0.7
    }
} 
```

- Possible options for config
	- inference_config
		- model_type: model type for SpeakerNet. Format must be **MODELFILENAME_SR_MAXFRAMES**.
		- model_path : weight file of the model.
        - device : 'cuda' for running with GPU, 'cpu' for running with CPU.
        - batch_size: Batch size of SpeakerNet, The number of speech chunks in one forward pass for speaker embedding extraction.
	- diarization_config
		- method : Clustering Method ['ahc']
		- max_seg_ms : window size when extracting speaker embedding(ms)
		- shift_ms : window step when extracting speaker embedding(ms)
		- num_cluster : 'None' indicates we don't know the number of cluster. Otherwise, must be integer
        - normalize : boolean, l2-normalize the speaker embedding if 'true'.
        - clustering_parameters:
            - You can pass the parameter to be used for clustering algorithm initialization as a JSON.
	- epd_config
		- epd_mode : [ webrtc ]
		- resolution : resolution of webrtc (10, 20, 30)
		- voice_criteria : voice_criteria when using webrtc (See epd_detector.py)