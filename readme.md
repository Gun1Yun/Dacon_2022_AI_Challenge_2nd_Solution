# 2022 AI 대학원 챌린지 

## Private : 0.7603 [전기톱원숭이] 팀
  
## 1. 개발환경 및 라이브러리
---
### 1.1 개발환경
```

OS:    Ubuntu 20.04.3 LTS
GPU:   GeForce RTX 3090 x4

```

### 1.2 라이브러리
- ```requirements.txt``` 참고

```
torch==1.9.0+cu111
fair-esm==0.4.2
pytorch-tabnet==3.1.1
scikit-learn==1.1.1
```
- ESM pretrained model 사용
- Papers : **Rives, A., Meier, J., Sercu, T., Goyal, S., Lin, Z., Liu, J., ... & Fergus, R. (2021). Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. Proceedings of the National Academy of Sciences, 118(15), e2016239118.**
- Github : https://github.com/facebookresearch/esm


## 2. 학습 완료 모델
---
- ```Tabnet_ESM_models/``` 디렉토리 아래에 있습니다.


## 3. 전체 실행 프로세스
---
### 3.1 실행 방법

- **Tabnet_pipeline.ipynb** 파일 실행하면 Embedding 추출부터 학습, 추론까지 진행됩니다.
- **Tabnet_inference.ipynb** 파일 실행하면 추출된 feature와 학습된 모델을 이용해 추론만 진행합니다.

### 3.2 실행 프로세스
- Preprocessing 단계
    - Embedding feature 추출을 위해 ESM encoder 모델을 활용해 Train, Test의 left antigen(64), epitope(128), right antigen(64) 임베딩을 추출해 ```./Embeddings/``` 디렉토리에 feature를 저장합니다. 이때 Embedding feature가 이미 존재한다면, 추출은 생략됩니다. 추출 시 Multi-GPU환경에서 실행하고 ```batch_size```를 수정하지 않아야 재현이 가능합니다.
    - CT-CTD feature를 사용하기 위해서는 ```CONFIG['CT_CTD_features']=True```로 설정합니다. 추출된 feature는 pkl 형태로 디렉토리에 저장됩니다.
    - CNT feature를 사용하기 위해서는 ```CONFIG['CNT_features]=True```로 설정합니다. 마찬가지로 추출된 feature는 pkl 형태로 디렉토리 내에 저장됩니다.

- Model training 단계
    - 추출된 Embedding 및 설정된 feature들을 이용해 Tabnet model을 학습시킵니다.
    - 이때, CT-CTD feature 혹은 CNT_feature에 PCA를 적용하려면 ```CONFIG['CT_CTD_PCA']``` 혹은 ```CONFIG['CNT_PCA']```를 설정해줍니다.
    - 학습은 5-fold cross validation을 이용해 학습되며 학습 결과는 Tabnet_ESM_models/log.txt에 저장됩니다.
    - 각 fold 마다 모델이 Tabnet_ESM_models에 저장됩니다.

- Inference 단계
    - 모든 fold 모델들을 이용해 추론 결과 probablity를 soft voting을 이용해 최종 추론을 결정합니다.
    - threshold를 설정하려면 ```CONFIG['threshold']```를 조절합니다.

## Apendix
- Tabnet 참고자료 : **Arik, S. Ö., & Pfister, T. (2021, May). Tabnet: Attentive interpretable tabular learning. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 35, No. 8, pp. 6679-6687).** 
  
- CT-CTD features 참고자료 : **Sharma, A., & Singh, B. (2020). AE-LGBM: Sequence-based novel approach to detect interacting protein pairs via ensemble of autoencoder and LightGBM. Computers in Biology and Medicine, 125, 103964.**
  
- CNT features 참고자료 : **Mu, Z., Yu, T., Liu, X., Zheng, H., Wei, L., & Liu, J. (2021). FEGS: a novel feature extraction model for protein sequences and its applications. BMC bioinformatics, 22(1), 1-15.**