
# **녹내장 진단 모델 개발**

<img src="https://image.aladin.co.kr/product/10155/54/cover500/e896848463_1.jpg" align="right" alt="밑바닥부터 시작하는 딥러닝1 표지" width="120" height="178">

이 프로젝트는 딥러닝 전공 수업의 개인 과제로, 주교재로 **밑바닥부터 시작하는 딥러닝1**을 사용하여 진행되고 있습니다.

주교재를 통해 학습한 기본적인 딥러닝 개념과 기법을 바탕으로, CNN 기반 분류 모델을 적용하여 녹내장을 진단하는 실습 프로젝트입니다.

개발할 녹내장 진단 모델은 [고해상도 안저 이미지](https://www.kaggle.com/datasets/arnavjain1/glaucoma-datasets/data)를 분석하여 **녹내장 여부**를 판단하는 모델로, 기타 모델을 사용하지 않고 **밑바닥부터 직접 구현**할 예정입니다.

<H1 align="center">
  Project Overview
</H1>


## **프로젝트 개요**

### **목표**
- 본 프로젝트의 목적은 **안저 이미지**를 통해 **녹내장 여부**를 자동으로 진단할 수 있는 **머신러닝 모델**을 개발하는 것입니다.
- 안저 이미지에서 **시신경 유두(Optic Disc, OD)** 와 **시신경 함몰부(Optic Cup, OC)** 영역을 세그멘테이션하고, **Cup-to-Disc 비율(vCDR)** 을 계산하여 녹내장 여부를 예측하는 과정을 수행할 예정입니다.

### **현재 진행 상황**
- **데이터 분석 및 준비 완료**: 데이터셋의 구조 및 특성을 분석하고, 모델 학습에 적합한 형태로 데이터를 준비했습니다.
- **세그멘테이션 정보 활용**: OD와 OC 좌표를 사용해 vCDR 계산을 완료하고, 해당 데이터 포맷을 모델에 입력할 수 있는 형태로 변환했습니다.
- **분류 모델 계획**: 세그멘테이션 결과와 이미지 데이터를 결합한 CNN 기반의 진단 모델을 설계하고 있으며, Optuna를 사용해 하이퍼파라미터 최적화를 진행 중입니다.

## **데이터셋 설명**

### **이미지 종류**
Kaggle G1020 안저 이미지 데이터셋에는 다음과 같은 종류의 이미지가 포함되어 있습니다:
1. **Original Images**: 원본 안저 이미지.
2. **Cropped Images**: 시신경 주변 영역만을 잘라낸 이미지.
3. **Nerve Removed Images**: 신경을 제거하여 안저 영역만을 강조한 이미지.
4. **Square Images**: 고정된 크기로 변환한 정사각형 이미지.

### **활용한 데이터**
- 본 프로젝트에서는 **Cropped**, **Nerve Removed**, **Square** 세 종류의 이미지를 사용하였습니다. 
- 각 이미지는 CNN 네트워크에 입력되어 각기 다른 방식으로 특징을 추출한 후, **vCDR** 값과 함께 결합하여 최종 예측 모델에 사용됩니다.

---

## **디렉토리 구조**

```plaintext
deeplearning
├─ .gitignore
├─ README.md
├─ main.py                   # 메인 실행 스크립트
├─ run_optuna.py             # Optuna를 사용한 하이퍼파라미터 탐색
|
├─ common                    # 공통 유틸리티 및 함수
│  ├─ functions.py
│  ├─ gradient.py
│  ├─ layers.py
│  ├─ optimizer.py
│  ├─ trainer.py
│  └─ util.py
|
├─ models                    # 모델 정의
│  └─ multimodal_net.py      # 멀티모달 입력을 처리하는 모델
|
└─ scripts                   # 데이터 처리 및 로딩 관련 스크립트
   ├─ dataset_loader.py      # 데이터 로더
   ├─ dataset_splitter.py    # 데이터 분할기
   └─ preprocess.py          # 전처리 스크립트

```
[캐글 데이터셋 다운로드 링크](https://www.kaggle.com/datasets/arnavjain1/glaucoma-datasets/data)

이 프로젝트는 *데이터 전처리*, *모델 학습*, *성능 최적화* 의 전체 과정을 디렉토리 구조로 나누어 관리하고 있습니다.

**common** 디렉토리는 여러 모듈에서 재사용 가능한 유틸리티 및 함수들을 포함하고 있으며, <br>
**models** 디렉토리는 다양한 딥러닝 모델을 정의합니다. <br>
**scripts** 디렉토리는 데이터 로딩과 전처리 과정을 담당하여 모델이 학습할 수 있는 형태로 데이터를 준비합니다.

---

## **방법론**

### **1. 데이터 준비 및 전처리**
- **데이터 분할**: Train (70%), Validation (15%), Test (15%) 세트로 분할하여 학습, 검증, 평가에 활용.
- **전처리**: 이미지 크기를 224x224로 리사이즈하고, 정규화(normalization) 처리. JSON 파일에서 **OD/OC 좌표**를 활용해 **vCDR** 값을 계산.

### **2. 모델 설계**
- **모델 구조**: **MultiLayerNetExtend**를 확장한 **MultimodalNet** 모델 사용.
  - **세 가지 이미지 입력**: Cropped, Nerve Removed, Square 이미지 데이터를 CNN 네트워크로 처리.
  - **특징 결합**: 각 CNN에서 추출한 특징 벡터를 결합하고, **vCDR** 값을 추가하여 최종 예측 수행.
  - **드롭아웃 및 배치 정규화**: 학습의 안정성 및 일반화 성능을 높이기 위해 적용.

### **3. 실험 설정 및 평가**
- **손실 함수**: 이진 분류 문제에 맞게 Cross-Entropy 손실 함수를 사용.
- **옵티마이저**: Adam 옵티마이저 사용, 학습률 0.001.
- **하이퍼파라미터 튜닝**: **Optuna**를 통해 학습률, 배치 크기, 드롭아웃 비율 등을 최적화.

이 README는 **현재 개발 진행 상황**에 맞추어 작성되었으며, 프로젝트가 진행됨에 따라 업데이트될 예정입니다.