
# **녹내장 진단 모델 개발**

<img src="https://image.aladin.co.kr/product/10155/54/cover500/e896848463_1.jpg" align="right" alt="밑바닥부터 시작하는 딥러닝1 표지" width="120" height="178">

이 프로젝트는 딥러닝 전공 수업의 개인 과제로, 주교재로 **밑바닥부터 시작하는 딥러닝1**을 사용하여 진행되고 있습니다.

주교재를 통해 학습한 기본적인 딥러닝 개념과 기법을 바탕으로, CNN 기반 분류 모델을 적용하여 녹내장을 진단하는 실습 프로젝트입니다.

개발할 녹내장 진단 모델은 [고해상도 안저 이미지](https://www.kaggle.com/datasets/arnavjain1/glaucoma-datasets/data)를 분석하여 **녹내장 여부**를 판단하는 모델로, 기타 모델을 사용하지 않고 **밑바닥부터 직접 구현**할 예정입니다.

<H1 align="center">
  Project Overview
</H1>


## **1. 프로젝트 개요**

### **1.1. 목표**
- 본 프로젝트의 목적은 **안저 이미지**를 통해 **녹내장 여부**를 자동으로 진단할 수 있는 **머신러닝 모델**을 개발하는 것입니다.
- 안저 이미지에서 **시신경 유두(Optic Disc, OD)** 와 **시신경 함몰부(Optic Cup, OC)** 영역을 세그멘테이션하고, **Cup-to-Disc 비율(vCDR)** 을 계산하여 녹내장 여부를 예측하는 과정을 수행할 예정입니다.

### **1.2. 현재 진행 상황**
- **데이터 분석 및 준비 완료**: 데이터셋의 구조 및 특성을 분석하고, 모델 학습에 적합한 형태로 데이터를 준비했습니다.
- **세그멘테이션 정보 활용**: JSON 파일에서 제공된 OD와 OC 좌표를 통해 마스크 생성 작업을 수행하였습니다.
- **분류 모델 계획**: 세그멘테이션 결과를 활용한 CNN 기반의 녹내장 진단 분류 모델 개발을 계획 중입니다.

---

## **2. 디렉토리 구조**

```plaintext
/project-root/
├── /data/                    # 데이터 로드 및 전처리
│   ├── dataset_loader.py    
│   └── preprocess.py        
├── /G1020/                   # 캐글 데이터셋(직접 다운로드)
└── README.md                 # 프로젝트 개요
```
[캐글 데이터셋 다운로드 링크](https://www.kaggle.com/datasets/arnavjain1/glaucoma-datasets/data)

---

## **3. 기대 효과**

- **진단 정확도 향상**: 자동화된 세그멘테이션 및 진단을 통해 녹내장 진단의 정확성을 높일 수 있습니다.
- **의료 현장에서의 활용성**: 의료 전문가의 도움 없이 신속하고 정확하게 녹내장을 진단할 수 있는 도구로 활용 가능합니다.
- **의료 인공지능 기술 연구**: 녹내장 진단 기술을 통해 의료 AI 분야에서 활용될 수 있습니다.

---

이 README는 **현재 개발 진행 상황**에 맞추어 작성되었으며, 프로젝트가 진행됨에 따라 업데이트될 예정입니다.