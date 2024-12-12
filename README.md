
# **언어 모델 개발**

<img src="https://contents.kyobobook.co.kr/sih/fit-in/458x0/pdt/9791162241745.jpg" align="right" alt="밑바닥부터 시작하는 딥러닝2 표지" width="120" height="178">

이 프로젝트는 딥러닝 전공 수업의 개인 과제로, 주교재로 **밑바닥부터 시작하는 딥러닝2**을 사용하여 진행되고 있습니다.

주교재를 통해 학습한 기본적인 딥러닝 개념과 기법을 바탕으로, 언어 모델을 개발하는 프로젝트를 수행할 예정입니다.

개발할 모델은 사용자의 텍스트 입력으로 질문을 받아, 응답 문장을 생성하는 챗봇(Chatbot)입니다. 지난 번 과제와 같이, 기타 모델을 사용하지 않고 **밑바닥부터 직접 구현**할 예정입니다.

<H1 align="center">
  Project Overview
</H1>


## **프로젝트 개요**

### **목표**
- 본 프로젝트의 목적은 [제공된 데이터셋](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71773)을 활용해 한글 텍스트 응답을 생성하는 언어 모델을 개발하는 것입니다.
- 구체적인 학습 방법은 추후 결정될 예정이며, 주교재의 개념과 기법을 활용하여 구현됩니다.

### **현재 진행 상황**
- **사용 데이터셋 분석**: 데이터셋의 구조와 특징을 파악하고, 언어 모델에 적합한 형태로 변환 작업 여부를 고민 중입니다.

## **데이터셋 설명**

### **데이터 구조 및 형식**
- 데이터는 tsv와 json 형식으로 제공되며, 각 파일에는 다음과 같은 정보가 포함되어 있습니다.
  - **tsv 파일**: 대화 데이터를 간단하게 분석하고 모델 학습 데이터로 빠르게 사용할 수 있는 형태입니다. 데이터가 간단한 구조로 되어 있어 스프레드시트, 데이터베이스, 또는 Python의 pandas 등으로 빠르게 읽고 처리할 수 있습니다.
  - **json 파일**: 대화 전체를 하나의 데이터로 취급하므로 발화 간의 맥락을 파악하거나, 대화 주제(topic) 및 발화자(speaker) 정보를 함께 분석할 수 있습니다.발화별로 세부적인 평가 항목(예: 언어학적 수용성, 일관성 등)을 포함하고 있어, 모델의 품질 평가 및 개선에 직접 활용 가능합니다.

| 특성         | TSV                           | JSON                                  |
|--------------|-------------------------------|---------------------------------------|
| 구조         | 간단한 평면 구조             | 계층적, 복잡한 구조                  |
| 포함 정보     | 발화별 기본 정보 (ID, 텍스트, 발화자 등) | 대화 메타데이터, 발화 평가, 대화 요약 등 |
| 활용 목적     | 간단한 분석, 통계 처리, 기초 학습 데이터 제공 | 맥락 이해, 모델 평가 및 성능 분석, 시뮬레이션 |
| 처리 난이도   | 쉬움                          | 상대적으로 복잡                      |


### **활용한 데이터**
- 추후 업데이트 예정

---

## **디렉토리 구조**

```plaintext
deeplearning
├─ .gitignore
├─ README.md
├─ main.py                   # 메인 실행 스크립트
|
├─ common                    # 공통 유틸리티 및 함수
│  ├─ 추가
│  └─ 예정
|
├─ models                    # 모델 정의
│  └─ 추가 예정
|
└─ scripts                   # 데이터 처리 및 로딩 관련 스크립트
   ├─ dataset_loader.py      # 데이터 로더
   ├─ dataset_splitter.py    # 데이터 분할기
   └─ preprocess.py          # 전처리 스크립트

```
[텍스트 데이터셋 다운로드 링크](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71773)

이 프로젝트는 *데이터 전처리*, *모델 학습*, *성능 최적화* 의 전체 과정을 디렉토리 구조로 나누어 관리하고 있습니다.

**common** 디렉토리는 여러 모듈에서 재사용 가능한 유틸리티 및 함수들을 포함하고 있으며, <br>
**models** 디렉토리는 사용할 딥러닝 모델을 정의합니다. <br>
**scripts** 디렉토리는 데이터 로딩과 전처리 과정을 담당하여 모델이 학습할 수 있는 형태로 데이터를 준비합니다.

---

## **방법론**

### **1. 데이터 준비 및 전처리**
- **데이터 분할**: 이미 분할되어 있는 Sublabel, Training, Validation 세트를 활영하여 학습, 검증, 평가에 활용.
- **전처리**: 추후 업데이트 예정

### **2. 모델 설계**
- **추가 예정**

### **3. 실험 설정 및 평가**
- **추가 예정**

이 README는 **현재 개발 진행 상황**에 맞추어 작성되었으며, 프로젝트가 진행됨에 따라 업데이트될 예정입니다.
