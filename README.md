
# 🎤 Inhale vs Exhale Audio Classification (Kaggle Challenge)

## 📌 Overview
이 프로젝트는 호흡 소리 오디오 데이터를 분석하여, 소리가 **들숨(Inhale)**인지 **날숨(Exhale)**인지 분류하는 이진 분류 모델을 개발하는 Kaggle 대회 과제를 다룹니다.  

---

## 🎯 목표
- `.wav` 오디오 데이터를 입력으로 받아 **Inhale (I)** 또는 **Exhale (E)** 를 판별  
- 오디오 신호의 특징을 추출하여 모델 학습에 활용  
- 다양한 모델링 시도를 통해 최적 성능 도출  

---

## 📂 데이터셋
- **train.csv**  
  - `ID`: 오디오 파일명  
  - `Target`: 'I' 또는 'E' (들숨/날숨 레이블)  
- **train/test 오디오 파일**: `.wav` 형식, 길이 약 3초  

---

## ⚙️ 데이터 전처리
1. **라벨 인코딩**  
   - Target: `I → 1`, `E → 0`  
   - 클래스 분포: 1:1 (4000개, Inhale 2000 / Exhale 2000)  

2. **오디오 처리**  
   - Resampling: 16kHz  
   - Mel-Spectrogram 변환 (`n_mels=128`)  
   - Log-scale 변환 (`power_to_db`)  
   - 고정 길이(200 frame) 중앙 패딩 or 잘라내기  

3. **데이터 저장**  
   - 모든 입력을 `(128, 200, 1)` 형태의 텐서로 정규화 후 `.npy` 파일 저장  

4. **예외 처리**  
   - 일부 손상된 오디오 → `try-except` 블록으로 처리  

---

## 🧪 모델링 과정

### 🔹 1차 시도 (Classical ML)
- Feature: **MFCC (80차원)**  
- Models: **XGBoost, SVM, MLP**  
- 결과: XGBoost 정확도 66.6%, Kaggle Score **0.71142**

---

### 🔹 2차 시도 (Basic CNN + Mel-Spectrogram)
- 입력: `(128, T, 1)` Mel-Spectrogram  
- 구조: Conv + Pooling 2층, Dropout 3개, Sigmoid 출력층  
- Optimizer: Adam, Loss: Binary Crossentropy  
- Accuracy: **69.6%**, Kaggle Score **0.69571**

---

### 🔹 3차 시도 (Improved CNN)
- 개선 사항  
  - CNN 깊이 확장 → 다양한 음성 패턴 학습  
  - Dropout 계층별 차등 적용 → 과적합 방지  
  - EarlyStopping → 불필요한 학습 방지  
  - ModelCheckpoint → 최고 성능 모델 저장  
- 결과  
  - Validation Accuracy: **76%**  
  - Kaggle Score: **0.75333**

---

## 📊 데이터 증강
- **SpecAugment**
  - Frequency Masking
  - Time Masking  
- **Gaussian Noise 추가**  
- → 모델 일반화 성능 향상 및 과적합 방지  

---

## 📈 최종 결과
- **최고 검증 정확도**: 76%  
- **Kaggle Public Score**: 0.75333 

---

## 🛠 기술 스택
- Python, NumPy, Pandas  
- Librosa (오디오 처리)  
- TensorFlow / Keras (CNN 모델링)  
- Scikit-learn (전처리 및 ML 모델)  

---
