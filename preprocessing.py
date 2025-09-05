import pandas as pd
import numpy as np

# CSV 파일 경로 (너의 환경에 맞게 수정)
csv_path = "/content/train.csv"

# 1. CSV 파일 불러오기
train_df = pd.read_csv(csv_path)

# 2. I/E → 1/0으로 매핑
label_map = {'I': 1, 'E': 0}
labels = train_df['Target'].map(label_map).values

# 결과 출력
print("🔍 라벨 처리 결과 예시 (앞 10개):", labels[:10])
print("✅ 총 샘플 수:", len(labels))
print("📊 클래스 분포 (0: E, 1: I):", np.bincount(labels))

import numpy as np
import librosa
import pandas as pd
import os

train_df = pd.read_csv("/content/train.csv")
base_path = "/content/train"

n_mels = 128
fixed_length = 200
mel_features = []
labels = []



for i, row in train_df.iterrows():
    file_id = row["ID"].replace("_I_", "_").replace("_E_", "_")
    path = os.path.join(base_path, file_id + ".wav")

    try:
        y, sr = librosa.load(path, sr=16000)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # 고정 길이 처리
        if mel_db.shape[1] < fixed_length:
            pad = fixed_length - mel_db.shape[1]
            left = pad // 2
            right = pad - left
            mel_db = np.pad(mel_db, ((0, 0), (left, right)), mode='constant')
        else:
            center = mel_db.shape[1] // 2
            mel_db = mel_db[:, center - fixed_length // 2:center + fixed_length // 2]

        mel_features.append(mel_db)
        labels.append(label_map[row["Target"]])

    except Exception as e:
        print("❌", file_id, e)

# 저장
mel_array = np.stack(mel_features)[..., np.newaxis]
np.save("/content/mel_array.npy", mel_array)
np.save("/content/mel_labels.npy", np.array(labels))

import numpy as np
import librosa
import pandas as pd
import os
import zipfile

# test.csv 불러오기
test_df = pd.read_csv("/content/test.csv")
test_base_path = "/content/test"

n_mels = 128
fixed_length = 200
test_mel_features = []
test_ids = []

for i, row in test_df.iterrows():
    file_name = row["ID"]
    path = os.path.join(test_base_path, file_name)

    try:
        y, sr = librosa.load(path, sr=16000)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # 고정 길이 처리
        if mel_db.shape[1] < fixed_length:
            pad = fixed_length - mel_db.shape[1]
            left = pad // 2
            right = pad - left
            mel_db = np.pad(mel_db, ((0, 0), (left, right)), mode='constant')
        else:
            center = mel_db.shape[1] // 2
            mel_db = mel_db[:, center - fixed_length // 2:center + fixed_length // 2]

        test_mel_features.append(mel_db)
        test_ids.append(file_name.replace(".wav", ""))  # ID용으로 저장

    except Exception as e:
        print(f"❌ {file_name}: {e}")

# 저장
test_mel_array = np.stack(test_mel_features)[..., np.newaxis]
np.save("/content/test_mel_array.npy", test_mel_array)
np.save("/content/test_ids.npy", np.array(test_ids))

print("✅ test_mel_array.npy, test_ids.npy 저장 완료")

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# 전처리된 Mel-spectrogram과 라벨 불러오기
mel_array  = np.load("/content/mel_array.npy")      # shape: (N, 128, 200, 1)
mel_labels = np.load("/content/mel_labels.npy")     # shape: (N,)

# 정규화
mel_array = (mel_array - mel_array.mean()) / (mel_array.std() + 1e-6)

# 학습/검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(
    mel_array, mel_labels,
    test_size=0.2, random_state=42, stratify=mel_labels
)

# 클래스 가중치 계산 (불균형 대응용)
cw = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {0: cw[0], 1: cw[1]}
