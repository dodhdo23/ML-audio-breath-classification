
import os
import pandas as pd
import librosa
import zipfile

# 압축 해제
zip_path = "/content/train.zip"
extract_path = "/content"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# CSV 불러오기
csv_path = "/content/train.csv"
train_df = pd.read_csv(csv_path)

# 오디오 폴더 경로 수정
base_path = "/content/train"

audio_data_list = []
sample_rate_list = []

for idx, row in train_df.iterrows():
    original_id = row["ID"]
    file_id = original_id.replace("_I_", "_").replace("_E_", "_")
    file_path = os.path.join(base_path, file_id + ".wav")

    if idx < 5:
        print(f"🔍 시도: {file_path} → 존재 여부: {os.path.exists(file_path)}")

    if not os.path.exists(file_path):
        print(f"❌ 파일 없음: {file_path}")
        continue

    try:
        y, sr = librosa.load(file_path, sr=None)
        audio_data_list.append(y)
        sample_rate_list.append(sr)
    except Exception as e:
        print(f"⚠️ 로딩 실패: {file_path}, 이유: {e}")

print(f"\n✅ 성공: {len(audio_data_list)}개, 🔥 실패: {len(train_df) - len(audio_data_list)}개")
