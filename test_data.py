import os
import pandas as pd
import librosa
import zipfile

# test 압축 해제
test_zip_path = "/content/test.zip"
test_extract_path = "/content"

with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
    zip_ref.extractall(test_extract_path)

# test CSV 불러오기
test_csv_path = "/content/test.csv"
test_df = pd.read_csv(test_csv_path)

# test 오디오 폴더 경로
test_base_path = "/content/test"

# 오디오 및 샘플링레이트 저장 리스트
test_audio_data_list = []
test_sample_rate_list = []

for idx, row in test_df.iterrows():
    file_id = row["ID"]
    file_path = os.path.join(test_base_path, file_id)

    if idx < 5:
        print(f"🔍 시도: {file_path} → 존재 여부: {os.path.exists(file_path)}")

    if not os.path.exists(file_path):
        print(f"❌ 파일 없음: {file_path}")
        continue

    try:
        y, sr = librosa.load(file_path, sr=None)
        test_audio_data_list.append(y)
        test_sample_rate_list.append(sr)
    except Exception as e:
        print(f"⚠️ 로딩 실패: {file_path}, 이유: {e}")

print(f"\n✅ 성공: {len(test_audio_data_list)}개, 🔥 실패: {len(test_df) - len(test_audio_data_list)}개")
