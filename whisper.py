from faster_whisper import WhisperModel, BatchedInferencePipeline
import os

# dates = ["녹음", "녹음 (2)"]
dates = ["녹음", "녹음 (2)", "녹음 (3)"]

# 모델 로드
model = WhisperModel("medium", device="cuda", compute_type="float16")
batched_model = BatchedInferencePipeline(model=model)


def transcribe_audio(audio_path, output_path, name):
   # 음성 파일 텍스트 변환
   segments, info = batched_model.transcribe(
    audio_path,
    batch_size=8,  # 여기서 배치 사이즈 설정
    language="ko"  ,
    initial_prompt="This is a computer science lecture with Korean and English mixed"
    )
   segments, info = model.transcribe(
       audio_path,
       language="ko",  # 한국어 강의면 ko로 설정
       vad_filter=True,  # 묵음 구간 제거
       word_timestamps=True  # 단어별 타임스탬프 불필요시 False
   )
   
   # 결과 텍스트 저장
   with open(output_path, "w", encoding="utf-8") as f:
       for segment in segments:
            f.write(segment.text + " ")

# 실행 예시

for date in dates:
    print(f"Processing: {date}")
    
    input_path = f"{date}.m4a"  # 원본 음성 파일 경로
    output_path = f"{date}.txt"  # 저장할 텍스트 파일 경로
    transcribe_audio(input_path, output_path, date)

    print(f"Completed: {date}")  # 처리 완료된 파일 출력