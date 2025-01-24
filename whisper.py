from faster_whisper import WhisperModel, BatchedInferencePipeline
import os

date = "20250124-2"

# 모델 로드
model = WhisperModel("medium", device="cuda", compute_type="float16")
batched_model = BatchedInferencePipeline(model=model)
segments, info = batched_model.transcribe(
   f"{date}.m4a",
   batch_size=8,  # 여기서 배치 사이즈 설정
   language="ko"  ,
   initial_prompt="This is a computer science lecture with Korean and English mixed"
)

def format_timestamp(seconds):
    hours = int(seconds // 3600)          # 시간 계산
    minutes = int((seconds % 3600) // 60) # 분 계산
    secs = int(seconds % 60)              # 초 계산
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"  # 02d는 2자리 숫자로 표시 (예: 01, 02, ...)

def transcribe_audio(audio_path, output_path):
   # 음성 파일 텍스트 변환
   segments, info = model.transcribe(
       audio_path,
       language="ko",  # 한국어 강의면 ko로 설정
       vad_filter=True,  # 묵음 구간 제거
       word_timestamps=True  # 단어별 타임스탬프 불필요시 False
   )
   
   # 결과 텍스트 저장
   with open(output_path, "w", encoding="utf-8") as f:
       for segment in segments:
            timestamp = f"[{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}] "
            f.write(timestamp + segment.text + "\n")

# 실행 예시

input_path = f"{date}.m4a"  # 원본 음성 파일 경로
output_path = f"{date}.txt"  # 저장할 텍스트 파일 경로
transcribe_audio(input_path, output_path)