import os
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

os.environ["PATH"] += os.pathsep + r"C:\workspace\tools\ffmpeg-2025-06-26-git-09cd38e9d5-full_build\bin"

# 받아쓰기 함수
def whisper_stt(
    audio_file_path: str,      
    output_file_path: str = "./output.csv"
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",  # 작업 유형: 자동 음성 인식
        model=model,  # 로드한 모델
        tokenizer=processor.tokenizer,  # 토크나이저
        feature_extractor=processor.feature_extractor,  # 특성 추출기
        torch_dtype=torch_dtype,  # 텐서 데이터 타입
        device=device,  # 실행 디바이스
        return_timestamps=True,   # ✅ 청크별 타임스탬프 반환 옵션
        chunk_length_s=10,        # ✅ 오디오를 10초 단위로 나누어 처리
        stride_length_s=2         # ✅ 청크 사이에 2초씩 겹치게 함 (음성 손실 방지)
    )

    result = pipe(audio_file_path)
    df = whisper_to_dataframe(result, output_file_path) 

    return result, df

# pandas 데이터 프레임 형태로 저장하는 함수
def whisper_to_dataframe(result, output_file_path):
    start_end_text = []

    for chunk in result["chunks"]:
        start = chunk["timestamp"][0]
        end = chunk["timestamp"][1]
        text = chunk["text"].strip()
        start_end_text.append([start, end, text])
        df = pd.DataFrame(start_end_text, columns=["start", "end", "text"])
        df.to_csv(output_file_path, index=False, sep="|")

    return df

if __name__ == "__main__":
    # print("현재 실행 위치:", os.getcwd())
    base_dir = Path(__file__).resolve().parent.parent  # chap05 기준
    audio_path = base_dir / "audio" / "싼기타_비싼기타.mp3"
    output_path = base_dir / "audio" / "싼기타_비싼기타.csv" 

    result, df = whisper_stt(str(audio_path), str(output_path))

    print(df)
        