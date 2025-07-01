import os
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pyannote.audio import Pipeline
from dotenv import load_dotenv

load_dotenv()

HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

os.environ["PATH"] += os.pathsep + \
    r"C:\workspace\tools\ffmpeg-2025-06-26-git-09cd38e9d5-full_build\bin"

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
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=True,  # 청크별로 타임스탬프를 반환
        chunk_length_s=10,  # 입력 오디오를 10초씩 나누기
        stride_length_s=2,  # 2초씩 겹치도록 청크 나누기
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


def speaker_diarization(
    audio_file_path: str,
    output_rttm_file_path: str,
    output_csv_file_path: str,
):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HUGGING_FACE_TOKEN
    )

    # cuda가 사용 가능한 경우 cuda를 사용하도록 설정
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        print('cuda is available')
    else:
        print('cuda is not available')

    diarization_pipeline = pipeline(audio_file_path)

    with open(output_rttm_file_path, "w", encoding='utf-8') as rttm:
        diarization_pipeline.write_rttm(rttm)

    df_rttm = pd.read_csv(
        output_rttm_file_path,
        sep=' ',
        header=None,
        names=['type', 'file', 'chnl', 'start', 'duration', 'C1', 'C2', 'speaker_id', 'C3', 'C4']
    )

    # 발언이 끝난 시간 추가
    df_rttm['end'] = df_rttm['start'] + df_rttm['duration']

    # 연속된 발화를 기록하기 위해 number 변수 추가
    df_rttm["number"] = None	# number 열 만들고 None으로 초기화
    df_rttm.at[0, "number"] = 0

    for i in range(1, len(df_rttm)):
        if df_rttm.at[i, "speaker_id"] != df_rttm.at[i-1, "speaker_id"]:
            df_rttm.at[i, "number"] = df_rttm.at[i-1, "number"] + 1
        else:
            df_rttm.at[i, "number"] = df_rttm.at[i-1, "number"]

    df_rttm_grouped = df_rttm.groupby("number").agg(
        start=pd.NamedAgg(column='start', aggfunc='min'),
        end=pd.NamedAgg(column='end', aggfunc="max"),
        speaker_id=pd.NamedAgg(column="speaker_id", aggfunc="first"),
    )

    df_rttm_grouped["duration"] = df_rttm_grouped["end"] - df_rttm_grouped["start"]
    
    df_rttm_grouped.to_csv(
        output_csv_file_path,
        sep= ',',
        index=False,
        encoding="utf-8",
    )

    return df_rttm_grouped


if __name__ == "__main__":
    # print("현재 실행 위치:", os.getcwd())
    base_dir = Path(__file__).resolve().parent.parent  # chap05 기준

    audio_file_path = base_dir / "audio" / "싼기타_비싼기타.mp3" # 원본 오디오 파일
    stt_output_file_path = base_dir / "audio" / "싼기타_비싼기타.csv" # STT 결과 파일
    rttm_file_path = base_dir / "audio" / "싼기타_비싼기타.rttm" # 화자 분리 원본 파일
    rttm_csv_file_path = base_dir / "audio" / "싼기타_비싼기타_rttm.csv" # 화자 분리 csv 파일

    # result, df = whisper_stt(str(audio_file_path), str(stt_output_file_path))

    df_rttm = speaker_diarization(
        audio_file_path,
        rttm_file_path,
        rttm_csv_file_path        
    )

    # print(df_rttm)
