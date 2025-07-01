import os
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pyannote.audio import Pipeline
from dotenv import load_dotenv

import sys
from pathlib import Path 
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from utils.logger import get_logger
logger = get_logger(__name__)

# 환경 변수 불러오기 (HuggingFace 토큰)
load_dotenv()
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

# ffmpeg 실행 경로 설정 (Whisper 오디오 처리용)
os.environ["PATH"] += os.pathsep + \
    r"C:\workspace\tools\ffmpeg-2025-06-26-git-09cd38e9d5-full_build\bin"

# =====================
# Whisper 모델을 이용한 STT 처리 함수
# =====================
def whisper_stt(
    audio_file_path: str,      # 입력 오디오 경로
    output_file_path: str = "./output.csv"  # 출력 CSV 경로
):
    # 디바이스 및 텐서 타입 설정
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3-turbo"

    # Whisper 모델 로드
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True
    )
    model.to(device)

    # 프로세서 로드 (토크나이저 + 특성 추출기)
    processor = AutoProcessor.from_pretrained(model_id)

    # ASR 파이프라인 구성
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=True,  # 타임스탬프 반환 (청크 단위)
        chunk_length_s=10,       # 오디오를 10초씩 청크로 분할
        stride_length_s=2,       # 청크 간 2초 겹치기
    )

    # STT 처리 실행
    result = pipe(str(audio_file_path))

    # 결과를 DataFrame으로 저장
    df = whisper_to_dataframe(result, output_file_path)

    return result, df

# =====================
# Whisper 결과를 CSV로 저장하는 함수
# =====================
def whisper_to_dataframe(result, output_file_path):
    start_end_text = []

    for chunk in result["chunks"]:
        start = chunk["timestamp"][0]  # 시작 시간
        end = chunk["timestamp"][1]    # 끝 시간
        text = chunk["text"].strip()   # 텍스트
        start_end_text.append([start, end, text])

    df = pd.DataFrame(start_end_text, columns=["start", "end", "text"])
    df.to_csv(output_file_path, index=False, sep="|")

    return df

# =====================
# 화자 분리 함수 (Pyannote 기반)
# =====================
def speaker_diarization(
    audio_file_path: str,
    output_rttm_file_path: str,
    output_csv_file_path: str,
):
    # Pyannote 파이프라인 로드
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HUGGING_FACE_TOKEN
    )

    # CUDA 설정
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        print('cuda is available')
    else:
        print('cuda is not available')

    # 화자 분리 실행
    diarization_pipeline = pipeline(audio_file_path)

    # RTTM 파일로 저장
    with open(output_rttm_file_path, "w", encoding='utf-8') as rttm:
        diarization_pipeline.write_rttm(rttm)

    # RTTM 파일을 CSV로 읽기
    df_rttm = pd.read_csv(
        output_rttm_file_path,
        sep=' ',
        header=None,
        names=['type', 'file', 'chnl', 'start', 'duration', 'C1', 'C2', 'speaker_id', 'C3', 'C4']
    )

    # 끝나는 시간 계산
    df_rttm['end'] = df_rttm['start'] + df_rttm['duration']

    # 연속 발화 묶기 위한 number 열 생성
    df_rttm["number"] = None
    df_rttm.at[0, "number"] = 0

    for i in range(1, len(df_rttm)):
        if df_rttm.at[i, "speaker_id"] != df_rttm.at[i-1, "speaker_id"]:
            df_rttm.at[i, "number"] = df_rttm.at[i-1, "number"] + 1
        else:
            df_rttm.at[i, "number"] = df_rttm.at[i-1, "number"]

    # 발화 묶음 단위로 그룹화
    df_rttm_grouped = df_rttm.groupby("number").agg(
        start=pd.NamedAgg(column='start', aggfunc='min'),
        end=pd.NamedAgg(column='end', aggfunc="max"),
        speaker_id=pd.NamedAgg(column="speaker_id", aggfunc="first"),
    )

    # duration 계산
    df_rttm_grouped["duration"] = df_rttm_grouped["end"] - df_rttm_grouped["start"]

    # CSV 저장
    df_rttm_grouped.to_csv(
        output_csv_file_path,
        sep= ',',
        index=False,
        encoding="utf-8",
    )

    return df_rttm_grouped

# =====================
# STT + 화자분리 결과를 합쳐서 최종 CSV 만드는 함수
# =====================
def stt_to_rttm(
        audio_file_path: str,
        stt_output_file_path: str,
        rttm_file_path: str,
        rttm_csv_file_path: str,
        final_output_csv_file_path: str
    ):

    # STT 처리
    result, df_stt = whisper_stt(audio_file_path, stt_output_file_path)

    # 화자 분리 처리
    df_rttm = speaker_diarization(audio_file_path, rttm_file_path, rttm_csv_file_path)

    # 텍스트 열 초기화
    df_rttm["text"] = ""

    # 각 STT 텍스트를 화자 타임라인과 매칭
    for i_stt, row_stt in df_stt.iterrows():
        overlap_dict = {}

        for i_rttm, row_rttm in df_rttm.iterrows():
            # 두 구간의 겹치는 시간 계산
            overlap = max(0, min(row_stt["end"], row_rttm["end"]) - max(row_stt["start"], row_rttm["start"]))
            overlap_dict[i_rttm] = overlap

        # 가장 겹치는 구간 찾기
        max_overlap = max(overlap_dict.values())
        max_overlap_idx = max(overlap_dict, key=overlap_dict.get)

        # 겹치는 구간에 해당 텍스트를 추가
        if max_overlap > 0:
            df_rttm.at[max_overlap_idx, "text"] += row_stt["text"] + "\n"

    # 최종 결과 저장
    df_rttm.to_csv(
        final_output_csv_file_path,
        index=False,
        sep='|',
        encoding='utf-8'
    )
    return df_rttm

# =====================
# 실행 진입점
# =====================
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent

    audio_file_path = base_dir / "audio" / "싼기타_비싼기타.mp3"
    stt_output_file_path = base_dir / "audio" / "싼기타_비싼기타.csv"
    rttm_file_path = base_dir / "audio" / "싼기타_비싼기타.rttm"
    rttm_csv_file_path = base_dir / "audio" / "싼기타_비싼기타_rttm.csv"
    final_csv_file_path = base_dir / "audio" / "싼기타_비싼기타_final.csv"

    df_rttm = stt_to_rttm(
        audio_file_path,
        stt_output_file_path,
        rttm_file_path,
        rttm_csv_file_path,
        final_csv_file_path,
    )

    print(df_rttm)