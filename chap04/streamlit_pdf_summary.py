import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
import pymupdf

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

# PDF → 텍스트 변환
def pdf_to_text(pdf_file_path):
    doc = pymupdf.open(pdf_file_path)
    header_height = 80
    footer_height = 80
    full_text = ''

    for page in doc:
        rect = page.rect
        text = page.get_text(clip=(0, header_height, rect.width, rect.height - footer_height))
        full_text += text + '\n------------------------------------\n'

    return full_text

# 요약 생성
def summarize_txt(txt: str):
    system_prompt = f'''
    너는 다음 글을 요약하는 봇이다. 아래 글을 읽고, 저자의 문제 인식과 주장을 파악하고, 주요 내용을 요약하라. 

    작성해야 하는 포맷은 다음과 같다. 
    
    # 제목

    ## 저자의 문제 인식 및 주장 (15문장 이내)
    
    ## 저자 소개

    
    =============== 이하 텍스트 ===============

    {txt}
    '''

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt}
        ]
    )

    return response.choices[0].message.content

# ✅ Streamlit 앱
st.set_page_config(page_title="PDF 요약기", layout="wide")
st.title("📄 PDF 문서 요약기 (OpenAI + Streamlit)")

uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])

if uploaded_file is not None:
    # 저장할 임시 경로
    temp_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("📖 문서 분석 중..."):
        extracted_text = pdf_to_text(temp_path)
        summary = summarize_txt(extracted_text)

    st.success("✅ 요약 완료!")
    st.subheader("📌 요약 결과:")
    st.text_area("요약 내용", summary, height=500)

    os.remove(temp_path)  # 임시 파일 삭제
