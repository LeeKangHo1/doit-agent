import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

# .env 파일에서 환경변수 로드 (예: OPENAI_API_KEY)
load_dotenv()

# ----- 사이드바 구성 -----
with st.sidebar:
    # .env에서 OpenAI API 키 가져오기
    openai_api_key = os.getenv('OPENAI_API_KEY')

    # 사이드바에 유용한 링크 제공
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

# 페이지 제목 출력
st.title("\ud83d\udcac Chatbot")

# ----- 대화 기록 초기화 -----
if "messages" not in st.session_state:
    # session_state에 대화 기록(messages)이 없으면 초기화
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

# ----- 이전 대화 기록 출력 -----
for msg in st.session_state.messages:
    # 사용자/AI의 역할(role)에 따라 채팅 형식으로 출력
    st.chat_message(msg["role"]).write(msg["content"])

# ----- 사용자 입력 받기 -----
if prompt := st.chat_input():
    # API 키가 없을 경우 메시지 출력 후 중단
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # OpenAI 클라이언트 생성
    client = OpenAI(api_key=openai_api_key)

    # 사용자 입력을 대화 기록에 추가하고 출력
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 전체 대화 기록을 바탕으로 GPT 응답 생성
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=st.session_state.messages
    )

    # 응답 텍스트 추출 및 대화 기록에 추가 후 출력
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
