# agents/__init__.py

import os
from langchain_openai import ChatOpenAI

# 프로젝트 루트 경로 반환
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 공통 LLM 모델
llm = ChatOpenAI(model="gpt-4o-mini")

# 공통 로깅 함수
def agent_log(agent_name, message=""):
    """에이전트 로깅 함수"""
    print(f"\n\n========== {agent_name.upper()} ==========")
    if message:
        print(message)

# 외부에서 함수만 가져가기 위한 조치
from .communicator import communicator
from .content_strategist import content_strategist
from .supervisor import supervisor
from .vector_search_agent import vector_search_agent
from .web_search_agent import web_search_agent
from .business_analysist import business_analysist
from .outline_reviewer import outline_reviewer

agent_nodes = {
    "communicator": communicator,
    "content_strategist": content_strategist,
    "supervisor": supervisor,
    "vector_search_agent": vector_search_agent,
    "web_search_agent": web_search_agent,
    "business_analysist": business_analysist,
    "outline_reviewer": outline_reviewer,
}
