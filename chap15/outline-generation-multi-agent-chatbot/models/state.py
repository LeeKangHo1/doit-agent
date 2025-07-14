# models/state.py

from typing_extensions import TypedDict
from typing import List
from langchain_core.messages import AnyMessage, SystemMessage
from datetime import datetime

from .task import Task

class State(TypedDict):
    messages: List[AnyMessage | str]
    task_history: List[Task]
    references: dict # vector search agent에서 검색한 정보를 저장하는 변수
    user_request: str # 사용자의 요구 사항을 저장하는 변수
    ai_recommendation: str # AI의 추천을 저장하는 변수
    supervisor_call_count: int # supervisor 호출 횟수를 저장하는 변수

# 상태 초기화 함수
def state_init():
    state = State(
        messages = [
            # 여러 에이전트가 추가될 것을 고려하여 설계
            SystemMessage(
                f"""
                너희 AI들은 사용자의 요구에 맞는 책을 쓰는 작가 팀이다.
                사용자가 사용하는 언어로 대화하라. 

                현재 시각은 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}이다.
                """
            )
        ],
        task_history = [],
        references={"queries": [], "docs": []},
        user_request="",
    )
    return state