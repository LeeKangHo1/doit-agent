# agents/communicator.py
from langchain_core.prompts import PromptTemplate

from . import llm, agent_log, root_path
from models import State
from file_utils import get_outline
from datetime import datetime

# communicator : 사용자와 대화할 노드
def communicator(state: State):
    agent_log("communicator")

    communicator_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI 팀의 communicator로서,
        AI 팀의 진행 상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위해 대화를 나눈다.

        사용자도 outline(목차)을 이미 보고 있으므로, 다시 출력할 필요는 없다.

        outline: {outline}
        ------------------------------
        messages: {messages}
        """
    )

    system_chain = communicator_system_prompt | llm

    # 상태에서 메시지 가져오기
    messages = state['messages']

    # 입력값 정의
    inputs = {
        "messages": messages,
        "outline": get_outline(root_path)
    }

    gathered = None

    print('\nAI\t: ', end="")
    for chunk in system_chain.stream(inputs):
        print(chunk.content, end="")

        if gathered is None:
            gathered = chunk
        else:
            gathered += chunk

    messages.append(gathered)

    task_history = state.get("task_history", [])
    if task_history[-1].agent != "communicator":
        raise ValueError(f"Communicator 아닌 agent가 목차 작성을 시도하고 있습니다.\n {task_history[-1]}")

    task_history[-1].done = True
    task_history[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    return {
        "messages": messages,
        "task_history": task_history,
    }

