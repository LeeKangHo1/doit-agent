# agents/content_strategist.py

from . import llm, agent_log, root_path
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage
from datetime import datetime

from models import State, Task
from file_utils import save_outline, get_outline

# content_strategist: 목차를 작성하는 에이전트
def content_strategist(state: State):
    agent_log("content_strategist")

    content_strategist_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI 팀의 콘텐츠 전략가(content_strategist)로서,
        이전 대화 내용을 바탕으로 사용자의 요구 사항을 분석하고, AI팀이 쓸 책의 세부 목차를 결정한다.

        지난 목차가 있다면 그 버전을 사용자의 요구에 맞게 수정하고, 없다면 새로운 목차를 제안한다.
        목차를 작성하는 데 필요한 정보는 "참고 자료"에 있으므로 활용한다.

        --------------------------
        - 지난 목차: {outline}
        --------------------------
        - 이전 대화 내용: {messages}
        --------------------------
        - 참고 자료: {references}
        """
    )

    content_strategist_chain = content_strategist_system_prompt | llm | StrOutputParser()

    messages = state["messages"]
    outline = get_outline(root_path)

    inputs = {
        "messages": messages,
        "outline": outline,
        "references": state.get("references", {"queries": [], "docs": []}),
    }

    gathered = ''
    for chunk in content_strategist_chain.stream(inputs):
        gathered += chunk
        print(chunk, end="")

    print()

    save_outline(root_path, gathered)

    content_strategist_message = f"[Content Strategist] 목차 작성 완료"
    print(content_strategist_message)
    messages.append(AIMessage(content_strategist_message))

    task_history = state.get("task_history", [])
    if task_history[-1].agent != "content_strategist":
        raise ValueError(f"Content Strategist가 아닌 agent가 목차 작성을 시도하고 있습니다.\n {task_history[-1]}")
    
    task_history[-1].done = True
    task_history[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    new_task = Task(
        agent="communicator",
        done=False,
        description="AI 팀의 진행 상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위해 대화를 나눈다.",
        done_at=""
    )
    task_history.append(new_task)

    print(f"===new_task: {new_task}===")

    return {
        "messages": messages,
        "task_history": task_history,
    }

