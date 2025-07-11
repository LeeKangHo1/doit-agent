# agents/supervisor.py
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage

from . import llm, agent_log, root_path
from models import State, Task
from file_utils import get_outline



# 일을 분배하는 조장 에이전트
def supervisor(state: State):
    agent_log("supervisor")

    supervisor_system_prompt = PromptTemplate.from_template(
        """
        너는 AI 팀의 supervisor로서 AI 팀의 작업을 관리하고 지도한다.
        사용자가 원하는 책을 써야 한다는 최종 목표를 염두에 두고,
        사용자의 요구를 달성하기 위해 현재 해야 할 일이 무엇인지 결정한다.

        supervisor가 활용할 수 있는 agent는 다음과 같다.
        - content_strategist: 사용자의 요구 사항이 명확해졌을 때 사용한다. 
        AI 팀의 콘텐츠 전략을 결정하고, 전체 책의 목차(outline)를 작성한다.
        - communicator: AI 팀에서 해야 할 일을 스스로 판단할 수 없을 때 사용한다. 
        사용자에게 진행상황을 보고하고, 다음 지시를 물어본다.
        - web_search_agent: 웹 검색을 통해 목차(outline) 작성에 필요한 정보를 확보한다.
        - vector_search_agent: 벡터 DB 검색을 통해 목차(outline) 작성에 필요한 정보를 확보한다.

        아래 내용을 고려하여, 현재 해야할 일이 무엇인지, 사용할 수 있는 agent를 단답으로 말하라.

        ----------------------------------
        previous_outline: {outline}
        ----------------------------------
        messages: {messages}

        """
    )

    supervisor_chain = supervisor_system_prompt | llm.with_structured_output(Task)

    messages = state.get("messages", [])

    inputs = {
        "messages": messages,
        "outline": get_outline(root_path)
    }

    task = supervisor_chain.invoke(inputs)
    task_history = state.get("task_history", [])
    task_history.append(task)

    supervisor_message = AIMessage(f"[Supervisor] {task}")
    messages.append(supervisor_message)
    print(supervisor_message.content)

    return {
        "messages": messages,
        "task_history": task_history,
    }

def supervisor_router(state: State):
    task = state['task_history'][-1]
    return task.agent