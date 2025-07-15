# agents/web_search_agent.py
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage

from tools.vector_search import add_web_pages_json_to_chroma
from tools.web_search import web_search

from . import llm, agent_log, root_path
from models import State, Task
from file_utils import get_outline
from datetime import datetime

# 웹 검색 에이전트 노드
def web_search_agent(state: State):
    agent_log("web_search_agent")

    task_history = state.get("task_history", [])
    task = task_history[-1]

    if task.agent != "web_search_agent":
        raise ValueError(f"Web Search Agent가 아닌 agent가 Web Search Agent를 시도하고 있습니다.\n {task}")
    
    web_search_system_prompt = PromptTemplate.from_template(
        """
        너는 다른 AI Agent 들이 수행한 작업을 바탕으로,
        목차(outline) 작성에 필요한 정보를 웹 검색을 통해 찾아내는 Web Search Agent이다.

        현재 부족한 정보를 검색하고, 복합적인 질문은 나눠서 검색하라.

        - 검색 목적: {mission}
        --------------------------------------------
        - 과거 검색 내용: {references}
        --------------------------------------------
        - 이전 대화 내용: {messages}
        --------------------------------------------
        - 목차(outline): {outline}
        --------------------------------------------
        - 현재 시각 : {current_time}
        """
    )
    # 언어 모델이 최신 자료를 검색할 때 검색어에 해당 언어 모델이 생성된 연도를 입력하는 문제가 있음
    # 현재 시각을 입력해주면 현재 시각을 활용해 최신 정보를 제대로 검색 가능

    messages = state.get("messages", [])

    inputs = {
        "mission": task.description,
        "references": state.get("references", {"queries": [], "docs": []}),
        "messages": messages,
        "outline": get_outline(root_path),
        "current_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    llm_with_web_search = llm.bind_tools([web_search])

    web_search_chain = web_search_system_prompt | llm_with_web_search

    search_plans = web_search_chain.invoke(inputs)

    queries = []

    for tool_call in search_plans.tool_calls:
        print('------- web search -------', tool_call)
        args = tool_call["args"]

        queries.append(args["query"])

        # 검색 결과 JSON 파일 경로 가져오기
        # web_search는 results, resource_json_path 2가지를 반환하는데 앞은 빼고 뒤만 가져온다.
        _, json_path = web_search.invoke(args)
        print('json_path:', json_path)

        # JSON 파일을 크로마DB에 추가
        add_web_pages_json_to_chroma(json_path)

    task_history[-1].done = True
    task_history[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    task_desc = "AI팀이 쓸 책의 세부 목차를 결정하기 위한 정보를 벡터 검색을 통해 찾아낸다."
    task_desc += f" 다음 항목이 새로 추가되었다.\n: {queries}"

    new_task = Task(
        agent="vector_search_agent",
        done=False,
        description=task_desc,
        done_at=""
    )

    task_history.append(new_task)

    msg_str = f"[WEB SEARCH AGENT] 다음 질문에 대한 검색 완료: {queries}"
    messages.append(AIMessage(msg_str))

    return {
        "messages": messages,
        "task_history": task_history,
    }
