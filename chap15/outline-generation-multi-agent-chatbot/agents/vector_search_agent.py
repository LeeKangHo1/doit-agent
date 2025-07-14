# agents/vector_search_agent.py
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage

from tools.vector_search import retrieve

from . import llm, agent_log, root_path
from models import State, Task
from file_utils import get_outline
from datetime import datetime

# 벡터 검색을 수행하는 에이전트
def vector_search_agent(state: State):
    agent_log("vector_search_agent")

    task_history = state.get("task_history", [])
    task = task_history[-1]

    if task.agent != "vector_search_agent":
        raise ValueError(f"Vector Search Agent가 아닌 agent가 목차 작성을 시도하고 있습니다.\n {task}")
    
    vector_search_system_prompt = PromptTemplate.from_template(
        """
        너는 다른 AI Agent 들이 수행한 작업을 바탕으로,
        목차(outline) 작성에 필요한 정보를 벡터 검색을 통해 찾아내는 Agent이다.

        현재 목차(outline)를 작성하는 데 필요한 정보를 확보하기 위해,
        다음 내용을 활용해 적절한 벡터 검색을 수행하라.

        - 검색 목적: {mission}
        --------------------------
        - 과거 검색 내용: {references}
        --------------------------
        - 이전 대화 내용: {messages}
        --------------------------
        - 목차(outline): {outline}
        """
    )

    mission = task.description
    references = state.get("references", {"queries": [], "docs": []})
    messages = state["messages"]
    outline = get_outline(root_path)

    inputs = {
        "mission": mission,
        "references": references,
        "messages": messages,
        "outline": outline,
    }

    llm_with_retriever = llm.bind_tools([retrieve])
    vector_search_chain = vector_search_system_prompt | llm_with_retriever

    # LLM은 직접 도구 실행을 하지 못하기 때문에 체인 실행 시 계획만 만들어 줍니다.
    search_plans = vector_search_chain.invoke(inputs)

    # 도구 호출 계획에서 도구 호출은 여러 번 있을 수 있습니다.
    for tool_call in search_plans.tool_calls:
        print('-----------------------------------', tool_call)

        args = tool_call["args"]

        query = args["query"]
        retrieved_docs = retrieve(args)

        # references에 검색 쿼리를 저장
        references["queries"].append(query)
        # references에 검색 결과(문서)를 저장
        references["docs"] += retrieved_docs 

    unique_docs = []
    # 벡터 DB에 저장된 문서가 적거나 한 문서가 여러 질문에 대한 답을 모두 포함하고 있을 때, 같은 문서가 반복해서 나올 수 있다. -> set으로 중복 처리
    unique_page_contents = set()

    for doc in references["docs"]:
        if doc.page_content not in unique_page_contents:
            unique_docs.append(doc)
            unique_page_contents.add(doc.page_content)
    references["docs"]  = unique_docs

    # 쿼리와 청크 문서를 개발자가 확인할 수 있도록 터미널에 출력
    print('Queries:-------------------')
    queries = references["queries"]
    for query in queries:
        print(query)

    print('References:----------------')
    for doc in references["docs"]:
        print(doc.page_content[:100])
        print('-----------------------')

    task_history[-1].done = True
    task_history[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 벡터 검색 에이전트의 작업 후기를 생성해 대화 기록에 추가
    msg_str = f"[VECTOR SEARCH AGENT] 다음 질문에 대한 검색 완료: {queries}"
    message = AIMessage(msg_str)
    print(msg_str)

    messages.append(message)
    # business_analysist로 전달되는 ai_recommendation 정의
    ai_recommendation = "현재 참고자료(references)가 목차(outoine)를 개선하는 데 충분한지 확인하라. 충분하다면 content_strategist로 목차 작성을 하라. 그렇지 않다면 supervisor에게 web_search_agent를 호출하라고 하라"

    # 작업 결과를 상태에 반영
    return {
        "messages": messages,
        "task_history": task_history,
        "references": references,
        "ai_recommendation": ai_recommendation,
    }