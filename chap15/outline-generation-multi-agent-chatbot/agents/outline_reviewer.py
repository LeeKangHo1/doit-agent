# agents/outline_reviewer.py

from . import llm, agent_log, root_path
from langchain_core.prompts import PromptTemplate

from models import State
from file_utils import get_outline

# 목차를 검토하는 에이전트
def outline_reviewer(state: State):
    agent_log("outline_reviewer")

    outline_reviewer_system_prompt = PromptTemplate.from_template(
        """
        너는 AI팀의 목차 리뷰어로서, AI팀이 작성한 목차(outline)를 검토하고 문제점을 지적한다.

        - outline이 사용자의 요구사항을 충족시키는지 여부
        - outline의 논리적인 흐름이 적절한지 여부
        - 근거에 기반하지 않는 내용이 있는지 여부
        - 주어진 참고자료(references)를 충분히 활용했는지 여부
        - example.com 같은 더미 URL이 있는지 여부:
        - 실제 페이지 URL이 아닌 대표 URL로 되어 있는 경우 삭제 해야 함: 어떤 URL이 삭제되어야 하는지 명시하라. 명시한 URL과 그 자료는 우선적으로 폐기하라.
        - 기타 리뷰 사항

        그 분석 결과를 설명하고, 다음에 어떤 작업을 하면 좋을지 제안하라.

        - 분석 결과: outline이 사용자의 요구사항을 충족시키는지 여부
        - 제안 사항: (vector_search_agent, communicator 중 어떤 agent를 호출할지)

        --------------------------
        - user_request: {user_request}
        --------------------------
        - references: {references}
        --------------------------
        - outline: {outline}
        --------------------------
        - messages: {messages} 
        --------------------------
        """
    )

    user_request = state.get("user_request", None)
    outline = get_outline(root_path)
    references = state.get("references", {"queries": [], "docs": []})
    messages = state.get("messages",  [])

    inputs = {
        "user_request": user_request,
        "outline": outline,
        "references": references,
        "messages": messages,
    }

    # 시스템 프롬프트와 모델을 연결
    outline_reviewer_chain = outline_reviewer_system_prompt | llm

    review = outline_reviewer_chain.stream(inputs)

    gathered = None

    # 목차 리뷰가 길어질 수 있으므로 출력 방식을 stream으로
    for chunk in review:
        print(chunk.content, end="")

        if gathered is None:
            gathered = chunk
        else:
            gathered += chunk

    if '[OUTLINE REVIEW AGENT]' not in gathered.content:
        gathered.content = f"[OUTLINE REVIEW AGENT] {gathered.content}"

    print(gathered.content)
    # AIMessageChunk 여러 개를 += 로 더하면 → AIMessage로 자동 합쳐짐 -> gathered는 AIMessage 객체가 됨
    # AIMessage(gathered.content)로 쓰지 않아도 된다.
    messages.append(gathered)

    ai_recommendation = gathered.content

    return {
        "messages": messages,
        "ai_recommendation": ai_recommendation,
    }