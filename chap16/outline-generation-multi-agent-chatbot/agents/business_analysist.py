# agents/business_analysist.py

from . import llm, agent_log, root_path
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from models import State
from file_utils import save_state, get_outline

# 사용자의 의도를 파악하는 에이전트
def business_analysist(state: State):
    agent_log("business_analysist")

    business_analysist_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI 팀의 비즈니스 애널리스트로서,
        AI 팀의 진행상황과 "사용자 요구 사항"을 토대로,
        현 시점에서 'ai_recommendation'과 최근 사용자의 발언을 바탕으로 요구 사항이 무엇인지 판단한다.
        지난 요구 사항이 달성되었는지 판단하고, 현 시점에서 어떤 작업을 해야 하는지 결정한다.

        다음과 같은 템플릿 형태로 반환하다.
        ```
        - 목표: 0000 \n 방법: 0000
        ```

        --------------------------
        *AI 추천(ai_recommendation)*: {ai_recommendation}
        ----------------------------
        최근 사용자의 발언: {user_last_comment}
        --------------------------
        참고 자료: {references}
        --------------------------
        목차 (outline): {outline}
        --------------------------
        "messages": {messages}
        """
    )

    ba_chain = business_analysist_system_prompt | llm | StrOutputParser()

    # 상태에서 메시지 가져오기
    messages = state["messages"]

    # 사용자의 마지막 발언 가져오기
    user_last_comment = None
    # messages[::-1]는 messages 리스트를 역순으로 복사한 새로운 리스트를 반환
    for m in messages[::-1]:
        if isinstance(m, HumanMessage):
            user_last_comment = m.content
            break

    # 입력 자료 준비
    inputs = {
        "ai_recommendation": state.get("ai_recommendation", None),
        "references": state.get("references", {"queries": [], "docs": []}),
        "outline": get_outline(root_path),
        "messages": messages,
        "user_last_comment": user_last_comment,
    }

    user_request = ba_chain.invoke(inputs)

    business_analysist_message = f"[Business Analyst] {user_request}"
    print(business_analysist_message)
    messages.append(AIMessage(business_analysist_message))

    # 원래는 사용자가 메시지를 입력한 직후에 저장했지만 앞으로는 사용자가 입력하지 않아도
    # 알아서 루프를 여러 번 돌 수 있으므로 여기에서도 state를 저장합니다.
    save_state(root_path, state)
    
    return {
        "messages": messages,
        "user_request": user_request,
        # business_analysist만 참고하고 다른 에이전트들이 영향 받지 않도록 비워준다.
        "ai_recommendation": "",
    }
