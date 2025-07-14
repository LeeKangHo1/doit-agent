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

    task_history = state.get("task_history", [])
    task = task_history[-1]
    if task.agent != "content_strategist":
        raise ValueError(f"Content Strategist가 아닌 agent가 목차 작성을 시도하고 있습니다.\n {task}")

    content_strategist_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI 팀의 콘텐츠 전략가(content_strategist)로서,
        이전 대화 내용을 바탕으로 사용자의 요구 사항을 분석하고, AI팀이 쓸 책의 세부 목차를 결정한다.

        지난 목차가 있다면 그 버전을 사용자의 요구에 맞게 수정하고, 없다면 새로운 목차를 제안한다.
        목차를 작성하는 데 필요한 정보는 "참고 자료"에 있으므로 활용한다.

        다음 정보를 활용하여 목차를 작성하라.
        - 사용자 요구사항(user_request)
        - 작업(task)
        - 검색 자료(references)
        - 기존 목차(previous_outline)
        - 이전 대화 내용(messages)

        너의 작업 목표는 다음과 같다:
        1. 만약 "기존 목차 구조(previous_outline)"이 존재한다면, 사용자의 요구사항을 토대로 "기존 목차 구조"에서 어떤 부분을 수정하거나 추가할지 결정한다.
        - "이번 목차 작성의 주안점"에 사용자 요구사항(user_request)을 충족시키는 것을 명시해야한다.
        2. 책의 전반적인 구조(chapter, section)를 설계하고, 각 chapter와 section의 제목을 정의한다.
        3. 책의 전반적인 세부 구조(chapter, section, sub-section)를 설계하고, sub-section 하부의 주요 내용을 리스트 형태로 정리한다.
        4. 목차의 논리적인 흐름이 사용자 요구를 충족시키는지  확인한다.
        5. 참고 자료(references)를 적극 활용하여 근거에 기반한 목차를 작성한다.
        6. 참고 문헌은 반드시 참고 자료(references) 자료를 근거로 작성해야 하며, 최대한 풍부하게 준비한다. URL은 전체 주소를 적어야 한다.
        7. 추가 자료나 리서치가 필요한 부분을 파악하여 supervisor에게 요청한다.

        사용자 요구사항(user_request)을 최우선으로 반영하는 목차로 만들어야 한다.

        --------------------------
        - 사용자 요구사항(user_request):
        {user_request}
        --------------------------
        - 작업(task):
        {task}
        --------------------------
        - 참고 자료(references):
        {references}
        --------------------------
        - 기존 목차(previous_outline):
        {previous_outline}
        --------------------------
        - 이전 대화 내용: 
        {messages}
        --------------------------

        작성 형식 아래 양식을 지키되 하부 항목으로 더 세분화해도 좋다. 목차(outline) 양식의 챕터, 섹션 등 항목의 개수는 필요한 만큼 추가하라.
        섹션 개수는 최소 2개 이상이어야 하며, 더 많으면 좋다.

        outline_template은 예시로 앞부분만 제시한 것이다. 각 장은 ':---CHAPTER DIVIDER---:'로 구분한다.
        outline_template:
        {outline_template}
        사용자가 피드백을 추가로 제공할 수 있도록 논리적인 흐름과 주요 목차 아이디어를 제안하라.

        """
    )

    # 시스템 프롬프트와 모델 연결
    content_strategist_chain = content_strategist_system_prompt | llm | StrOutputParser()

    # 목차 생성 에이전트가 현재 사용자의 요구 사항을 염두에 두고 작업하도록 유도
    # user_request는 business_analysist가 생성
    user_request = state.get("user_request", "")
    messages = state["messages"]
    previous_outline = get_outline(root_path)

    # 템플릿 이용하기
    with open(f"{root_path}/template/outline_template.md", "r", encoding='utf-8') as f:
        outline_template = f.read()

    inputs = {
        "user_request": user_request,
        "task": task,
        "messages": messages,
        "previous_outline": previous_outline,
        "references": state.get("references", {"queries": [], "docs": []}),
        "outline_template": outline_template,
    }

    gathered = ''
    for chunk in content_strategist_chain.stream(inputs):
        gathered += chunk
        print(chunk, end="")

    print()

    # 목차 저장
    save_outline(root_path, gathered)

    # 작업 후기 가져오기
    if '-----: DONE :-----' in gathered:
        # split('구분자')[1] 이므로 구분자 뒤의 내용을 가져온다. (0은 구분자 앞)
        review = gathered.split('-----: DONE :-----')[1] 
    else:
        # 끝에서부터 200자만 잘라서 review로 사용
        review = gathered[-200:]

    content_strategist_message = f"[Content Strategist] 목차 작성 완료: outline 작성 완료\n {review}"
    print(content_strategist_message)
    messages.append(AIMessage(content_strategist_message))
    
    task_history[-1].done = True
    task_history[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    return {
        "messages": messages,
        "task_history": task_history,
    }

