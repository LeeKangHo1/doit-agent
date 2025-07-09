# chap14/sec01/book_writer.py

# 커뮤니케이터 에이전트
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from typing_extensions import TypedDict
from typing import List

# 이후 작성
from utils import save_state, get_outline, save_outline

from datetime import datetime
import os

# 현재 폴더 경로 찾기
# 랭그래프 이미지로 저장 및 추후 작업 결과 파일 저장 경로로 활용
filename = os.path.basename(__file__) # 현재 파일명 반환
absolute_path = os.path.abspath(__file__) # 현재 파일의 절대 경로 반환
current_path = os.path.dirname(absolute_path) # 현재 .py 파일이 있는 폴더 경로

# 모델 초기화
llm = ChatOpenAI(model="gpt-4o-mini")

# 상태 정의
class State(TypedDict):
    # AnyMessage는 여러 종류의 메시지 타입(SystemMessage, HumanMessage, AIMessage, ToolMessage)을 하나로 표현하기 위한 type alias
    messages: List[AnyMessage | str]

# 목차를 작성하는 콘텐츠 전략가 에이전트
def content_strategist(state: State):
    print("\n\n======= CONTENT STRATEGIST =======")

    content_strategist_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI 팀의 콘텐츠 전략가(Content Strategist)로서,
        이전 대화 내용을 바탕으로 사용자의 요구 사항을 분석하고, AI팀이 쓸 책의 세부 목차를 결정한다.

        지난 목차가 있다면 그 버전을 사용자의 요구에 맞게 수정하고, 없다면 새로운 목차를 제안한다.

        ----------------------------
        - 지난 목차: {outline}
        ----------------------------
        - 이전 대화 내용: {messages}
        """
    )

    content_strategist_chain = content_strategist_system_prompt | llm | StrOutputParser()

    messages = state["messages"]
    outline = get_outline(current_path)

    inputs = {
        "messages": messages,
        "outline": outline,
    }

    # StrOutputParser 가 있어서 간단함
    gathered = ''
    for chunk in content_strategist_chain.stream(inputs):
        gathered += chunk
        print(chunk, end='')

    print()

    save_outline(current_path, gathered)

    content_strategist_message = f"[Content Strategist] 목차 작성 완료"
    print(content_strategist_message)
    messages.append(AIMessage(content_strategist_message))

    return {"messages": messages}

# 사용자와 대화할 노드(agent): communicator
def communicator(state: State):
    print("\n\n======== COMMUNICATOR =======")

    communicator_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI 팀의 커뮤니케이터로서,
        AI 팀의 진행 상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위해 대화를 나눈다.

        사용자도 outline(목차)을 이미 보고 있으므로, 다시 출력할 필요는 없다.

        messages: {messages}
        """
    )

    system_chain = communicator_system_prompt | llm

    # 상태에서 메시지 가져오기
    messages = state["messages"]

    # 입력값 정의
    inputs = {"messages": messages}

    # StrOutputParser 가 없어서 객체로 연결(None으로 설정하는 이유)
    gathered = None

    print('\nAI\t: ', end='')
    for chunk in system_chain.stream(inputs):
        print(chunk.content, end='')

        if gathered is None:
            gathered = chunk
        else:
            gathered += chunk
    messages.append(gathered)

    # dict만 반환해도 랭그래프가 알아서 state["messages"] 업데이트함
    return {"messages": messages}
    
# 상태 그래프 정의
graph_builder = StateGraph(State)

# Nodes
graph_builder.add_node("communicator", communicator)
graph_builder.add_node("content_strategist", content_strategist)

# Edges
graph_builder.add_edge(START, "content_strategist")
graph_builder.add_edge("content_strategist", "communicator")
graph_builder.add_edge("communicator", END)

graph = graph_builder.compile()

# 그래프 도식화하고 PNG 파일 생성하기
graph.get_graph().draw_mermaid_png(output_file_path=absolute_path.replace('.py', '.png'))


# 상태 초기화
state = State(
    messages= [
        # 여러 에이전트가 추가될 것을 고려하여 설계
        SystemMessage(
            f"""
            너희 AI들은 사용자의 요구에 맞는 책을 쓰는 작가 팀이다.
            사용자가 사용하는 언어로 대화하라.

            현재 시각은 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}이다.
            """
        )
    ],
)

# 터미널에서 사용자의 입력을 받고 그래프 실행하기
while True:
    user_input = input("\nUser\t: ").strip()

    if user_input.lower() in ['exit', 'quit', 'q']:
        print("Goodbye!")
        break

    state["messages"].append(HumanMessage(user_input))
    state = graph.invoke(state)

    print('\n------------------------------ MESSAGE COUNT\t', len(state["messages"]))

    # 현재 state 내용 저장
    save_state(current_path, state) 

