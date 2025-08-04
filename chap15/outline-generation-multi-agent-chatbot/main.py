# main.py
from langgraph.graph import StateGraph, START, END
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"
from langchain_core.messages import HumanMessage

from models import State, state_init
from agents import agent_nodes
from agents.supervisor import supervisor_router
from agents.vector_search_agent import vector_search_router
from file_utils import save_state, clear_outline_and_state


# 상태 그래프 정의(랭그래프가 models/state의 State 클래스가 자신이 관리할 상태임을 인식)
graph_builder = StateGraph(State)

# Nodes
# agent_nodes는 딕셔너리라 key, value 받기 위해서는 .items()필요
# key(이름), node_func(함수 객체)
for name, node_func in agent_nodes.items():
    graph_builder.add_node(name, node_func)

# Edges
graph_builder.add_edge(START, "business_analysist")
graph_builder.add_edge("business_analysist", "supervisor")
graph_builder.add_conditional_edges(
    "supervisor",
    supervisor_router,
    {
        "content_strategist": "content_strategist",
        "communicator": "communicator",
        "vector_search_agent": "vector_search_agent",
        "web_search_agent": "web_search_agent",
    }
)
# 핵심 개선사항 1: vector_search_agent에서 조건부 분기
graph_builder.add_conditional_edges(
    "vector_search_agent",
    vector_search_router,  # 새로운 라우터 함수
    {
        "business_analysist": "business_analysist",  # 충분한 정보가 있을 때
        "web_search_agent": "web_search_agent",      # 정보가 부족할 때
    }
)

# 핵심 개선사항 2: web_search_agent에서 직접 business_analysist로
graph_builder.add_edge("web_search_agent", "business_analysist")

graph_builder.add_edge("content_strategist", "outline_reviewer")
graph_builder.add_edge("outline_reviewer", "business_analysist")
graph_builder.add_edge("communicator", END)

graph = graph_builder.compile()

# 그래프 도식화 하고 PNG 파일 생성하기
# 현재 파일(main.py)의 경로 기준
current_path = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_path, "data", "graph.png")

# 프로그램 시작 시 이전 세션에서 생성한 outline.md 파일이 있으면 삭제
clear_outline_and_state(current_path)

# 폴더가 없으면 생성
if not os.path.exists(os.path.dirname(output_path)):
    os.makedirs(os.path.dirname(output_path))

# 그래프 저장
graph.get_graph().draw_mermaid_png(output_file_path=output_path)
print("그래프 이미지 저장 완료")


# 상태 초기화
state = state_init()

# 터미널 창에서 사용자의 입력을 받고 graph를 실행(invoke)하는 부분
while True:
    user_input = input("\nUser\t: ").strip()

    if user_input.lower() in ["exit", "quit", "q"]:
        print("GoodBye!")
        break

    state["messages"].append(HumanMessage(user_input))
    state = graph.invoke(state)

    print('\n-------------------------------- MESSAGE COUNT\t', len(state["messages"]))

    # 현재 state 내용 파일로 저장
    save_state(current_path, state)