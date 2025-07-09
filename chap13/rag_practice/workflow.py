# StateGraph 객체 생성하기
from langgraph.graph import START, StateGraph, END
from graph_nodes import GraphState, retrieve, grade_documents, generate, casual_talk, route_question

workflow = StateGraph(GraphState)

# 노드 등록
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("casual_talk", casual_talk)

# 엣지 연결 + 라우팅
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "vectorstore": "retrieve",
        "casual": "casual_talk",
    }
)

workflow.add_edge("casual_talk", END)
workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("grade_documents", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

from IPython.display import Image, display

# try:
#     display(Image(app.get_graph().draw_mermaid_png()))
# except Exception:
#     pass

# 이미지를 파일로 저장
# try:
#     with open("workflow_graph.png", "wb") as f:
#         f.write(app.get_graph().draw_mermaid_png())
#     print("그래프가 workflow_graph.png로 저장되었습니다!")
# except Exception as e:
#     print(f"그래프 저장 실패: {e}")