# 그래프 상태 선언
from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document

class GraphState(TypedDict):
    question: str # 사용자 질문
    generation: str # LLM 생성 결과
    documents: List[Document] # 필요없는 청크 거른 vectordb 검색 자료

from models_and_router import question_router

def route_question(state: GraphState):
    """
    사용자 질문을 vectorstore 또는 casual로 라우팅 합니다.
    """
    print('-----ROUTE-----')
    question = state['question']
    route = question_router.invoke({"question": question})

    print(f"---Routing to {route.datasource}---")
    return route.datasource

from data_setup import retriever
# retrieve 노드
def retrieve(state):
    """vectorstore에서 질문에 대한 문서를 검색합니다."""
    print('-----RETRIEVE-----')
    question = state['question']

    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

from rag_function import retrieval_grader
# grade_documents 노드
def grade_documents(state):
    """
    검색된 문서를 평가하여 질문과 관련성이 있는지 확인합니다.
    """
    print('-----GRADE-----')
    question = state['question']
    documents = state['documents']
    filtered_docs = []

    for i, doc in enumerate(documents):
        is_relevant = retrieval_grader.invoke({"question": question, "document": doc.page_content})
        if is_relevant.binary_score == "yes":
            filtered_docs.append(doc)
    
    return {"documents": filtered_docs, "question": question}

from rag_function import rag_chain

# generate 노드
def generate(state):
    """
    LLM을 사용하여 문서와 사용자 질문에 대한 답변을 생성합니다.
    """
    print('-----GENERATE-----')
    question = state["question"]
    documents = state['documents']
    generation = rag_chain.invoke({"question": question, "context": documents})
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
    }

from data_setup import model

# casual 대답을 하는 노드
def casual_talk(state):
    """일상 대화를 위한 답변을 생성합니다."""
    print('-----CASUAL_TALK-----')
    question = state['question']
    generation = model.invoke(question)
    return {
        "question": question,
        "generation": generation
    }