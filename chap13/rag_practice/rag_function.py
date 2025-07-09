from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from data_setup import model

# yes or no로 청크 거르는 클래스
class GradeDocuments(BaseModel):
    """검색된 문서가 질문과 관련이 있는지 'yes' 또는 'no로 평가합니다."""

    binary_score: Literal["yes", "no"] = Field(
        # description 생략
    )

structured_llm_grader = model.with_structured_output(GradeDocuments)

# grade 프롬프트 작성
grader_prompt = PromptTemplate.from_template("""
당신은 검색된 문서가 사용자 질문과 관련이 있는지 평가하는 평가자입니다. \n
문서에 사용자 질문과 관련된 키워드 또는 의미가 포함되어 있으면, 해당 문서를 관련성이 있다고 평가하십시오. \n
엄격한 테스트가 필요하지 않습니다. 목표는 잘못된 검색 결과를 걸러내는 것입니다. \n
문서가 질문과 관련이 있는지 여부를 나타내기 위해 'yes' 또는 'no'로 이진 점수를 부여하십시오      

Retrieved document: \n {document} \n\n
User question: {question}                                                                                                                                                                                                                              
""")

retrieval_grader = grader_prompt | structured_llm_grader

# question = "서울시 자율주행 관련 계획"
# documents = retriever.invoke(question)
# # 관련 청크만 리스트에 추가
# filtered_docs = []

# for i, doc in enumerate(documents):
#     print(f"Document {i+1}:")
#     is_relevant = retrieval_grader.invoke({"question": question, "document": doc.page_content})
#     print(is_relevant)
#     print(doc.page_content[:100])

#     if is_relevant.binary_score == "yes":
#         filtered_docs.append(doc)

# print(f"Filtered documents: {len(filtered_docs)}")


# RAG 답변 생성을 위한 프롬프트 생성
rag_generate_system = """
너는 사용자의 질문에 대해 주어진 context에 기반하여 답변하는 도시 계획 전문가이다.
주어진 context는 vectorstore에서 검색된 결과이다.
주어진 context를 기반으로 사용자의 question에 대해 답변하라.

=============================
question: {question}
context: {context}
"""

rag_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=rag_generate_system,
)

rag_chain = rag_prompt | model