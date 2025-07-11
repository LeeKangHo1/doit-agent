# tools/vector_search.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os

from .web_search import web_page_json_to_documents

# 프로젝트 루트 경로 반환
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

embedding = OpenAIEmbeddings(model="text-embedding-3-large")
persist_directory = f"{root_path}/data/chroma_store"

# Chroma 객체 생성
vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding,
)

# 문서들(documents)을 지정된 크기의 청크 단위로 분할
def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    print('Splitting documents...')
    print(f"{len(documents)}개의 문서를 {chunk_size}자 크기로 중첩 {chunk_overlap}자로 분할합니다.\n")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    splits = text_splitter.split_documents(documents)

    print(f"총 {len(splits)}개의 문서로 분할되었습니다.")
    return splits

# 중복되지 않은 URL의 documents만 ChromaDB에 저장
def documents_to_chroma(documents, chunk_size=1000, chunk_overlap=100):
    print("Documents를 Chroma DB에 저장할 준비를 합니다.")

    urls = [document.metadata['source'] for document in documents]

    stored_metadatas = vectorstore._collection.get()['metadatas']
    stored_web_urls = [metadata['source'] for metadata in stored_metadatas]

    # url 중복 제거, new_urls는 set 자료형
    new_urls = set(urls) - set(stored_web_urls)

    new_documents = []

    for document in documents:
        # in 뒤에 set자료형 가능. 심지어 list보다 빠름
        if document.metadata["source"] in new_urls:
            new_documents.append(document)
            print(document.metadata)

    splits = split_documents(new_documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if splits:
        vectorstore.add_documents(splits)
        print("벡터 DB에 문서 추가 완료")
    else:
        print("No new urls to process")

# json 파일 내용을 chromaDB에 추가
def add_web_pages_json_to_chroma(json_file, chunk_size=1000, chunk_overlap=100):
    documents = web_page_json_to_documents(json_file)
    documents_to_chroma(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

# 리트리버 만들기
@tool
def retrieve(query: str, top_k: int=5):
    """
    주어진 query에 대해 벡터 검색을 수행하고, 결과를 반환한다.
    """
    # 다양한 검색 파라미터(가져오는 청크 수)를 하나의 딕셔너리(search_kwargs)에 저장
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    retrieved_docs = retriever.invoke(query)

    return retrieved_docs

# chromaDB 저장 테스트
# if __name__ == "__main__":
    # add_web_pages_json_to_chroma(f'{root_path}/data/resources_2025_0711_150931.json')