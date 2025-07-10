# sec03/tools.py

from tavily import TavilyClient
from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from datetime import datetime
import json
import os
absolute_path = os.path.abspath(__file__) # __file__은 현재 실행 중인 Python 파일의 경로(파일명 포함)를 나타내는 특수 변수입니다.
current_path = os.path.dirname(absolute_path)

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embedding = OpenAIEmbeddings(model="text-embedding-3-large")
persist_directory = f"{current_path}/data/chroma_store"

vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

# 타빌리 검색 함수
@tool
def web_search(query: str):
    """
    주어진 query에 대해 웹 검색을 하고 결과를 반환한다.

    Args:
        query (str): 검색어

    Returns:
        tuple: (검색 결과 리스트, 저장된 json 파일 경로)
    """
    # json 파일로 저장하는 코드 추가하면서 return이 튜플 언팩킹으로 2개(results, resource_json_path)를 반환해서 내가 임의로 수정
    
    client = TavilyClient()

    content = client.search(
        query,
        search_depth="advanced",
        include_raw_content=True, # 페이지 전문을 가져오기 위한 옵션
    )

    results = content["results"]

    for result in results:
        if result["raw_content"] is None:
            try:
                result["raw_content"] = load_web_page(result["url"])
            except Exception as e:
                print(f"Error loading page: {result['url']}")
                print(e)
                result["raw_content"] = result["content"]

    # 검색 결과 JSON 파일로 저장
    resource_json_path = f'{current_path}/data/resources_{datetime.now().strftime('%Y_%m%d_%H%M%S')}.json'
    if not os.path.exists(f"{current_path}/data"):
        os.makedirs(f"{current_path}/data")
    with open(resource_json_path, 'w', encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # 검색 결과와 저장된 파일 경로를 튜플로 반환
    return results, resource_json_path

# 하나의 웹 페이지 정보가 들어오면 이를 랭체인의 Document 객체로 변환
def web_page_to_document(web_page):
    # 최근 웹 페이지 내용이 삭제되었거나 수정된 경우 raw_content가 content 보다 정보가 부족한 경우가 있다.
    if len(web_page['raw_content']) > len(web_page['content']):
        page_content = web_page['raw_content']
    else:
        page_content = web_page['content']
    
    # 랭체인에서 제공하는 Document 클래스는 기본적으로 벡터 검색에 활용할 수 있는 문서의 실제 내용인 page_content와 문서에 대한 추가 정보인 metadata를 갖고 있습니다.
    document = Document(
        page_content=page_content,
        metadata={
            'title': web_page['title'],
            'source': web_page['url'],
        }
    )

    return document

# json 파일을 받아서 웹페이지 하나당 하나의 document로 변환
def web_page_json_to_documents(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        # 검색 결과 딕셔너리들의 리스트
        resources = json.load(f)

    # documents 리스트에 저장
    documents = []

    for web_page in resources:
        document = web_page_to_document(web_page)
        documents.append(document)
    
    return documents

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
    print("Documents를 Chroma DB에 저장합니다.")

    urls = [document.metadata['source'] for document in documents]

    stored_metadatas = vectorstore._collection.get()['metadatas']
    stored_web_urls = [metadata['source'] for metadata in stored_metadatas]

    # url 중복 제거, new_urls는 set 자료형
    new_urls = set(urls) - set(stored_web_urls)

    new_documents = []

    for document in documents:
        # in set자료형 가능. 심지어 list보다 빠름
        if document.metadata["source"] in new_urls:
            new_documents.append(document)
            print(document.metadata)

    splits = split_documents(new_documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if splits:
        vectorstore.add_documents(splits)
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

# 타빌리 검색에서 raw_content(전문)을 불러오지 못하는 경우가 있다.
# WebBaseLoader를 활용한 페이지 전문을 읽는 함수 필요
def load_web_page(url: str):
    print("=============== WebBaseLoader 사용=================")
    loader = WebBaseLoader(url, verify_ssl=False) # ssl 인증서 검증 생략. 보안 문제 생길 수 있음

    content = loader.load()

    # WebBaseLoader는 여러 URL을 한 번에 처리하여 리스트로 반환하는데 여기서는 URL을 하나씩 처리하기 때문에 [0]만 호출
    raw_content = content[0].page_content.strip()

    # 과도한 줄바꿈 정리
    while '\n\n\n' in raw_content or '\t\t\t' in raw_content:
        raw_content = raw_content.replace('\n\n\n', '\n\n')
        raw_content = raw_content.replace('\t\t\t', '\t\t')

    return raw_content

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

# 이 파일 테스트용 코드
if __name__ == "__main__":
    # results, resource_json_path = web_search.invoke("2025년 한국 경제 전망")
    # print(results)

    # result = load_web_page("https://www.kofic.or.kr/kofic/business/board/selectBoardList.do?boardNumber=6")
    # print(result)

    # documents = web_page_json_to_documents(f'{current_path}/data/resources_2025_0709_175903.json')

    # splits = split_documents(documents)

    # add_web_pages_json_to_chroma(f'{current_path}/data/resources_2025_0709_175903.json')

    retrieved_docs = retrieve.invoke({"query": "한국 경제 위험 요소"})
    print(retrieved_docs)