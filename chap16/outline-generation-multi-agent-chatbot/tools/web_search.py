# tools/web_search.py 

from tavily import TavilyClient
from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
import os
import json
from datetime import datetime

# 프로젝트 루트 경로 반환
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@tool
def web_search(query:str):
    """
    주어진 query에 대해 웹 검색을 하고 결과를 반환한다.

    Args:
        query (str): 검색어

    Returns:
        tuple: (검색 결과 리스트, 저장된 json 파일 경로)
    """
    client = TavilyClient()

    content = client.search(
        query,
        search_depth="advanced",
        include_raw_content=True,
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
    resource_json_path = f'{root_path}/data/resources_{datetime.now().strftime('%Y_%m%d_%H%M%S')}.json'
    if not os.path.exists(f"{root_path}/data"):
        os.makedirs(f"{root_path}/data")
    with open(resource_json_path, 'w', encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # 검색 결과와 저장된 파일 경로를 튜플로 반환
    return results, resource_json_path

# 타빌리 검색에서 raw_content(전문)을 불러오지 못하는 경우가 있다.
# WebBaseLoader를 활용한 페이지 전문을 읽는 함수
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
        # 검색 결과: 요소가 딕셔너리인 리스트
        resources = json.load(f)

    # documents 리스트에 저장
    documents = []

    for web_page in resources:
        document = web_page_to_document(web_page)
        documents.append(document)
    
    return documents

# 타빌리 검색 테스트
# if __name__ == "__main__":
#     result, resources_json_path = web_search.invoke("2025년 한국 영화 시장 전망")
#     print(result[0])