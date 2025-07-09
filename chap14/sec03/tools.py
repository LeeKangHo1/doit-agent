# sec03/tools.py

from tavily import TavilyClient
from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from datetime import datetime
import json
import os
absolute_path = os.path.abspath(__file__) # __file__은 현재 실행 중인 Python 파일의 경로(파일명 포함)를 나타내는 특수 변수입니다.
current_path = os.path.dirname(absolute_path)

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
    with open(resource_json_path, 'w', encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # 튜플 언팩킹
    return results, resource_json_path

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

# 이 파일 테스트용 코드
if __name__ == "__main__":
    results, resource_json_path = web_search.invoke("2025년 한국 경제 전망")
    print(results)

    # result = load_web_page("https://www.kofic.or.kr/kofic/business/board/selectBoardList.do?boardNumber=6")
    # print(result)