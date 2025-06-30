import pymupdf
import os

# PDF 파일 경로 설정
pdf_file_path = "chap04/data/과정기반 작물모형을 이용한 웹 기반 밀 재배관리 의사결정 지원시스템 설계 및 구축.pdf"

# PDF 문서 열기
# pymupdf.open()은 문서 전체를 불러오는 함수이며, 반환값은 문서 객체(doc)
doc = pymupdf.open(pdf_file_path)

# 헤더/푸터 높이 설정 (픽셀 기준)
header_height = 80
footer_height = 80

# 전체 본문 텍스트를 저장할 변수 초기화
full_text = ''

# 문서의 각 페이지 반복
for page in doc:
    rect = page.rect  # 페이지의 전체 크기를 나타내는 rectangle 객체 (x0, y0, x1, y1)

    # 페이지 상단 영역(헤더) 텍스트 추출 (선택적으로 사용할 수 있음)
    header = page.get_text(clip=(0, 0, rect.width , header_height))

    # 페이지 하단 영역(푸터) 텍스트 추출 (선택적으로 사용할 수 있음)
    footer = page.get_text(clip=(0, rect.height - footer_height, rect.width , rect.height))

    # 본문 영역 텍스트 추출 (헤더/푸터 제외 영역)
    text = page.get_text(clip=(0, header_height, rect.width , rect.height - footer_height))

    # 본문 텍스트 누적 저장 (페이지 구분선 추가)
    full_text += text + '\n------------------------------------\n'

# PDF 파일 이름만 추출 (확장자 제거)
pdf_file_name = os.path.basename(pdf_file_path)
pdf_file_name = os.path.splitext(pdf_file_name)[0]

# 출력 텍스트 파일 경로 설정 (원래 PDF 이름 기반)
txt_file_path = f'chap04/output/{pdf_file_name}_with_preprocessing.txt'

# 추출된 본문 텍스트를 파일로 저장
with open(txt_file_path, 'w', encoding='utf-8') as f:
    f.write(full_text)
