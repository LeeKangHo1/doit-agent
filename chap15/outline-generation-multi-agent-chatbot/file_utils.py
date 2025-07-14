# file_utils.py

import os
import json

# 대화 기록을 json 파일에 저장
def save_state(root_path, state):
    if not os.path.exists(f"{root_path}/data"):
        os.makedirs(f"{root_path}/data")
    
    state_dict = {}

    messages = [(m.__class__.__name__, m.content) for m in state["messages"]]
    state_dict["messages"] = messages
    state_dict["task_history"] = [task.to_dict() for task in state.get("task_history", [])]

    # references
    references = state.get("references", {"queries": [], "docs": []})
    state_dict["references"] = {
        "queries": references["queries"],
        # 전체 내용 말고 metadata(title, source)만 저장
        "docs": [doc.metadata for doc in references["docs"]]
    }
    state_dict["user_request"] = state.get("user_request", "")

    with open(f"{root_path}/data/state.json", 'w', encoding='utf-8') as f:
        # indent -> 들여쓰기할 칸 수. 4면 스페이스 4번, tab 1번 정도로 들여쓰기함
        json.dump(state_dict, f, indent=4, ensure_ascii=False)

# 저장된 outline 정보가 있으면 가져온다.
def get_outline(root_path):
    outline = '아직 작성된 목차가 없습니다.'
    
    if os.path.exists(f"{root_path}/data/outline.md"):
        with open(f"{root_path}/data/outline.md", "r", encoding='utf-8') as f:
            outline = f.read()
    return outline

# outline을 저장한다.
def save_outline(root_path, outline):
    if not os.path.exists(f"{root_path}/data"):
        os.makedirs(f"{root_path}/data")
    
    with open(f"{root_path}/data/outline.md", "w", encoding='utf-8') as f:
        f.write(outline)

    return outline

def clear_outline_and_state(root_path):
    """기존 목차 파일과 상태 파일(state.json)을 삭제합니다."""
    data_dir = os.path.join(root_path, "data")

    outline_path = os.path.join(data_dir, "outline.md")
    state_path = os.path.join(data_dir, "state.json")

    if os.path.exists(outline_path):
        os.remove(outline_path)

    if os.path.exists(state_path):
        os.remove(state_path)


    