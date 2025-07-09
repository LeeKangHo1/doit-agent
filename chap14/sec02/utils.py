# chap14/sec01/utils.py

import os
import json

# state 대화 내용 json으로 따로 저장
def save_state(current_path, state):
    if not os.path.exists(f"{current_path}/data"):
        os.makedirs(f"{current_path}/data")

    state_dict = {}

    # m.__class__.__name__: 클래스 이름 (예: "HumanMessage", "AIMessage" 등), m.content: 실제 메시지 내용 (텍스트)
    messages = [(m.__class__.__name__, m.content) for m in state["messages"]]
    state_dict["messages"] = messages
    state_dict["task_history"] = [task.to_dict() for task in state.get("task_history", [])]

    with open(f"{current_path}/data/state.json", "w", encoding="utf-8") as f:
        # indent -> 들여쓰기할 칸 수. 4면 스페이스 4번, tab 1번 정도로 들여쓰기함
        json.dump(state_dict, f, indent=4, ensure_ascii=False)

# 작성된 목차가 있을 경우 불러오기
def get_outline(current_path):
    outline = '아직 작성된 목차가 없습니다.'

    if os.path.exists(f"{current_path}/data/outline.md"):
        with open(f"{current_path}/data/outline.md", "r", encoding='utf-8') as f:
            outline = f.read()
    return outline

# 작성한 목차 저장
def save_outline(current_path, outline):
    if not os.path.exists(f"{current_path}/data"):
        os.makedirs(f"{current_path}/data")

    with open(f"{current_path}/data/outline.md", "w", encoding='utf-8') as f:
        f.write(outline)
    return outline