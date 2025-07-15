# test_llm_identity.py

# agents 폴더의 llm 객체를 불러옵니다
from agents import llm

def test_llm_identity():
    print(f"📦 LLM 객체 타입: {type(llm)}\n")

    try:
        print("🤖 모델에게 물어봅니다: '지금 답변하는 너는 누구야?'")
        response = llm.invoke("지금 답변하는 너는 누구야?")
        print("\n💬 응답:\n", response.content)
    except Exception as e:
        print("❌ LLM 호출 중 오류 발생:", e)

if __name__ == "__main__":
    test_llm_identity()
