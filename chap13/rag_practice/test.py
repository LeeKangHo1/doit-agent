from workflow import app

inputs = {
    "question": "서울시 자율주행 계획"
}

for msg, meta in app.stream(inputs, stream_mode="messages"):
    print(msg.content, end="")