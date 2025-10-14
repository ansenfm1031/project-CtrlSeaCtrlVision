# 파일 이름: start_server.sh
#!/bin/bash
<<<<<<< Updated upstream
=======
deactivate
>>>>>>> Stashed changes
source .venv_server/bin/activate
# 1. API 키 설정 (가장 중요)
# 'sk-your-private-key-here' 부분을 발급받은 실제 OpenAI API 키로 대체하세요.
export OPENAI_API_KEY='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'

# 2. 파이썬 서버 코드 실행
# your_server_code.py는 현재 사용하시는 파이썬 파일명으로 바꿔주세요.
python3 server.py

# 스크립트 종료
