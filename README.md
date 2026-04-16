# WeatherAgent
인공지능기초 7주차 과제

단순히 날씨 수치만 전달하는 기존 챗봇의 한계를 넘어, 실시간 데이터와 맞춤형 로직을 결합하여 사용자에게 실질적인 가이드를 제공합니다.

# System Architecture
* UI: FastAPI를 통해 웹 기반 인터페이스를 제공하며, OpenAI Whisper 모델을 사용하여 사용자의 음성 입력을 텍스트로 변환하는 STT 기능을 수행합니다.

* Backend: OpenAI GPT-4.1-mini 모델이 시스템의 '두뇌' 역할을 하며, 사용자의 질문을 분석해 필요한 도구를 결정합니다. 이 과정에서 Function Calling 기술을 사용하여 날씨 정보 획득과 외출 체크리스트 생성을 연쇄적으로 판단합니다.

* External Data Layer: AI의 요청에 따라 서버가 실제 외부 API인 OpenWeatherMap에 접근하여 실시간 기상 데이터를 수집하거나, 사전에 정의된 파이썬 비즈니스 로직을 실행하여 복장 및 준비물 추천 데이터를 생성합니다.

# Setup
1. dependency 설치 (uv 사용):
```bash
uv sync
```

2. Environment Variable 설정
```bash
OPENAI_API_KEY=your_openai_key
WEATHERMAP_API_KEY=your_weathermap_api_key
WHISPER_MODEL=large-v3
```

WEATHERMAP_API_KEY는 [openweathermap.org](https://openweathermap.org/)에서 발급할 수 있습니다.

3. 앱 실행
```bash
uv run app.py
```