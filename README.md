# ops-bot

에이전트와의 대화를 시뮬레이션하기 위해 LiveKit 룸에 참여하는 봇 클라이언트입니다.

## 폴더 구조

```
ops-bot/
├── bot/
│   ├── .env                  # 환경 변수 (.env.example에서 복사)
│   ├── .env.example          # 환경 변수 템플릿
│   ├── bot_client.py         # 봇 클라이언트 구현
│   ├── config.py             # 설정 파일
│   └── stress_test_bots.py   # 스트레스 테스트 실행 스크립트
├── video/                    # 봇 스트리밍용 비디오 파일
│   └── bot1.mp4           # 샘플 비디오 파일
├── requirements.txt          # Python 의존성
└── README.md
```

## 설정

1. 환경 변수 템플릿 복사:
   ```bash
   cp bot/.env.example bot/.env
   ```

2. `bot/.env` 파일에 인증 정보 입력

3. 의존성 설치:
   ```bash
   pip install -r requirements.txt
   ```

## 봇 실행

```bash
cd bot
python stress_test_bots.py -n 1
```

**옵션:**
- `-n <숫자>`: 생성할 봇 인스턴스 수 (예: `-n 1`은 1개, `-n 10`은 10개)

## 동작 방식

봇이 실행되면:
1. ops-api를 통해 봇 세션 생성
2. LiveKit 룸에 참가자로 입장
3. 한국어로 초기 인사말 전송
4. 에이전트의 음성을 듣고 미리 정의된 문구로 응답


추가 다운로드 - EC2 instance용 
sudo apt-get update
sudo apt-get install -y ffmpeg libsm6 libxext6 libgl1