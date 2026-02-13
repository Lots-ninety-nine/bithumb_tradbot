# Raspberry Pi Appliance 모드 (Debian/RPi5)

목표: 홈브릿지처럼 부팅 후 자동 실행 + 웹으로 상태/시작/중지 제어.

## 1) 개요

설치 스크립트가 아래를 자동으로 구성합니다.

- `bithumb-tradbot.service` (봇)
- `bithumb-tradbot-web.service` (웹 제어 패널)
- 웹 URL: `http://<라즈베리파이IP>:8080`
- 웹 토큰: `/etc/bithumb-tradbot/appliance.env`의 `UI_TOKEN`

## 2) 설치 순서

라즈베리파이에 프로젝트를 복사/클론한 뒤:

```bash
cd /path/to/bithumb_tradbot
BOT_USER=pi sudo bash deploy/raspi/install_appliance.sh "$(pwd)"
```

`BOT_USER`는 봇을 실행할 사용자 계정입니다(보통 `pi`).

## 3) 웹 제어

브라우저에서 아래 접속:

- `http://<라즈베리파이IP>:8080`

가능한 동작:

- Bot Start
- Bot Stop
- Bot Restart
- runtime.log/journal tail 확인

## 4) 운영 명령어

```bash
sudo systemctl status bithumb-tradbot
sudo systemctl status bithumb-tradbot-web
sudo systemctl restart bithumb-tradbot
sudo systemctl restart bithumb-tradbot-web
tail -f /path/to/bithumb_tradbot/logs/runtime.log
```

## 5) 보안 주의

- 기본 포트 `8080`은 내부망에서만 접근 권장.
- `UI_TOKEN`은 노출 금지.
- 외부 공개 시 라우터 포트포워딩은 권장하지 않습니다.

## 6) 참고

- 맥북 `nohup`은 터미널 종료에는 버티지만, sleep/전원 종료에는 멈출 수 있습니다.
- 라즈베리파이 서비스 모드가 24시간 운영에 더 적합합니다.
