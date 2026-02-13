# Oracle Always Free 24시간 운영 가이드 (Ubuntu 22.04)

이 문서는 `bithumb_tradbot`을 Oracle Cloud Always Free VM에서 24시간 서비스로 실행하는 절차를 설명합니다.

## 0. 목표

- 서버 재부팅 후에도 자동 시작
- 프로세스 죽으면 자동 재시작
- 로그 상시 확인 가능
- 설정/키를 코드와 분리(`config.yaml`, `.env`)

## 1. 서버 준비 (Oracle 콘솔)

1. Oracle Cloud에서 Compute 인스턴스 생성
2. 이미지: Ubuntu 22.04
3. Shape: Always Free 가능 Shape (예: A1 Flex)
4. SSH 키 등록 후 접속

참고 공식 문서:
- https://docs.oracle.com/iaas/Content/Compute/Tasks/launchinginstance.htm
- https://docs.oracle.com/iaas/Content/Compute/Tasks/connect-to-linux-instance.htm
- https://docs.oracle.com/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm

## 2. 코드 배치

```bash
git clone <YOUR_REPO_URL> /home/ubuntu/bithumb_tradbot
cd /home/ubuntu/bithumb_tradbot
```

## 3. 환경파일 준비

### `/home/ubuntu/bithumb_tradbot/.env`

```env
BITHUMB_API_KEY=...
BITHUMB_SECRET_KEY=...
GEMINI_API_KEY=...
DISCORD_WEBHOOK_URL=...
BITHUMB_API_BASE_URL=https://api.bithumb.com
```

### `/home/ubuntu/bithumb_tradbot/config.yaml`

실거래 기준 최소 확인:

```yaml
app:
  dry_run: false
  enable_official_orders: true

trade:
  seed_krw: 70000
  slot_count: 1

notification:
  enabled: true
```

## 4. 설치/서비스 등록 (자동)

프로젝트에 포함된 스크립트를 사용:

```bash
cd /home/ubuntu/bithumb_tradbot
bash deploy/oracle/install_ubuntu_22.sh /home/ubuntu/bithumb_tradbot
```

이 스크립트는 다음을 수행합니다.

- OS 패키지 설치
- Python venv 생성
- `requirements.txt` 설치
- systemd 서비스 생성/시작

## 5. 상태 확인

```bash
sudo systemctl status bithumb-tradbot --no-pager
journalctl -u bithumb-tradbot -f
```

정상 로그 예:

- `Trading loop started ...`
- `Watchlist updated ...`
- `API usage public(... fail=0)`

## 6. 운영 명령어

```bash
# 재시작
sudo systemctl restart bithumb-tradbot

# 중지
sudo systemctl stop bithumb-tradbot

# 시작
sudo systemctl start bithumb-tradbot

# 부팅 자동시작 해제
sudo systemctl disable bithumb-tradbot
```

## 7. 배포 체크리스트 (실거래 전)

아래 명령을 수동으로 1회 점검:

```bash
/home/ubuntu/bithumb_tradbot/.venv/bin/python /home/ubuntu/bithumb_tradbot/main.py --check-account
/home/ubuntu/bithumb_tradbot/.venv/bin/python /home/ubuntu/bithumb_tradbot/main.py --validate-data
/home/ubuntu/bithumb_tradbot/.venv/bin/python /home/ubuntu/bithumb_tradbot/main.py --test-notify
/home/ubuntu/bithumb_tradbot/.venv/bin/python /home/ubuntu/bithumb_tradbot/main.py --run-once
```

기준:

- `--check-account`: `private_api_enabled=true`, `error=null`
- `--validate-data`: `ok=true`
- `--test-notify`: `sent=true`
- `--run-once`: 예외 없이 종료, API usage 로그 출력

## 8. 트러블슈팅

- 알림이 안 오면:
  - `.env`의 `DISCORD_WEBHOOK_URL` 확인
  - `--test-notify` 실행 결과 확인
- 주문이 안 나가면:
  - `config.yaml`의 `app.dry_run`, `app.enable_official_orders` 확인
  - `trade.min_order_krw`가 너무 큰지 확인
- 프로세스가 죽으면:
  - `journalctl -u bithumb-tradbot -n 200 --no-pager`로 최근 에러 확인

## 9. 보안

- API 키는 절대 Git에 커밋하지 말 것
- 보안그룹(보안목록)은 SSH(22)만 허용 권장
- 서버 사용자 비밀번호 로그인 비활성 + SSH 키 로그인 권장
