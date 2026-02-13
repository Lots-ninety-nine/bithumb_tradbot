# Oracle Always Free 24시간 운영 가이드 (Bybit Tradbot)

이 문서는 `bybit_tradbot`을 Oracle Cloud Always Free VM에서 24시간 서비스로 운영하는 절차입니다.

## 1. 서버 준비

1. Oracle Cloud Compute 인스턴스 생성
2. 이미지: Ubuntu 22.04
3. SSH 키 등록 후 접속

공식 문서:
- https://docs.oracle.com/iaas/Content/Compute/Tasks/launchinginstance.htm
- https://docs.oracle.com/iaas/Content/Compute/Tasks/connect-to-linux-instance.htm
- https://docs.oracle.com/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm

## 2. 코드 배치

```bash
git clone <YOUR_REPO_URL> /home/ubuntu/bybit_tradbot
cd /home/ubuntu/bybit_tradbot
```

## 3. 환경 파일

`/home/ubuntu/bybit_tradbot/.env`

```env
BYBIT_API_KEY=...
BYBIT_API_SECRET=...
GEMINI_API_KEY=...
DISCORD_WEBHOOK_URL=...
```

`/home/ubuntu/bybit_tradbot/config.yaml`

```yaml
app:
  dry_run: false

trade:
  seed_capital: 100
  slot_count: 2

notification:
  enabled: true
```

## 4. 설치/서비스 등록

```bash
cd /home/ubuntu/bybit_tradbot
bash deploy/oracle/install_ubuntu_22.sh /home/ubuntu/bybit_tradbot
```

스크립트 동작:
- Python venv 생성
- 의존성 설치
- systemd 서비스(`bybit-tradbot`) 생성/시작

## 5. 상태 확인

```bash
sudo systemctl status bybit-tradbot --no-pager
journalctl -u bybit-tradbot -f
```

정상 로그 예:
- `Trading loop started ...`
- `Watchlist updated ...`
- `API usage public(... fail=0)`

## 6. 운영 명령

```bash
sudo systemctl restart bybit-tradbot
sudo systemctl stop bybit-tradbot
sudo systemctl start bybit-tradbot
sudo systemctl disable bybit-tradbot
```

## 7. 실거래 전 체크

```bash
/home/ubuntu/bybit_tradbot/.venv/bin/python /home/ubuntu/bybit_tradbot/main.py --check-account
/home/ubuntu/bybit_tradbot/.venv/bin/python /home/ubuntu/bybit_tradbot/main.py --validate-data
/home/ubuntu/bybit_tradbot/.venv/bin/python /home/ubuntu/bybit_tradbot/main.py --test-notify
/home/ubuntu/bybit_tradbot/.venv/bin/python /home/ubuntu/bybit_tradbot/main.py --run-once
```

기준:
- `--check-account`: `private_api_enabled=true`, `error=null`
- `--validate-data`: `ok=true`
- `--test-notify`: `sent=true`

## 8. 트러블슈팅

- 알림 미수신:
  - `DISCORD_WEBHOOK_URL` 확인
  - `--test-notify` 점검
- 주문 실패:
  - Bybit API 권한(거래 권한) 확인
  - `bybit.position_idx`/계정 포지션 모드 일치 확인
  - `trade.min_order_notional` 과도 설정 여부 확인
- 서비스 다운:
  - `journalctl -u bybit-tradbot -n 200 --no-pager`

## 9. 보안

- API 키는 절대 Git 커밋 금지
- SSH(22) 외 포트는 최소 허용
- 비밀번호 로그인 비활성 + SSH 키 로그인 권장
