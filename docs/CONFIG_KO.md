# 설정 가이드 (한국어)

이 프로젝트는 매매 파라미터를 `config.yaml`에서 읽습니다.

- 설정 파일: `/Users/mosiwon/dev/bithumb_tradbot/config.yaml`
- API 키 파일: `/Users/mosiwon/dev/bithumb_tradbot/.env`

## 1. 필수 준비

`.env`에 키 입력:

```env
BITHUMB_API_KEY=...
BITHUMB_SECRET_KEY=...
GEMINI_API_KEY=...
BITHUMB_API_BASE_URL=https://api.bithumb.com
```

## 2. 실행 방법

```bash
# 계좌/API 체크
/Users/mosiwon/dev/bithumb_tradbot/.venv/bin/python /Users/mosiwon/dev/bithumb_tradbot/main.py --check-account

# 데이터/지표 품질 체크
/Users/mosiwon/dev/bithumb_tradbot/.venv/bin/python /Users/mosiwon/dev/bithumb_tradbot/main.py --validate-data

# 1회 실행
/Users/mosiwon/dev/bithumb_tradbot/.venv/bin/python /Users/mosiwon/dev/bithumb_tradbot/main.py --run-once

# 무한 루프 실행
/Users/mosiwon/dev/bithumb_tradbot/.venv/bin/python /Users/mosiwon/dev/bithumb_tradbot/main.py
```

## 3. 자주 조절하는 파라미터

### 손절/익절
- `risk.stop_loss_pct`: 손절 비율
  - 예: `0.05` = 진입가 대비 -5%면 손절
- `risk.trailing_start_pct`: 트레일링 시작 수익률
- `risk.trailing_gap_pct`: 고점 대비 이탈 허용폭

### 매수 강도
- `llm.min_buy_confidence`: Gemini 신뢰도 하한
- `llm.max_dead_cat_risk`: 데드캣 리스크 허용치 상한
- `strategy.required_signal_count`: 하드룰 충족 최소 개수(3개 지표 중)

### 시장/체결 안전장치
- `trade.max_spread_bps`: 스프레드 허용치 (bps)
- `trade.min_order_krw`: 최소 주문 금액
- `trade.slot_count`: 동시 포지션 슬롯 수
- `trade.seed_krw`: 총 운용 자본

### 운영
- `app.dry_run`: `true`면 주문 없이 로그만 출력
- `app.enable_official_orders`: REST 주문 API 허용 여부
- `app.interval_sec`: 루프 주기(초)

### 뉴스/RAG
- `news.enabled`: 뉴스 자동수집 on/off
- `news.refresh_interval_sec`: 뉴스 갱신 주기(초)
- `news.use_bithumb_notice`: 빗썸 공지 수집
- `news.use_coindesk_rss`: CoinDesk RSS 수집
- `news.use_naver_openapi`: 네이버 OpenAPI 뉴스 수집
- `news.per_source_limit`: 소스별 최대 수집 건수

### 고급 시그널
- `advanced.enabled`: 고급 시그널 게이트 사용 여부
- `advanced.min_total_score`: 진입 최소 점수
- `advanced.support_resistance_weight`: 지지/저항 점수 가중치
- `advanced.pattern_weight`: 캔들 패턴 점수 가중치
- `advanced.orderbook_weight`: 호가 불균형 점수 가중치

### 디스코드 알림
- `notification.enabled`: 디스코드 알림 사용 여부
- `notification.discord_webhook_url`: 웹훅 URL (또는 `.env`의 `DISCORD_WEBHOOK_URL`)
- `notification.notify_on_startup`: 시작 알림
- `notification.notify_on_error`: 예외 알림
- `notification.notify_on_buy`: 매수 알림
- `notification.notify_on_sell`: 매도 알림

## 4. 추천 시작값

초기에는 아래처럼 보수적으로 시작하세요.

- `app.dry_run: true`
- `llm.min_buy_confidence: 0.8`
- `risk.stop_loss_pct: 0.03`
- `trade.max_spread_bps: 25.0`
- `trade.slot_count: 1`
- `news.enabled: true`
- `advanced.enabled: true`
- `notification.enabled: false` (초기엔 비활성 권장)

드라이런 로그를 본 뒤에만 실거래로 전환하는 걸 권장합니다.

## 5. 실거래 전환

`config.yaml`에서 아래 두 값을 변경:

- `app.dry_run: false`
- `app.enable_official_orders: true`

주의: 실거래는 손실 위험이 있으므로 반드시 소액으로 테스트 후 확대하세요.

## 6. 경고/오류 관련

- Gemini SDK는 `google.genai` 기준으로 구성되어 `google.generativeai` deprecate 경고가 나오지 않습니다.
- Python UTC 시간은 timezone-aware(`datetime.now(timezone.utc)`)로 처리되어 `utcnow()` 경고가 나오지 않습니다.
