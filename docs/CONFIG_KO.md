# 설정 가이드 (Bybit, 한국어)

이 프로젝트는 **Bybit 선물(Perp) 전용** 자동매매 봇입니다.

- 설정 파일: `/Users/mosiwon/dev/bithumb_tradbot/config.yaml`
- 비밀키 파일: `/Users/mosiwon/dev/bithumb_tradbot/.env`
- 24시간 서버 운영: `/Users/mosiwon/dev/bithumb_tradbot/docs/ORACLE_24H_KO.md`
- Bybit API 맵: `/Users/mosiwon/dev/bithumb_tradbot/docs/BYBIT_API_KO.md`

## 1. 필수 준비

`.env` 예시:

```env
BYBIT_API_KEY=...
BYBIT_API_SECRET=...
OPENAI_API_KEY=...
DISCORD_WEBHOOK_URL=...
```

`Gemini`를 쓰고 싶으면:
- `llm.provider: gemini`
- `.env`에 `GEMINI_API_KEY=...`

## 2. 실행 방법

```bash
# 계좌/API 체크
/Users/mosiwon/dev/bithumb_tradbot/.venv/bin/python /Users/mosiwon/dev/bithumb_tradbot/main.py --check-account

# 데이터/지표 품질 체크
/Users/mosiwon/dev/bithumb_tradbot/.venv/bin/python /Users/mosiwon/dev/bithumb_tradbot/main.py --validate-data

# 1회 루프 실행
/Users/mosiwon/dev/bithumb_tradbot/.venv/bin/python /Users/mosiwon/dev/bithumb_tradbot/main.py --run-once

# 디스코드 테스트
/Users/mosiwon/dev/bithumb_tradbot/.venv/bin/python /Users/mosiwon/dev/bithumb_tradbot/main.py --test-notify

# 성능 기준점 초기화
/Users/mosiwon/dev/bithumb_tradbot/.venv/bin/python /Users/mosiwon/dev/bithumb_tradbot/main.py --reset-performance

# 상시 실행
/Users/mosiwon/dev/bithumb_tradbot/.venv/bin/python /Users/mosiwon/dev/bithumb_tradbot/main.py
```

## 3. 핵심 파라미터

### 운영
- `app.dry_run`: `true`면 주문 없이 시그널만 검증
- `app.interval_sec`: 루프 주기(초)
- `app.max_consecutive_errors`: 연속 오류 허용치
- `app.log_api_usage`: API 사용량 로그
- `app.log_performance`: 시작 대비 손익 로그

### Bybit
- `bybit.base_url`: 기본 `https://api.bybit.com`
- `bybit.category`: `linear` 권장
- `bybit.quote_coin`: 보통 `USDT`
- `bybit.account_type`: 보통 `UNIFIED`
- `bybit.position_idx`: 원웨이 `0`, 헤지모드 롱 `1`/숏 `2`
- `bybit.leverage`: 기본 레버리지
- `bybit.allow_long`: 롱 허용
- `bybit.allow_short`: 숏 허용
- `bybit.short_min_advanced_score`: 숏 고급시그널 최소 절대값

### 자금/체결
- `trade.seed_capital`: 총 운용 시드(quote 코인 기준)
- `trade.slot_count`: 동시 포지션 슬롯 개수
- `trade.min_order_notional`: 최소 주문금액(quote 코인)
- `trade.max_spread_bps`: 허용 스프레드
- `trade.use_available_balance_as_seed`: 가용잔고 기반 자동 분배
- `trade.order_retry_count`: 주문 재시도 횟수
- `trade.cancel_unfilled_before_retry`: 미체결 취소 후 재시도

### 리스크
- `risk.stop_loss_pct`: 손절 (예: `0.03` = -3%)
- `risk.trailing_start_pct`: 트레일링 시작 수익구간
- `risk.trailing_gap_pct`: 트레일링 폭

### 전략/LLM
- `llm.provider`: `openai` 또는 `gemini`
- `strategy.required_signal_count`: 하드룰 최소 충족 개수
- `strategy.rsi_buy_threshold`, `strategy.rsi_sell_threshold`
- `strategy.use_macd_golden_cross`, `strategy.use_macd_dead_cross`
- `llm.model_name`: 예) `gpt-4o-mini`
- `llm.openai_base_url`: 기본 `https://api.openai.com/v1`
- `llm.openai_timeout_sec`: OpenAI 요청 타임아웃
- `llm.min_buy_confidence`: 롱 LLM 신뢰도 하한
- `llm.min_sell_confidence`: 숏 LLM 신뢰도 하한
- `llm.max_dead_cat_risk`: 데드캣 리스크 상한

### 뉴스/RAG
- `news.use_exchange_notice`: 거래소 공지 수집(Bybit)
- `news.use_coindesk_rss`: CoinDesk RSS 수집
- `news.use_naver_openapi`: 네이버 뉴스 API 수집

### 알림
- `notification.enabled`: Discord 알림 on/off
- `notification.discord_webhook_url`: 웹훅 URL
- `notification.notify_on_buy`, `notification.notify_on_sell`
- 시작/청산 알림에는 누적 손익도 함께 표시

## 4. 안전한 시작 순서

1. `app.dry_run: true`로 최소 1~2일 로그 관찰
2. `--check-account`, `--run-once` 모두 정상 확인
3. `trade.seed_capital` 소액으로 시작
4. 그 다음 `app.dry_run: false` 전환

## 5. 전략 튜닝 팁

- 거래가 너무 적으면:
  - `strategy.required_signal_count`를 낮추기
  - `llm.min_buy_confidence`, `llm.min_sell_confidence`를 소폭 완화
- 거래가 너무 많으면:
  - `advanced.min_total_score` 상향
  - `trade.max_spread_bps` 하향
  - `risk.stop_loss_pct` 보수적으로 재설정

## 6. 참고

- 본 봇은 현물 빗썸용이 아니라 **Bybit 선물 롱/숏** 구조입니다.
- Python UTC 처리는 timezone-aware로 구성되어 `utcnow()` 경고가 없습니다.
