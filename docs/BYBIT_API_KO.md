# Bybit API 분석 요약 (V5)

이 문서는 현재 코드에서 사용하는 Bybit V5 API를 정리합니다.

## 1. 인증

- 사설 API는 `X-BAPI-API-KEY`, `X-BAPI-TIMESTAMP`, `X-BAPI-RECV-WINDOW`, `X-BAPI-SIGN` 헤더 사용
- 서명 원문:
  - GET: `timestamp + api_key + recv_window + query_string`
  - POST: `timestamp + api_key + recv_window + compact_json_body`
- 해시: HMAC-SHA256

코드: `/Users/mosiwon/dev/bithumb_tradbot/core/bybit_exchange.py`

## 2. 공용 API (시세/시장)

- `GET /v5/market/instruments-info`
  - 심볼 메타(수량 스텝, 최소수량 등)
- `GET /v5/market/tickers`
  - 현재가, 24h 거래대금
- `GET /v5/market/kline`
  - 캔들 데이터
- `GET /v5/market/orderbook`
  - 호가창
- `GET /v5/announcements/index`
  - 거래소 공지

## 3. 사설 API (잔고/주문/포지션)

- `GET /v5/account/wallet-balance`
  - 가용 잔고, 총자산
- `GET /v5/position/list`
  - 오픈 포지션 조회 (재시작 동기화)
- `POST /v5/position/set-leverage`
  - 레버리지 설정
- `POST /v5/order/create`
  - 시장가 진입/청산

## 4. 주문 안정성 규칙

코드 반영 사항:

1. 심볼별 `qtyStep`, `minOrderQty`, `maxOrderQty`, `minNotionalValue` 조회 후 캐시
2. 주문 수량은 step 기준 내림(floor)
3. 최소수량/최소노셔널 미달 시 주문 차단
4. `position_idx` 설정으로 원웨이/헤지모드 대응

## 5. 실제 운영 시 체크 포인트

- API 키 권한: `Contract Trade` 활성화
- 계정 포지션 모드와 `bybit.position_idx` 일치
- `trade.min_order_notional`이 거래 심볼 최소 주문금액보다 작지 않게
- `dry_run=true`로 먼저 시그널/주문 파라미터 검증 후 실거래 전환
