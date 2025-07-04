# vrp

배송 프로세스에 영향을 미치는 주요 요소는 크게 라우팅과 하차 작업으로 나누어 지게 됩니다. 라우팅 비용 절감을 위해서는 주어진 시간에 사용하는 트럭수, 이동 거리에 따른 유류비 등이 영향을 미칩니다. 하차 작업의 시간을 절약하기 위해서는 배송 순서에 따라 상품이 하차할 수 있도록 트럭의 도어에 가깝게 있어야 하며, 하차해야하는 상품을 꺼내기 위하여 다른 상품들을 꺼냇다 넣었다 하는 작업이 적어야 합니다. 라우팅 비용을 절감하고 하차 작업 시간을 절약하기 위하여 두가지 문제를 융합하여 주어진 시간에 최소 비용으로 배송을 완료할 수 있는 라우팅과 적재 알고리즘을 구현하는 팀이 우승하게 됩니다.

물류에서 배송 비용 절감을 위하여 착지 정보를 기반으로 라우팅을 최적화 하는 방법을 개발하여 적용하고 있으나, 상품을 하차하는 순서까지는 고려를 하고 있지 않았습니다. 이번 과제에서는 라우팅 최적화에 덧붙여 각 착지별로 상품을 하차하는 순서와 물량을 고려하여 적재 순서를 최적화 하는 새로운 방법론을 구현하는 것을 목적으로 합니다. 이는 배송프로세스의 비용 절감에 큰 도움이 되며, 배송을 담당하는 담당자의 육체적 노동량을 절감시킬 것으로 기대됩니다.



# 문제 설명

## 개발용 제공 데이터셋

- 상품 정보
-- 박스 3종 체적 정보 (30x40x30, 30x50x40, 50x60x50cm)

- 1일치 주문 데이터
-- 발착지 위/경도
--- Depot: 물류 센터 (상품들을 상차하는 장소)
--- Destinations: 주문에 따른 착지 번호별 정보
-- Orders
--- Order number: 주문 번호
--- Box ID: 상품ID
--- Destination: 착지 번호
--- Dimension: 상품 넓이, 길이, 높이 정보

* 개발용 제공 데이터셋은 300착지, 1~2주문수로 총 437개 주문 데이터 제공
* 평가용 데이터셋은 300이상의 착지수로 약 1,000개 이상의 주문 데이터로 평가 예정

- OD matrix
-- 경로 산출을 위한 착지 및 센터 간의 Origin-Destination matrix (Integer)

- Output 양식 예시 파일
-- 제출한 소스코드의 Output 파일 규정
-- Output 파일 이름: Result.xlsx

## 차량 정보
- 배송 차량 (1톤)
-- 사용가능한 차량 수 : 무제한
-- 최대 적재 부피 : 160x280x180cm (넓이X/깊이Y/높이Z)
-- 기본 단가 (고정비용) : 15만원
-- 거리당 유류대 : 500원/km
-- 적재/하차 방식: 후입선출
-- 차량 적재함 좌표계: Right-handed coordinate system
--- Origin: 차량 적재함 가장 안쪽의 좌측
--- 입출 Plane (Door): XZ plane (Y=280cm)

## Ground Rule
* 1대의 차량이 다회전 할 수 없음
* 각 착지는 차량 한대가 한번만 방문이 가능함

## 평가 기준

제출한 라우팅과 적재 최적화가 융합된 소스코드와 소스코드에서 나오는 Output 파일의 결과물로 평가합니다.

라우팅 비용 = 고정비 + 유류비
- 고정비 : 차량 1대 사용하는데에 따른 고정비용, 150,000원
- 유류비 : 거리에 따른 유류비용, 500원/km

하차 비용 = 셔플링 횟수 x 셔플링 비용
- 셔플링 횟수 : 상품을 꺼내기 위하여 이동해야하는 주변 상품의 수
- 셔플링 비용 : 500원/셔플링

Total Score = 라우팅 비용 + 하차 비용 (Total Score가 작은 팀이 우승)

시간제한 = 20분
- 소스코드는 시간제한 안에 결과를 도출해야 함

※ 실제 평가에서는 제공된 데이터셋보다 더 큰 규모의 데이터로 평가가 진행되므로, 실행 시간 제한(Timeout)에 유의해야 함.
제공된 예시 데이터셋 기준으로 5분 이내에 결과가 출력되도록 구현하는 것을 권장.

## 참고파일
github repo 내 problem-docs 폴더 하위 파일 참고.
