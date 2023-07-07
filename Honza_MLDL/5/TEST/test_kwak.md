### Can we use decision trees for multi-class prediction problems?

- yes

### What is the maximum depth parameter of decision tree training?

- n-1, n은 훈련 샘플의 개수

### How can we select a good value for maximum depth parameters?

- 교차 검증

### What is pruning in the decision tree?

- 트리의 최대 깊이를 설정하여 결정 트리의 성장을 제한하는 방법

### Cross Validation은 무엇이고 어떻게 해야하나요?

- 검증세트를 떼어 평가하는 과정을 여러 번 반복하고 이 점수를 평균내어 최종 검증 점수를 얻음. k-폴드 교차검증은 널리 사용되는 교차 검증 방법 중 하나로, 데이터를 k개로 분할 한 뒤, k-1개를 훈련 세트로 1개를 검증 세트로 사용하는데, 이 방법을 k번 반복하여 k개의 성능 지표를 얻어내는 방법

### K-means의 대표적 의미론적 단점은 무엇인가요? (계산량 많다는것 말고)

- 모든 데이터를 거리로 판단하게 되므로, 결과 해석이 어렵다는 단점이 있음.

### 50개의 작은 의사결정 나무는 큰 의사결정 나무보다 괜찮을까요? 왜 그렇게 생각하나요?

- 훈련 세트에는 잘 맞지만, 일반화가 되지 않아 테스트 세트에서는 그 점수에 미치지 못함.
- 설명하기에도 너무 깊으면 어려움이 있음.

### 앙상블 방법엔 어떤 것들이 있나요?

- 랜덤  포레스트, 엑스트라 트리, 그레이디언트 부스팅, 히스토그램 부스팅, XGBoost, LightGBM