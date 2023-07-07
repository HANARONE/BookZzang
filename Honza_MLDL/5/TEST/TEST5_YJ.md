### Can we use decision trees for multi-class prediction problems?

- 멀티클래스 예측 가능, 바이너리도 가능





### What is the maximum depth parameter of decision tree training?

- 훈련 세트 개수 -1 까지 가능한데 과적합때매 거기까지 잘 안감





### How can we select a good value for maximum depth parameters?

- GridSearch를 통해서





### What is pruning in the decision tree?

- 가지치기, 의사결정 나무 성장 제한, 과적합 방지







### Cross Validation은 무엇이고 어떻게 해야하나요?

- 교차검증, train set과 test set 분류, train set 에서 validation set을 한번 더 분류하여 검증, 이를 통해 test set에 대한 과적합 방지



### K-means의 대표적 의미론적 단점은 무엇인가요? (계산량 많다는것 말고)

- 처음 중심점을 랜덤으로 결정하기 때문에 분류가 명확하지 않을 수 있음





### 50개의 작은 의사결정 나무는 큰 의사결정 나무보다 괜찮을까요? 왜 그렇게 생각하나요?

- 경우에 따라 더 좋음, 비선형성이 강한 문제를 해결할때 모든 특성을 학습하는 큰 의사결정나무를 사용하는것보다 특성 일부들을 사용하여 학습한 여러개의 작은 의사결정나무가 과적합 방지에도 효과적이고 데이터 노이즈에도 강함
앙상블 방법엔 어떤 것들이 있나요?



### 앙상블 방법엔 어떤 것들이 있나요?

- 랜덤포레스트, 엑스트라트리, 그레디언트 부스팅, 히스토그램 부스팅, XG부스트, LightGBM, ADA부스트 등등