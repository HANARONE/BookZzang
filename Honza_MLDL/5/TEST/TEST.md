# chp.01-05 TEST (2023.07.07)

> 9:00 - 9:30 테스트 후 영지에게 제출



### Can we use decision trees for multi-class prediction problems?

- answer 네





### What is the maximum depth parameter of decision tree training?

- answer 리프노드 





### How can we select a good value for maximum depth parameters?

- answer 교차검증





### What is pruning in the decision tree?

- answer 가지가 계속해서 이어져서 훈련세트의 과대적합이 일어나는걸
- 막기위해서 특정 MAX_depth의 매개변수를 지정해서 가지가 내가정해준 만큼만 이어지게 해서 테스트세트의 설명력을 높여준다.







### Cross Validation은 무엇이고 어떻게 해야하나요?

- answer 교차검증은 훈련세트를 여러 폴드로 나눈 다음 한 폴드가 검증 세트의 역할을 하고 나머지 폴드에서는 훈련을 하는 방식이다. 교차검증은 위와 같은 방법으로 모든 폴드에 대해 검증점수를 얻어 평균하는 방법이다.





### K-means의 대표적 의미론적 단점은 무엇인가요? (계산량 많다는것 말고)

- answer  군집의 개수를 사용자가 결정해야한다 또한, 처음에 초기 군집의 중심점을 랜덤으로 고르기 때문에 알고리즘을 수행할 때마다 다른 결과가 나온다.







### 50개의 작은 의사결정 나무는 큰 의사결정 나무보다 괜찮을까요? 왜 그렇게 생각하나요?

- answer 데이터의 규모가 작고 별로 없다면 50개의 작은 의사결정 나무가 큰 의사결정나무보다 좋겠지만 그반대라면 큰의사 결정 나무가 더 좋다. 왜냐하면 데이터가 희소할때는 적은가지로도 분류가 잘 될수있지만 데이터가 많으면 적은가지로는 분류하는데 어려움이 있을것이기 때문이다.









### 앙상블 방법엔 어떤 것들이 있나요?

- answer 랜덤 포레스트, 엑스트라 트리, 그레이디언트부스팅, 히스토그램 기반 그레이디언트 부스팅,
- XGboost, LightGBM











-----

[복습 & 질문 리스트]

- 불순도 간 차이 : 측정방법, 쓰임에는 차이가 없다
- 지니불순도 : 일반적으로 0.5 이하