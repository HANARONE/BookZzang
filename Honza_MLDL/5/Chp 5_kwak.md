# 5-1.결정 트리

## 코랩
https://gist.github.com/Kwak00912/15ed1a57bd6f999f57198ef60f390591


## 로지스틱 회귀로 와인 분류하기
```python
# 와인 데이터셋 가져오기
import pandas as pd
wine = pd.read_csv('')
```

```python
# 와인데이터 확인
# class는 0이면 레드와인, 1이면 화이트 와인
wine.head()
```

```python
# 보면 총 6,497개 샘플이 있고 4개 열은 모두 실숫값
wine.info()

# describe()를 통해서 간략한 통계값 확인
wine.describe()
```
![](https://velog.velcdn.com/images/kwak00912/post/0bc4a408-789e-4c39-8615-f5abe377beb4/image.png)

> - mean 평균
> - std  표준편차
- min 최소
- 25% 1사분위수
- 50% 중간값
- 75% 3사분위수
- max 최대

```
# 자 표준화 할 차례
data= wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
```

```python
# train_test_split() 함수는 설정값을 지정하지 않으면 25%를 테스트 세트로 지정
# 샘플 개수가 충분히 많으므로 20% 정도만 테스트 세트로 나눴음
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42
)
```

```python
# 만들어진 훈련 세트와 테스트 세트의 크기를 확인
print(train_input.shape, test_input.shape)

>> (5197, 3) (1300, 3)
# 훈련 세트는 5,197개이고 테스트 세트는 1,300개
```

```python
# StandardScaler 클래스를 사용해 훈련 세트를 전처리
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
```

```python
#로지스틱 회귀모델 훈련
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

>> 0.7808350971714451
0.7776923076923077
```
### 설명하기 쉬운 모델과 어려운 모델

```python
print(lr.coef_, lr.intercept_)
>> [[ 0.51270274  1.6733911  -0.68767781]] [1.81777902]

```
> 이 모델은 알코올 도수 값에 0.51270274를 곱하고, 당도에 1.6733911을 곱하고  pH 값에 -0.68767781을 곱한 다음 모두 더합니다. 마지막으로 1.81777902를 더합니다. 이 값이 0보다 크면 화이트 와인, 작으면 레드 와인. 현재 약 77% 정도를 정확히 화인트 와인으로 분류했습니다. 모델의 출력 결과는...

## 결정 트리

>- 결정트리 장점 : 이유를 설명하기 쉽다.
>- 사이킷런이 결정 트리 알고리즘을 제공함
>- 사이킷런의 DecisionTreeClassfier 클래스를 사용해 결정 트리 모델을 훈련할 예정

```python
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))
```
```python
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(10, 7))
plot_tree(dt)
plt.show()
```
![](https://velog.velcdn.com/images/kwak00912/post/21cda96b-0b22-444d-864b-d1ced953cfdd/image.png)

>- 맨 위의 노드를 루트 노드라 하고,
>- 맨 아래 끝에 달린 노드를 리프 노드라고 함
>- 노드는 결정 트리를 구성하는 핵심 요소.

```python
plt.figure(figsize=(10,7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()
```
![](https://velog.velcdn.com/images/kwak00912/post/b65d758f-6b55-42c5-9b9a-346c3fb2b35d/image.png)
### 불순도
>- gini는 지니 불순도를 의미. 
>- DecisionTreeClassifier 클래스의 criterion 매개변수 기본값 'gini'
>- criterion 매개변수의 용도는 노드에서 데이터를 분할할 기준을 정하는 것
>- 지니 불순도는 클래스의 비율을 제곱해서 더한 다음 1에서 빼면 됨
>- 지니 불순도 = 1 - (음성 클래스 비율^2 + 양성 클래스 비율^2)
>- ex) 루트 노드는 총 5,197개 샘플이 있고 그중 1,258개가 음성 클래스, 3,939개가 양성 클래스
### 가지치기
```python
# 결정 트리에서 가지치기를 하는 가장 간단한 방법은 자라날 수 있는 트리의 최대 깊이를 지정하는 것
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))
```
```python
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input, train_target)
print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))
```
```python
plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show
```
![](https://velog.velcdn.com/images/kwak00912/post/57dabb05-9e19-4c04-83a2-2928e5a4351a/image.png)

```python
# 어떤 특성이 가장 유용한지 나타내는 특성 중요도 확인
# 당도가 0.87 정도로 특성 중요도가 가장 높음을 확인할 수 있었음.
print(dt.feature_importances_)
>> [0.12345626 0.86862934 0.0079144 ]
```

## 핵심 포인트
>
- **결정 트리**는 예 / 아니오에 대한 질문을 이어나가면서 정답을 찾아 학습하는 알고리즘
- **불순도**는 결정 트리가 최적의 질문을 찾기 위한 기준
- **정보 이득** 부모 노드와 자식 노드의 불순도 차이. 결정 트리 알고리즘은 정보 이득이 최대화되도록 학습
- **가지치기**는 결정 트리의 성장을 제한하는 방법
- **특성 중요도**는 결정 트리에 사용된 특성이 불순도를 감소하는 데 기여한 정도를 타나내는 값
>

## 핵심 패키지와 함수
### <u> Pandas </u>
>
>- **info()**는 데이터 프레임의 요약된 정보를 출력.
>- **describe()**는 데이터프레임 열의 통계 값 제공. 수치형일 경우 최소, 최대, 평균, 표준편차와 사분위값 등이 출력.
>

### <u> scikit-learn </u>
>
>- **DecisionTreeClassifier**는 결정 트리 분류 클래스.
>- **plot_tree()** 결정 트리 모델을 시각화함. 
>

# 5-2. 교차 검증과 그리드 서치
## 코랩
https://gist.github.com/Kwak00912/c67bc7af46965b8ba076da20ed9e55a0
## 검증 세트

- 테스트 세트를 사용하지 않고, 이를 측정하는 간단한 방법은 훈련 세트를 나누는 것
- 6:2:2 정도로 나눠서 만들면 됨
```python
import pandas as pd
wine = pd.read_csv('https://bit.ly/wine_csv_data')

# class 열을 타깃으로 사용하고 나머지 열은 특성 배열에 저장함
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

# 훈련세트와 테스트 세트를 나눌 차례!
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)

sub_input, val_input, sub_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)
```
```python
print(sub_input.shape, val_input.shape)
>> (4157, 3) (1040, 3)
```
```python
# sub_input, sub_target과 val_input, val_target 사용 모델 평가
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(sub_input, sub_target)
print(dt.score(sub_input, sub_target))
print(dt.score(val_input, val_target))
>> 0.9971133028626413
>> 0.864423076923077
```
## 교차 검증
>
- 교차 검증을 이용하면 안정적인 검증 점수를 얻고 훈련에 더 많은 데이터를 사용할 수 있음
- 3-폴드 교차 검증: 훈련 세트를 세 부분으로 나눠서 교차 검증을 수행하는 것
- 5-폴드 교차 검증이나 10-폴드 교차 검증을 많이 사용함
>
> - 사이킷런에는 cross_validate()라는 교차 검증 함수가 있음. 먼저 평가할 모델 객체를 첫 번째 매개변수로 전달

```python
from sklearn.model_selection import cross_validate
scores = cross_validate(dt, train_input, train_target)
print(scores)
>> {'fit_time': array([0.01113915, 0.01089001, 0.01154613, 0.01579189, 0.01167011]), 'score_time': array([0.0026896 , 0.00195718, 0.00219226, 0.00411606, 0.00201726]), 'test_score': array([0.86923077, 0.84615385, 0.87680462, 0.84889317, 0.83541867])}

```

>- 처음 2개의 키는 각각 모델을 훈련하는 시간과 검증하는 시간을 의미. 

```python
# 교차 검증의 최종 점수는 test_score 키에 담긴 5개의 점수를 평균하여 얻을 수 있음
# 이름은 test_score지만 검증 폴드의 점수. 혼동 하지 않을 것
import numpy as np
print(np.mean(scores['test_score']))
```

>- cross_validate() 함수는 기본적으로 회귀 모델일 경우 KFold 분할기를 사용
>- 분류모델일 경우 타깃 클래스를 골고루 나누기 위해 StratifiedKFold 사용

```python
# 앞서 수행한 교차 검증은 다음 코드와 동일함
from sklearn.model_selection import StratifiedKFold
scores = cross_validate(dt, train_input, train_target, cv=StratifiedKFold())
print(np.mean(scores['test_score']))
 >> 0.855300214703487
```
```python
# 만약 훈련 세트를 섞은 후 10 -폴드 교차 검증 수행한다면
splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, train_input, train_target, cv=splitter)
print(np.mean(scores['test_score']))
 >> 0.8574181117533719
```
## 하이퍼파라미터 튜닝

```python
from sklearn.model_selection import GridSearchCV

params = {'min_impurity_decrease' : [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
```
```python
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)

```
```python
gs.fit(train_input, train_target)
```
![](https://velog.velcdn.com/images/kwak00912/post/f7529f18-234f-47f3-ab80-932ac78fceb3/image.png)
```python
dt = gs.best_estimator_
print(dt.score(train_input, train_target))
 >> 0.9615162593804117
```
```python
# 여기서는 0.0001이 가장 좋은 값으로 선택됨
print(gs.best_params_)
 >> {'min_impurity_decrease': 0.0001}
```
```python
print(gs.cv_results_['mean_test_score'])
 >> [0.86819297 0.86453617 0.86492226 0.86780891 0.86761605]
```
```python
best_index = np.argmax(gs.cv_results_['mean_test_score'])
print(gs.cv_results_['params'][best_index])
 >> {'min_impurity_decrease': 0.0001}
```

>- 이 과정을 정리하면
>>- 1. 먼저 탐색할 매개변수를 지정
>- 2. 다음 훈련 세트에서 그리드 서치를 수행하여 최상의 평균 검증 점수가 나오는 매개변수 조합을 찾음.
>- 3. 그리드 서치는 최상의 매개변수에서 전체 훈련 세트를 사용해 최종 모델을 훈련함

```python
# min_impurity_decrease
# 노드를 분할하기 위한 불순도 감소 최소량을 지정함
params = {'min_impurity_decrease' : np.arange(0.0001, 0.001, 0.0001),
          # 첫 번째 매개변수 값에서 시작하여 두 번째 매개변수에 도달할 때까지 세 번째 매개변수를 계속 더한 배열 생성
          'max_depth': range(5, 20, 1),
          # max_depth를 5에서 20까지 1씩 증가하면서 15개의 값을 만듬
          'min_samples_split' : range(2, 100, 10)
          # 2에서 100까지 10씩 증가하면서 10개 값을 만듬
          }

# 이렇게 하면 교차 검증 횟수는 9 X 15 X 10 = 1,350개
# 기본 5-폴드 교차 검증을 수행하므로 만들어지는 모델 수는 6,750개
```
```python
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
```
![](https://velog.velcdn.com/images/kwak00912/post/545925a0-841d-48b0-9e32-6a04540443f8/image.png)
```python
# 최상의 매개변수 조합 확인
print(gs.best_params_)
```
```python
# 최상의 교차 검증 점수 확인
print(np.max(gs.cv_results_['mean_test_score']))
```
## 랜덤 서치

```python
# 매개변수의 값이 수치일 때 값의 범위나 간격을 미리 정하기 어려울 수 있음
from scipy.stats import uniform, randint
```

```python
rgen = randint(0, 10)
rgen.rvs(10)
 >> array([0, 6, 9, 5, 0, 1, 7, 4, 8, 4])
```
```python
np.unique(rgen.rvs(1000), return_counts=True)
 >> (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
 array([105, 100,  92, 117, 104,  94,  93,  97, 110,  88]))
```

```python
ugen = uniform(0, 1)
ugen.rvs(10)
 >> array([0.9321647 , 0.32088138, 0.94530902, 0.48431893, 0.52330475,
       0.24772438, 0.77390882, 0.48520097, 0.35190224, 0.14109198])
```

```python
# min_imputiry_decrease 0.0001에서 0.001 사이의 실숫값을 샘플링
# max_depth는 20에서 50 사이의 정수
# min_samples_split은 2에서 25 사이의 정수
# min_samples_leaf는 1에서 25 사이의 정수

params = {'min_impurity_decrease': uniform(0.0001, 0.001),
          'max_depth': randint(20, 50),
          'min_samples_split': randint(2, 25),
          'min_samples_leaf': randint(1, 25),
          }

```
```python
from sklearn.model_selection import RandomizedSearchCV
gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, n_iter=100, n_jobs=-1, random_state= 42)
gs.fit(train_input, train_target)
```
![](https://velog.velcdn.com/images/kwak00912/post/be8267a0-6231-42ed-aa35-f91fb9b6280b/image.png)

```python
print(gs.best_params_)
 >> {'max_depth': 39, 'min_impurity_decrease': 0.00034102546602601173, 'min_samples_leaf': 7, 'min_samples_split': 13}
```

```python
print(np.max(gs.cv_results_['mean_test_score']))
 >> 0.8695428296438884
```

```python
# 최적의 모델은 이미 전체 훈련 세트(train_input, train_target)로 훈련되어 best_estimator 속성에 저장
dt = gs.best_estimator_
print(dt.score(test_input, test_target))
 >> 0.86
```

## 핵심 포인트
> - **검증 세트**는 하이퍼파라미터 튜닝을 위해 모델 평가할 때, 테스트 세트를 사용하지 않기 위해 훈련 세트에서 다시 떼어 낸 데이터 세트
> - **교차 검증**은 훈련 세트를 여러 폴드로 나눈 다음 한 폴드가 검증 세트의 역할을 하고 나머지 폴드에서는 모델을 훈련. 교차 검증은 이런 식으로 모든 폴드에 대해 검증 점수를 얻음
> - **그리드 서치**는 하이퍼파라미터 탐색을 자동화해 주는 도구. 탐색할 매개변수를 나열하면 교차 검증을 수행하여 가장 좋은 검증 점수의 매개변수 조합을 선택
> - **랜덤 서치**는 연속된 매개변수 값을 탐색할 때 유용. 탐색할 값을 직접 나열하는 것이 아니고 탐색 값을 샘플링할 수 있는 확률 분포 객체 전달

## 핵심 패키지와 함수
### scikit-learn
> - **cross_validate()**는 교차 검증을 수행하는 함수
> - **GridSearchCV**는 교차 검증으로 하이퍼파라미터 탐색을 수행
> - **RandomizedSearchCV**는 교차 검증으로 랜덤한 하이퍼파라미터 탐색 수행

## 추가 자료



# 5-3.트리의 앙상블
## 코랩
https://gist.github.com/Kwak00912/67e3a5c8ba955953c8aeed1f9a63c389
## 서론

>- 정형 데이터를 다루는 데 가장 뛰어난 성과를 내는 알고리즘이 앙상블 학습
>- 랜덤 포레스트는 앙상블 학습의 대표 주자 중 하나로 안정적인 성능 덕분에 널리 사용되고 있음


## 트리의 앙상블
## 랜덤포레스트

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
wine = pd.read_csv('https://bit.ly/wine_csv_data')
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)
```

```python
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
 >> 0.9973541965122431 0.8905151032797809
```

```python
rf.fit(train_input, train_target)
print(rf.feature_importances_)
 >> 0.9973541965122431 0.8905151032797809
```

```python
rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state= 42)
rf.fit(train_input, train_target)
print(rf.oob_score_)
 >> 0.8934000384837406
```

## 엑스트라트리

```python
from sklearn.ensemble import ExtraTreesClassifier

et = ExtraTreesClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(et, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
 >> 0.9974503966084433 0.8887848893166506
```
```python
et.fit(train_input, train_target)
print(et.feature_importances_)
 >> [0.20183568 0.52242907 0.27573525]
```
## 그레이디언트 부스팅

>- 그레이디언트 부스팅은 깊이가 얕은 결정 트리를 사용하여 이전 트리의 오차를 보완하는 방식으로 앙상블하는 방법
>- 사이킷런의 GradientBoostingClassifier는 기본적으로 깊이가 3인 결정 트리를 100개 사용

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(random_state=42)
scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
 >> 0.8881086892152563 0.8720430147331015
```

```python
# 학습률을 증가시키고 트리의 개수를 늘리면 성능이 향상될 수 있음
# 학습률의 기본값은 0.1임
gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.2, random_state=42)
scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
 >> 0.9464595437171814 0.8780082549788999
```

```python
# 그레이디언트 부스팅이 랜덤 포레스트보다 일부 특성(당도)에 더 집중
gb.fit(train_input, train_target)
print(gb.feature_importances_)
 >> [0.15872278 0.68010884 0.16116839]
```

## 히스토그램 기반 부스팅

>- **히스토그램 기반 그레이디언트 부스팅** 정형 데이터를 다루는 머신러닝 알고리즘 중 가장 인기가 높은 알고리즘
>- 입력 특성을 256개 구간으로 나눔


```python
from sklearn.ensemble import HistGradientBoostingClassifier
hgb = HistGradientBoostingClassifier(random_state=42)
scores = cross_validate(hgb, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# 과대 적합을 작 어제하면서 그레이디언트 부스팅보다 조금 더 높은 성능 제공
 >> 0.9321723946453317 0.8801241948619236
```

```python
from sklearn.inspection import permutation_importance

hgb.fit(train_input, train_target)
result = permutation_importance(hgb, train_input, train_target, n_repeats=10,
                                random_state=42, n_jobs=-1)
print(result.importances_mean)
 >> [0.08876275 0.23438522 0.08027708]
```

```python
result = permutation_importance(hgb, test_input, test_target, n_repeats=10,
                                random_state=42, n_jobs=-1)
print(result.importances_mean)
 >> [0.05969231 0.20238462 0.049     ]
```

```python
# 테스트 세트에서 성능 최종적으로 확인
hgb.score(test_input, test_target)
 >> 0.8723076923076923
```

#### XGBoost

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(tree_method='hist', random_state=42)
scores = cross_validate(xgb, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
 >> 0.9555033709953124 0.8799326275264677
```

#### LightGBM

```python
# 마이크로소프트에서 만든 lightgbm
# 빠르고 최신 기술을 많이 적용하고 있음
from lightgbm import LGBMClassifier

lgb = LGBMClassifier(random_state=42)
scores = cross_validate(lgb, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
 >> 0.935828414851749 0.8801251203079884
```

## 핵심 포인트
> -**앙상블 학습**은 더 좋은 예측 결과를 만들기 위해 여러 개의 모델을 훈련하는 머신러닝 알고리즘
> -**랜덤 포레스트**는 대표적인 결정 트리 기반의 앙상블 학습 방법
> -**엑스트라 트리**는 랜덤 포레스트와 비슷하게 결정 트리를 사용하여 앙상블 모델을 만들지만, 부트스트랩 샘플을 사용 안 함. 단 랜덤하게 노드를 분할해 과대 적합을 감소시킴.
> -**그레이디언트 부스팅**은 랜덤 포레스트나 엑스트라 트리와 달리 결정 트리를 연속적으로 추가하여 손실 함수를 최소화하는 앙상블 방법

## 핵심 패키지와 함수
### <u>scikit-learn</u>
> - **RandomForestClassifier**: 랜덤 포레스트 분류 클래스
> - **ExtraTreesClassifier**: 엑스트라 트리 분류 클래스
> - **GraidientBoostingClassifier**: 그레이디언트 부스팅 분류 클래스
