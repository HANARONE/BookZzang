# 04-1.로지스틱 회귀

## 학습 전
> - 로지스틱 회귀에 대해서 제대로 배워보고 싶은 생각이 있었음
> - 암기보다는 제대로 이해하는 것을 중점적으로 생각하여 실습하면서 학습
## 핵심 포인트
> - **로지스틱 회귀** 선형 방정식을 사용한 분류 알고리즘. 선형 회귀와 달리 시그모이드 함수나 소프트맥스 함수를 사용하여 클래스 확률 출력
> - **다중 분류** 타깃 클래스가 2개 이상인 분류 문제
> - **시그모이드 함수** 선형 방정식의 출력을 0과 1 사이의 값으로 압축하여 이진 분류를 위해 사용
> - **소프트맥스 함수** 다중 분류에서 여러 선형 방정식의 출력 결과를 정규화하여 합이 1이 되도록 만듦
## 핵심 패키지와 함수
###  - <u>scikit-learn</u>
> - **LogisticRegression**은 선형 분류 알고리즘인 로지스틱 회귀를 위한 클래스
> - **predict_proba()** 메서드는 예측 확률 반환
>   - 이진 분류의 경우 샘플마다 음성 클래스와 양성 클래스에 대한 확률을 반환
>   - 다중 분류의 경우 샘플마다 모든 클래스에 대한 확률 반환
> - **decision_function()**은 모델이 학습한 선형 방정식의 출력을 반환
>   - 이진 분류의 경우 양성 클래스의 확률이 반환. 이 값이 0보다 크면 양성 클래스, 작거나 같으면 음성 클래스로 예측
>   - 다중 분류의 경우 각 클래스마다 선형 방정식 계산.

 ## 연습문제
> - 2개보다 많은 클래스가 있는 분류 문제는?
>  - 로지스틱 회귀가 이진 분류에서 확률을 출력하기 위해 사용하는 함수는?
>  - decision_function() 메서드의 출력이 0일 때 시그모이드 함수의 값은 얼마?


## 실습코드
> **로지스틱 회귀**
> - 로지스틱 회귀는 이름은 회귀이지만 분류 모델
> - 선형회귀와 동일하게 선형 방정식을 학습
> - 예) z = a x W + b x L + c x D + f
> - z가 아주 큰 음수일 때 0이 되고, z가 아주 큰 양수일 때 1이 되도록 하는 방법?
> - 시그모이드 함수 또는 로지스틱 함수 사용하면 됨

```python
import numpy as np
import matplotlib.pyplot as plt
z = np.arange(-5, 5, 0.1)
phi = 1 / (1 + np.exp(-z))

plt.plot(z, phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()
```
![](https://velog.velcdn.com/images/kwak00912/post/1a21bb15-32e2-4b16-ad0c-6130b339c8c8/image.png)

> ** 로지스틱 회귀로 이진 분류 **

```python
char_arr = np.array(['A', 'B', 'C', 'D', 'E'])
print(char_arr[[True, False, True, False, False]])
## >> ['A' 'C']

bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

print(lr.predict(train_bream_smelt[:5]))
## >> ['Bream' 'Smelt' 'Bream' 'Bream' 'Bream']

# predict_proba() . train_bream_smelt에서 처음 5개 샘플 예측 확률
print(lr.predict_proba(train_bream_smelt[:5]))

print(lr.classes_)

print(lr.coef_, lr.intercept_)

z = -4.04 * (Weight) - 0.567 

decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)

from scipy.special import expit

print(expit(decisions))
```
## 코랩 코드
https://colab.research.google.com/drive/1veESEV2WNcgjKKw4L3IVZCxETFtS21vM?usp=sharing

# 4-2.확률적 경사 하강법
## 핵심 포인트
> - **확률적 경사 하강법**은 훈련 세트에서 샘플 하나씩 꺼내 손실 함수의 경사를 따라 최적의 모델을 찾는 알고리즘
> - **손실 함수**는 확률적 경사 하강법이 최적화할 대상. 대부분의 문제에 잘 맞는 손실 함수가 이미 정의되어 있음.
> - **에포크**는 확률적 경사 하강법에서 전체 샘플을 모두 사용하는 한 번 반복 의미

## 핵심 패키지와 함수
### - scikit-learn
> - **SGDClassifier**는 확률적 경사 하강법을 사용한 분류 모델을 만듦
> - **SGDRegessor**는 확률적 경사 하강법을 사용한 회귀 모델을 만듦


## 코랩 코드
https://colab.research.google.com/drive/1Rw4oJeNmvQy7hlh8Zs_AkzAhQ2tsSG0y?usp=sharing





