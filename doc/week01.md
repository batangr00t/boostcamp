# Week 1 - 주간학습정리


## 깨달은 내용
* 텐서란 무엇인가?
* 텐서가 제공하는 method의 특징
* 선형회귀란 무엇인가?
* StandardScaler 활용법
* 시그모이드 함수란?
* BCELoss(Binary Cross Entropy) 함수란?
* CrossEntropyLoss() 함수란?
* Linear 모델의 파라미터 초기화

## 텐서란 무엇인가?
* 다음과 같이 귀납적으로 정의할 수 있음
* 0-D Tensor(Scalr) : 한개의 숫자
* 1-D Tensor(Vector) : 0-D Tensor들이 순서대로 나열된 구조, 결과적으로 벡터 형태
* 2-D Tensor(Matrix) : 1-D Tensor들이 순서대로 나열된 구조, 결과적으로 행과 열로 구성된 매트릭스 형태
* 3-D Tensor         : 2-D Tensor들이 순서대로 나열된 구조, 결과적으로 입체로 구성, (ex) 컬러 이미지
* N-D Tensor         : (N-1)-D Tensor들이 순서대로 나열된 구조, (ex) 동영상