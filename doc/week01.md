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
    * 0-D Tensor(Scalar) : 한개의 숫자
    * 1-D Tensor(Vector) : 0-D Tensor들이 순서대로 나열된 배열, 벡터 형태
    * 2-D Tensor(Matrix) : 1-D Tensor들이 순서대로 나열된 배열, 행과 열로 구성된 매트릭스 형태
    * 3-D Tensor         : 2-D Tensor들이 순서대로 나열된 배열, 입체로 구성, (ex) 컬러 이미지
    * N-D Tensor         : (N-1)-D Tensor들이 순서대로 나열된 배열, (ex) 동영상
* 텐서는 다차원 데이터 뿐만 아니라 다양한 메타정보를 가지고 있음
    * 차원의 갯수 : ndim, dim()
    * 각 차원의 크기 : shape, size()
    * 데이타 타입 : dtype
    * 텐서가 저장된 디바이스 : cpu, cuda, mps, xpu, ...
    * gradient 계산 여부 : requires_grad
    * 저장된 gradient : grad
    * gradient 계산을 위한 함수 : grad_fn
* 예시
    ```python
        import torch

        # 0-D tensor
        scalar = torch.tensor(3)   
        print(scalar)       # tensor(3)
        print(scalar.shape) # torch.Size([])
        print(scalar.ndim)  # 0
        print(scalar.dtype) # torch.int64

        scalar2 = torch.randn(10).sum()
        print(scalar2)       # tensor(-0.6778)
        print(scalar2.shape) # torch.Size([])
        print(scalar2.ndim)  # 0
        print(scalar2.dtype) # torch.float32

        # 1-D tensor
        vector1 = torch.tensor([1,2,3]) # 0-D tensor가 1,2,3 인 배열
        print(vector1)       # tensor([1, 2, 3])
        print(vector1.shape) # torch.Size([3])
        print(vector1.ndim)  # 1
        print(vector1.dtype) # torch.int64

        vector2 = torch.randint(5, [1]) # 0-D tensor가 1개인 배열
        print(vector2)       # tensor([3]), 0-D 텐서인 tensor(3)과 다름
        print(vector2.shape) # torch.Size([1])
        print(vector2.ndim)  # 1
        print(vector2.dtype) # torch.int64

        # 2-D tensor
        matrix1 = torch.zeros(3, 4) # 1-D tensor가 3개인 배열
        print(matrix1)
        # tensor([[0., 0., 0., 0.],
        #         [0., 0., 0., 0.],
        #         [0., 0., 0., 0.]])
        print(matrix1.size()) # torch.Size([3, 4])
        print(matrix1.dim())  # 2
        print(matrix1.dtype)  # torch.float32

        matrix2 = torch.eye(4, dtype=torch.double)  # 1-D tensor가 4개인 배열
        print(matrix2)
        # tensor([[1., 0., 0., 0.],
        #         [0., 1., 0., 0.],
        #         [0., 0., 1., 0.],
        #         [0., 0., 0., 1.]], dtype=torch.float64)
        print(matrix2.size()) # torch.Size([4, 4])
        print(matrix2.dim())  # 2
        print(matrix2.dtype)  # torch.float64
    ```

## 텐서가 제공하는 method의 특징
* 텐서 생성 method
    * zeros(), ones(), eye(), full(), rand(), randn(), randint()
    * shape 을 파라미터로 입력
        * 0-D : [], ()
        * 1-D : [1], (30), [1,], (30,)
        * 2-D : [3,4], (3,4)
        * 3-D : [3, 28, 28], (3, 28, 28)
    * *_like() method
        * shape, dtype, device : 상속(O)
        * requires_grad : 상속(X)
        * empty_like, zeros_like, ones_like, full_like, rand_like, randn_like
        * 같은 크기의 메모리를 새로 할당함
* shape 변경 method
    * reshape() : shape 변경, 메모리 공유/복사
    * view() : shape만 변하고, 메모리 공유
    * squeeze() : 크기가 1인 차원 제거, 메모리 공유/복사
    * unsqueeze() : 크기 1인 차원 추가, 메모리 공유
    * transpose(), t(), .T : 주어진 차원을 교환, 메모리 공유
    * permute() : 차원의 순서 재배열, 메모리 공유
    * flatten() : 1차원으로 평면화, 메모리 공유/복사
    * expand() : 동일 데이터로 차원을 확장해서 보여줌, 메모리 공유
    * repeat() : 동일 데이터로 차원을 확장해서 보여줌, 메모리 복사
    * stack() : 동일 size의 텐서의 배열로 새로운 차원을 추가함, 
    * cat() : 지정한 차원의 크기를 증가시킴
* 텐서의 대부분의 method는 텐서를 반환 ( 일부 함수는 그렇지 않음 )
    * 텐서끼리의 연산 : tensor 를 반환
        * add(), sub(), mul(), div(), +, -, *, /
        * matmul(), mm(), @
    * 수학함수 method : tensor 를 반환
        * 공용함수, 삼각함수, 지수함수, 로그함수, 비교함수, 논리함수, 통계함수, 선형대수 함수 등
        * abs(), sin(), exp(), log(), gt(), logical_and(), mean(), norm()
        * tensor(3), tensor([ True,  True,  True]) 같이 스칼라나 논리값도 텐서를 반환
    * 텐서가 아닌 값으로 출력하는 경우
        * item() : scalar tensor의 값을 Python 기본 타입으로 반환 
        * numel() : 텐서의 모든 요소 개수를 Python 기본 타입으로 반환
        * equal() : 전체 동등성 검사는 Python bool로 반환
        * size() : torch.Size([2, 2]) - 특별한 튜플 타입
        * 검사 함수 : is_* 형태의 검사 함수는 Python bool로 반환
            * is_floating_point()
            * is_complex()
            * is_same_size()
            * is_contiguous()

* dim 의미
    * 텐서는 ndim 차원의 수만큼의 dim을 갖음
    * 가장 외각 배열의 차원을 dim-0으로 하고 가장 마지막 단계의 차원을 dim-(ndim-1)로 칭함
    * 가장 마지막 차원을 -1로도 칭함, 그 다음은 -2, ... 가장 외각 배열의 차원은 -ndim 으로도 칭함
    * [3, 4, 5] shape의 tensor 의 경우
        * 0 차원, -3 차원 : 가장 외각 배열, 사이즈 3
        * 1 차원, -2 차원 : 중간 안쪽 배열, 사이즈 4
        * 2 차원, -1 차원 : 가장 안쪽 배열, 사이즈 5
    * 예시
        ```python
        a = torch.randint(10, (2,3,4))
        print(a)
        # tensor([[[9, 7, 3, 6],
        #          [5, 4, 3, 3],
        #          [7, 8, 5, 9]],
        #
        #         [[3, 4, 2, 7],
        #          [5, 6, 8, 5],
        #          [8, 1, 8, 6]]])
        
        # 전체 element 더하여 스칼라 텐서 생성됨, 결과는 0-D tensor
        a.sum()
        # tensor(132)
        
        # dim-0차원 2개의 배열을 더함, 더한 결과가 스칼라 탠서임으로 차원이 사라짐 => 결과는 (3,4) shape의 탠서
        a.sum(dim=0)
        # tensor([[12, 11,  5, 13],
        #         [10, 10, 11,  8],
        #         [15,  9, 13, 15]])

        # dim-1차원 3개의 배열을 더함, 더한 결과가 스칼라 탠서임으로 차원이 사라짐 => 결과는 (2,4) shape의 탠서
        a.sum(dim=1)
        #tensor([[21, 19, 11, 18],
        #        [16, 11, 18, 18]])

        # dim-2차원 4개의 배열을 더함, 더한 결과가 스칼라 탠서임으로 차원이 사라짐 => 결과는 (2,3) shape의 탠서
        a.sum(dim=2)
        # tensor([[25, 15, 29],
        #         [16, 24, 23]])

        a.sum(dim=-1)
        # tensor([[25, 15, 29],
        #         [16, 24, 23]])

        a.sum(dim=-2)
        # tensor([[21, 19, 11, 18],
        #         [16, 11, 18, 18]])

        a.sum(dim=-3)
        # tensor([[12, 11,  5, 13],
        #        [10, 10, 11,  8],
        #        [15,  9, 13, 15]])
                
        ```

## 선형회귀란 무엇인가?
* 독립변수 x와 종속변수 y의 선형관계를 모델링 하고, 이 모델을 통해 임의의 x로부터 y를 예측 하는데 활용
* 선형회귀의 종류
    * 단순선형회귀(Simple Linear Regression)
        * 독립 변수 x가 한개 => size가 1개인 1-D 텐서 (사이즈가 1인 Vector)
        * 종속 변수 y가 한개 => size가 1개인 1-D 텐서 (사이즈가 1인 Vector)
        * y는 x, b의 선형결합이라고 가정
    * 다중선형회귀(Multiple Linear Regression)
        * 독립 변수 x가 2개 이상 => size가 2 이상인 1-D 텐서 (사이즈가 2이상인 Vector)
        * 종속 변수 y가 한개 => size가 1개인 1-D 텐서 (사이즈가 1인 Vector)
        * y는 x의 각 원소들과 b의 선형결합이라고 가정
    * 다변량다중선형회귀(Multivariate Multiple Linear Regression)
        * 독립 변수 x가 2개 이상 => size가 2 이상인 1-D 텐서 (사이즈가 2이상인 Vector)
        * 종속 변수 y가 2개 이상 => size가 2 이상인 1-D 텐서 (사이즈가 2이상인 Vector)
        * y는 x의 각 원소들과 b의 선형결합이라고 가정
* 모델을 찾는 방법
    * 모델을 수립 : y = W*x + b
    * 목표 : 수집한 데이터 X, y로 W, b를 찾기
    * 방법 : train_X, train_y 데이터로 손실함수가 최소가 되는 W, b를 반복적인 방법으로 수행하면서 찾아 나감
    * 테스트 : 찾은 모델 W, b와 test_X로 예측 값을 계산하여 test_y값과 비교
* 손실함수 종류
    * MAE, MSE, RMSE
        * MAE(Mean Absolute Error) : 평균 절대 오차
        * MSE(Mean Squared Error) : 평균 제곱 오차
        * RMSE(Root Mean Squared Error) : 평균 제곱근 오차
    * 특징 
        * 학습 속도 : MSE = RMSE > MAE, 오류를 크게 증폭할 수록 학습 속도가 빠름
        * robustness : MAE > RMSE > MSE, 모델이 노이즈나 과장된 입력에 안정성을 유지, MSE는 outlier의 영향을 과도하게 받음
        * 학습 안정성 : MSE = RMSE > MAE, MSE는 전범위에서 미분가능하여 안정적인 학습 가능, MAE는 0에서 미분 불가능
    * 선택 방법
        * 일반적으로 MSE를 선택
        * 데이터에 이상치가 많고, 제거할 수 없을 때는 MAE를 사용
        * 빠른 학습을 필요할 때 MSE를 사용
* 반복적으로 최적의 Weight, bias를 찾아나가는 알고리즘
    * 대부분이 Gradient Descent 방법을 사용
    * 기본 알고리즘
        * BGD(Batch Gradient Descent) : 배치 경사 하강법
        * MGD(Mini-batch Gradient Descent) : 미니배치 경사 하강법
        * SGD(Stochastic Gradient Descent) : 확률적 경사 하강법
    * 기본 알고리즘 특징
        * batch 사이즈를 기준으로 SGD, BGD를 양극단에 두고 그 사이에 MGD가 있음
        * SGD는 batch size가 1, BGD는 batch size가 train data 건수, 그 사이에 모든 경우가 MGD 방식임
        * batch size의 의미
            * batch 사이즈 만큼의 train_X, train_y 데이터로 손실함수를 계산하고 W, b를 update 하는 것을 반복
            * 결국 W,b(파라미터)를 update 하는 횟수의 차이가 발생 
                * BGD : batch_size = train_데이터_건수 => 파라미터_업데이트_횟수 = 1
                * MGD : batch_size = 2이상의 수 => 파라미터_업데이트_횟수 = train_데이터_수 / batch_size
                * SGD : batch_size = 1 => 파라미터_업데이트_횟수 = train_데이터_수
            * 배치 사이즈 수 : SGD < MGD < BGD
            * 파라미터 updat 횟수 : SGD > MGD > BGD
        * epoch 이란?
            * 모든 train data를 한 번 완전히 학습하는 단위
            * 모든 훈련 데이터로 학습을 마치면 1 epoch를 마치는 것
            * 모든 배치를 다 훈련하면 1 epoch을 마치는 것
            * 1 epoch 동안 파라미터 update 횟수 : SGD=데이터_수, MGD=데이터_수~1, BGD=1
        * 기본 알고리즘 중에서는 대부분 MGD를 사용함
            * SGD, BGD는 양극단의 방법론이며 현재는 이론적인 의미만 있는 듯 ( 여기서도 정-반-합 패턴이 보임 )
            * SGD 쪽으로 갈수록 파라미터 업데이트가 빨라져 빠른 학습이 이루어지며, 목표달성을 조기에 판단할 수 있어서 효율적임
            * BGD 쪽으로 갈수록 많은 데이터로 손실함수를 계산하여 정확도가 증가, 대신 업데이트 주기가 길어져 비효율적임
        * torch 에서는 SGD만 제공하고 여기에서 모든 경우를 수행 : 수학적으로 세가지 모두 동일해서 가능함
          ```python
          optimizer = optim.SGD(model.parameters(), lr =0.01)
          ```
    * 다양한 개선 알고리즘
        * Momentum : 관성 추가
        * Adam : 적용적 학습률
        * 모두 GD 방식을 기초로 효율성 개선하고 있음 ( 학습률 등 hyper parameter 튜닝 )
        * 일반적으로 Adam 사용

## StandardScaler 활용법
* Linear Regression 모델 찾기 = 손실함수의 값을 거의 0에 수렵시키는 parameter를 찾기
* 학습을 진행해도 손실 함수의 값이 큰 경우
    * 학습률이 큰 경우 (ex, 1.0) : 손실함수 값이 진동하거나 발산할 수 있음
    * 학습률이 작은 경우(ex, 0.000001) : 손실함수 값이 거의 변동이 없음
    * 모델 문제 : 너무 단순하거나 너무 복잡
    * 데이터 문제 : 이상치가 많거나, X,y의 feature간 데이터 크기가 너무 차이가 남
    * 초기화 문제 : 파라미터 초기값에 문제가 있음 ( 최저점을 찾아갈 때 어디에서 부터 시작하는지에 따라 성능 차이가 발생 )
    * 손실 함수 문제 : 적절한 손실함수가 필요한데 그렇지 못한 경우
        * 일반적 회귀 : MSELoss()
        * 이진 분류 : BCELoss(), BCEWithLogitsLoss()
        * 다중 분류 : CrossEntropyLoss()
* 데이터 표준화
    * 데이터 표준화 : 데이터의 평균을 0, 표준편차를 1로 조정함 (표준정규분포를 가지도록 조정)
    * 표준화 방법 : original data ---(X-=평균)--> zero-centered data ---(X/=표준편차)--> normalized data
    * 데이터 문제의 경우 표준화 진행하면 효과가 큼
* StandardScaler
    * sklearn에 포함된 모듈
    * 주어진 데이터의 모든 feature의 평균을 0, 분산을 1로 조정
    * 피처별로 적용
    * 사용 방법
        * 한개나 두개의 Scaler를 생성하여 사용
        * train_X에 적용한 Scaler 생성, y에 적용할 Scaler는 필요할 때만 생성
        * 생성 후 fit -> transform() -> [ inverse_transform() ] 순서로 사용함
        * 주의 : training data에 대해서만 fit 수행( test 데이터 정보가 학습시 사용되면 안됨 )
        * y데이터에 scaler 적용한 경우는 원래 의미를 해석하기 위해 inverse_transform() 수행함
    * 예시
        ```python
        from sklearn.preprocessing import StandardScaler

        X = torch.randn(5, 3).numpy() # 5개의 샘플과 3개의 특성
        y = torch.randn(5).numpy()    # 5개의 샘플에 대한 타겟 값

        # X에 적용할 scaler 객체 생성
        scaler = StandardScaler().fit(X)
        # y에 적용할 scaler 객체 생성
        scaler_y = StandardScaler().fit(y.reshape(-1, 1))
        # X에 transform 적용
        X_scaled = scaler.transform(X)
        # y에 transform 적용
        y_scaled = scaler_y.transform(y.reshape(-1, 1)) 
        ```

* 그 외 scaler
    * MinMaxScaler : 피처별로 [min, max] 사이의 값으로 변경, 디폴트는 [0, 1], 
    * Normalizer : 각 샘플별로 단위벡터로 변환, 크기는 중요하지 않고 방향만 중요할 때 사용( 코사인유사도 )
    * 예시
        ```python
        from sklearn.preprocessing import MinMaxScaler, Normalizer

        mm = MinMaxScaler()
        a = torch.rand((2,3))
        print(a)
        # tensor([[0.2034, 0.1720, 0.1887],
        #         [0.5116, 0.8208, 0.9890]])
        b = mm.fit_transform(a)
        print(b)
        # [[0. 0. 0.]
        #  [1. 1. 1.]]

        norm = Normalizer(norm='l2')
        a = torch.rand((2,3))
        print(a)

        b = norm.fit_transform(a)
        print(b)
        print(torch.norm(torch.tensor(b), p=2, dim=1))  # 행벡터의 L2 norm 계산, 모두 1.0이 나옴
        # tensor([1.0000, 1.0000])
        ```
## 시그모이드 함수란?
* 정의
    * 원래는 S 모양의 함수를 일컫는 용어지만, 현재는 대부분 logistic fuction으로 사용하고 있음
    * 시그모이드 함수는 logistic function외에도 tanh(), arctan(), erf() 등이 있음
    * logistic function = 1/(1+exp(-z))
* 용도
    * 이진분류 문제에서 입력변수를 0~1사이의 확률로 매핑하는 역할을 수행함
    * 예를들면, 시그모이드 함수 출력 결과가 0.5를 넘으면 True, 아니면 False로 판단
    * 암 진단, 질병 예측, 고객 구매 확률, 대출 상황 여부 등
* logit()과의 관계
    * logit() : (0, 1) 확률 -> 실수 전체
    * sigmoid() : 실수 전체 -> (0, 1) 확률
    * logit(p) = ln(p/(1-p))

## BCELoss(Binary Cross Entropy) 함수란?
* 이진 분류 : BCELoss(), BCEWithLogitsLoss()
* sigmoid()로 확률을 예측할 때 0, 1로 구성된 y 값을 MSE로만 파악하기에는 한계가 있음
    * Grandient Vanishing 문제 : sigmoid() 미분이 극값에서 0에 가까워짐
    * 학습 속도 저하 : 틀릴수록 gradient가 작아지는 역설
    * 대안으로 제시된 함수가 BCELoss() 함수
* 특징
    * 0~1 사이의 확률을 입력 받음
    * sigmoid() 거친 값을 입력으로 전달되어야 함
    * 0, 1 예측이 얼마나 잘못되었는지를 나타냄
    * 0~무한대 : 0은 예측이 매우 정확함을 의미, 무한대는 예측이 매우 부정확함을 의미
* Cross 의미
    * 예측 분포화, 실제 분포 사이의 차이를 계산한다는 의미
    * 예측과 실제 값이 단위 초정육면체내의 벡터로 한정되어 있어 유클리디언 거리로 loss를 구하는 것은 한계가 있음
    * 단위 초정육면체내의 두 벡터간의 거리의 최대값은 √n 임
    * 잘못 예측한 경우라도 MSE 값은 한정되어 있는 반면, BCE는 무한대까지 커질 수 있음
* enthropy의미
    * 예측과 실제의 정보량 차이를 분석한다는 의미
    * 엔트로피가 0에 가까우면 예측이 맞는 것을 의미
    * 엔트로피가 커지면 예측이 틀리다는 것을 의미
    * 엔트로피 0, 확실함
    * 엔트로피 커지면, 불활실
    * 무작위로 0, 1을 예측했을 때 cross entropy가 0.693이 나옴 ( 반은 맞고, 반은 틀림 )

## CrossEntropyLoss() 함수란?
* 다중 분류 : CrossEntropyLoss()
* 특징 
    * logit()값을 입력 받고 있음 
    * softmax() 결과를 입력하면 안됨 
    * softmax 함수 호출이 내장됨
    * 모델 설계시 마지막에 activation 함수가 불필요 함

## Linear 모델의 파라미터 초기화
* 파라미터 초기화 방법에 따라 학습 퍼포먼스 차이가 심하게 남
* 다양한 초기화 방법
    * zero 초기화 (비추천) : 학습이 제대로 이루어 지지 않음
    * random 값 (비추천) : 가능은 하지만 효율적이지 않음
    * Xavier 초기화(추천) : sigmoid 함수에 적합
    * He초기화(추천) : ReLU 함수에 적합
* 검증된 모듈 사용 ( torch 같은 )
    ```python
    from torch import nn

    weight = torch.empty(3, 5)

    # sigmoid ( xavier 초기화 )
    nn.init.xavier_normal_(weight)
    print(weight)
    nn.init.xavier_uniform_(weight)
    print(weight)

    # ReLU ( He 초기화 )
    nn.init.kaiming_normal_(weight, mode='fan_in', nonlinearity='relu')
    print(weight)
    nn.init.kaiming_uniform_(weight, mode='fan_in', nonlinearity='relu')
    print(weight)
    ```

## 총정리
* 데이터 전처리
    * StandardScaler 사용
* 문제별 해결 방법 정리
    |문제              | 해결 방법                  | 활성화 함수| 손실함수 | 출력 |
    |------------------|--------------------------|-----------|--------|------|
    |변수간 선형관계 분석 |선형회귀<br>(Linear Regression)|None, ReLU| MSELoss | y feature 수만큼의 임의의 수 |
    |이진분류<br>(Binary Classification)|로지스틱회귀<br>(Logistic Regression)|sigmoid  | BCELosss | 1개 확률 |
    |다중분류<br>(Muliclass Classification)|다항로지스틱회귀<br>(Multinomial Logistic Regresssion)|softmax  | CrossEntropyLoss | 클래수 수만큼 (확률분포)|
* parameter초기화
    * paramer 초기화 방법에 따라 학습 속도가 차이가 심하게 나기 때문에 반드시 적절한 초기화 필수
    * Xavier 초기화(추천) : sigmoid 함수에 적합
    * He초기화(추천) : ReLU 함수에 적합
