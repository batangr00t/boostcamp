# Week 2 - 주간학습정리 - AI Math

## 깨달은 내용
* 혼란스러운 용어 정리
* pseudo-inverse ( Moore-Penrose inverse ) - left, right
* Ridge/Lasso regression
* eigendecomposition
* PCA(Principal Component Analysis)
* SVD(Singular value Decomposition)
* 다중선형회귀 구조 분석

## 혼란스러운 용어 정리
* 차원 (dimention, ndim)
    * 선형대수에서 사용하는 차원의 의미 : vector space의 base element의 원소 갯수
    * phtion, numpy, pytorch에서 사용하는 차원의 의미
        * ndim : 차원의 갯수 = 중첩배열의 중첩 횟수
        * 여러 함수에서 사용하는 dim=0, dim=1 같은 코드의 의미는 가장 바깥 배열을 0으로 하여 안쪽 배열로 들어갈 수록 1씩 증가하여 index를 부여한 것
    * 비교 예시
        * $A = [[1,3], [2,6]]$
        * 수학에서 A column space 의 차원은 1임 : base가 {$[1,3]$}으로 원소가 하나이기 때문
        * A.ndim=2 임 : 배열이 두번 중첩되어 있기 때문
* inner prodoct : 라이브러리, 수학 차이
    * vector : 두 벡터의 내적을 구하는 것으로 라이브러리, 수학에서 동일한 개념
        * np.inner(a, b) = np.sum(a*b) : 각 성분별 곱의 합
    * matrix : 서로 다름
        * 라이브러리 : 마지막 차원의 벡터 내적을 계산, 결과는 행렬
        * 수학 : 프로베니우스 내적이 대표적인 행렬간 내적, 결과는 스칼라
        * 프로베니우스 내적 : $tr(XY^T)$ = X, Y의 element-wise 곱의 합

## pseudo-inverse ( Moore-Penrose inverse ) - left, right
* m, n이 달라 역행렬을 계산할 수 없을 때 유사 역행렬을 이용할 수 있음
    * left inverse
        * $A^+ = (A^TA)^{-1}A^T$
        * 답이 없는 상황에서 유사해 찾기
        * 열공간 Col(A)에서 b와 가장 가까운 점 찾기
    * right inverse 
        * $A^+ = A^T(AA^T)^{-1}$
        * 답이 무수히 많은 상황에서 경제적인 답 찾기
        * $Ax=b$를 만족하는 무수히 많은 해 중 Row(A)에 속하는 해 찾기
* 조건
    * 위 계산식은 row full rank이거나, column full rank일 때 사용하는 방식
    * rank가 row나 column 보다도 적은 경우는 SVD 이용하여 구할 수 있음

## Ridge/Lasso regression
* 다중회귀 모형에서 과적합이 발생할 때 일반화 성능을 높이기 위해 개선한 방법
* Ridge 회귀
    * 손실에 파라미터 제곱을 패널티 항으로 더해준다
    * 패널티 항이 $\lambda \sum_{i=1}^p\beta^2$ 이기 때문에 L2 Regression 이라고도 한다
    * 제약조건 영역을 그리면 원형으로 나타나기 때문에 일부 파라미터가 0에 가까워 질수 있다
* Lasso 회귀
    * 손실에 파라미터 절대값을 패널티 항으로 더해준다
    * 패널티 항이 $\lambda \sum_{i=1}^p\lvert\beta\rvert$ 이기 때문에 L1 Regression 이라고도 한다
    * 제약조건에 경계를 그리면 마름모꼴로 나타나기 때문에 $\beta$의 제약조건에 맞는 값을 찾을 대 모서리에서 찾을 확률이 있어서, 그 때는 일부 파라미터 값이 0이기 때문에 파라미터 selection 효과가 있다

## matrix decomposition
### QR decomposition
* 행렬 A는 아래와 같이 분해될 수 있음
    * $A = QR$
    * $Q: m \times m \text{ orthogonal, unitary matrix}$
    * $R: m \times n \text{ upper triangular matrix}$ 
* 조건 
    * column full rank,  rank(A) = m
* 용도
    * $Ax=b$ 방정식을 푸는데 편리함 : $ Rx = Q^{T}b$ 로 변경하여 쉽게 품
    * eigen value, eigen vector 구하는데 활용할 수 있음
        * $A_1 = Q_1 R_1$ 일 때,
        * $A_2 = R_1 Q_1 \text{ 라 하면} \\
        \rightarrow A_2 = (I) R_1 Q_1 = (Q_1^TQ_1) R_1 Q_1 = Q_1^T (Q_1 R_1) Q_1 = Q_1^T A_1 Q_1 \\
        \rightarrow A_1 \text{ is similar to } A_2 \\
        \rightarrow A_1, A_2\text { 는 engenvalue가 같다}$ 
        * $A_{k+1} = R_k Q_k \text{ 로 두면} \\
        \rightarrow A_{k+1} = (Q_k^TQ_k) R_k Q_k = Q_k^T (Q_k R_k) Q_k = Q_k^T A_k Q_k \\
        \rightarrow A_{k+1} \text{ is similar to }  A_k$
        * 결국 $A_1, A_2, A_3, ... A_n$ 은 모두 같은 eigen value를 갖는다

### eigen decomposition
* 행렬 A는 아래와 같이 분해할 수 있음
    * $A = V\Lambda V^{-1}$
    * $V$ : 고유벡터를 열벡터로 한 행렬
    * $\Lambda$ : 고유값이 대각성분인 행렬
* 고유값 분해의 조건
    * 행렬 A가 square matrix 
        * eigen value, eigen vector 정의가 $Av=\lambda v$
        * $\Rightarrow$ 정의역과 공역에 속해있는 원소들의 차원이 동일해야 함
        * $\Rightarrow$ A가 nxn matrix
    * A의 모든 고유벡터가 선형독립
        * eigendecomposition이 성립하려면 $V^{-1}$가 존재해야 함
        * $\Leftrightarrow$ V가 full rank  
        * $\Leftrightarrow$ 모든 고유벡터가 선형독립
* 고유값 분해 방법
    * QR Decomposition 이용하는 방법
    * Singular Value Decomposition 이용하는 방법
* 용도
    * $A^n$ 계산 : $A^n = V \Lambda^n V^{-1}$
    * PCA(Principal Component Analysis) : [PCA process](week02_PCA.ipynb)

## Singular Value Decomposition(SVD)
* 행렬 A는 아래와 같이 분해할 수 있음
    * $A=U \Sigma V^{T}$
* 활용
    * 인버스에 활용
        * $ A^{+} = V\Sigma^{+}U^{T} $ 
        * 조건별 인버스 구하기
            |  조건  |  방법  |  numpy | torch   |
            |-------|-------|--------|---------|
            |square matrix, full rank |inverse| np.linalg.inv(A) | torch.linalg.inv(A) |
            |full rank(row or column)|pseudo inverse<br>(Moore-Penrose inverse)| np.linalg.pinv(A) |  torch.linalg.pinv(A) |
            |square matrix일 필요 없음<br>singular도 됨<br>rank=0일 때도 됨|SVD| np.linalg.svd(A, full_matrices=False) | torch.linalg.svd(A, full_matrices=False)|
    * eigen decposition 구하는데 활용 ( 제한적 )

## 다중선형회귀 구조 분석
* 일반적인 선형회귀 구조
    * 데이터 $ X=(m,n), y=(m,1)$가 주어짐
    * 주어진 데이터는 X -> Linear function => activation function -> loss function 를 거처 scalar로 계산
        * Linear function : $z = X\beta + b, \qquad z=(m,1), X=(m,n), \beta=(n,1), b=(1,1)$
            * 엄밀한 관점에서는 선형변환이 아니지만, $[X, 1]$ 형태로 해석할 수 있기 때문에 선형 변환이라고 인식함
            * $[X, 1]$ column들의 선형결합(linear combination)을 계산하여 변환
        * activation function : $\hat{y} = \sigma(z), \qquad \hat{y}=(m,1), z=(m,1)  $
            * 비선형성을 가미하여 다양한 비선형 모델을 만들기 위해 추가
            * activation 함수를 사용하지 않을 경우엔 activation function으로 identity() 함수를 사용한 것으로 간주할 수 있음
            * 활성화 함수 종류 : step(), sigmoid(), tanh(), ReLU(), leakyReLU(), softmax()
            * 모두 vector를 같은 shape의 vector로 변환하는 element-wise 함수
        * loss function : $L = loss(y, \hat{y}) \qquad L=(1,1), y=(m,1), \hat{y}=(m,1) $
            * 예측값과 target의 차이를 측정하기 위해 사용, 이 손실을 최소화 하는 방향으로 최적화 진행
            * 손실 함수 종류 : MAE, MSE, RMSE, BCE, CCE
            * 모두 vector를 스칼라로 변환하는 함수
    * 학습한 결과를 활용할 때는
        * 취득한 데이터가 $x$일 때 $x\beta +b$로 결과 예측, $x=(1,n), \beta=(n,1), b=(1,1)$
    * 이를 시각적으로 표현하면 
        * training
            * 주어진 데이터 $X, y$로 training과정을 거쳐 $\beta, b$ 구하기
            * loss function의 결과가 최소화 되도록 $\beta, b$를 update 시킴
            * $X, y$: 상수,  $\beta, b$: 변수
        $$
        \begin{bmatrix}
            \cdots & (x^{(1)})^T & \cdots\\
            \cdots & (x^{(2)})^T & \cdots\\
            & \vdots & \\
            & \vdots & \\
            & \vdots & \\
            & \vdots & \\
            \cdots & (x^{(m)})^T & \cdots\\
        \end{bmatrix} @
        \begin{bmatrix}
            \beta_1 \\
            \beta_2 \\
            \vdots \\
            \beta_n \\
        \end{bmatrix} +
         \begin{bmatrix}
            b 
        \end{bmatrix} =
        \begin{bmatrix}
            z_1 \\
            z_2 \\
            \vdots \\
            \vdots \\
            \vdots \\
            \vdots \\
            z_m \\
        \end{bmatrix} \xrightarrow{\sigma}
        \begin{bmatrix}
            \hat{y}_1 \\
            \hat{y}_2 \\
            \vdots \\
            \vdots \\
            \vdots \\
            \vdots \\
            \hat{y}_m \\
        \end{bmatrix} \approx
        \begin{bmatrix}
            y_1 \\
            y_2 \\
            \vdots \\
            \vdots \\
            \vdots \\
            \vdots \\
            y_m \\
        \end{bmatrix}
        $$
        * 추론
            * 주어진 파라미터 $\beta, b$로 임의의 $x$에 대한 결과 구하기
            * $\sigma(x\beta +b)$로 예측하고 결과 $\hat{y}$은 real $y$에 근접할 것이라고 기대
            * $\beta, b$: 상수, $x$: 변수
        $$
        \begin{bmatrix}
         x_1 x_2 \cdots x_n
        \end{bmatrix} @
        \begin{bmatrix}
            \beta_1 \\
            \beta_2 \\
            \vdots \\
            \beta_n \\
        \end{bmatrix} +
        \begin{bmatrix}
            b 
        \end{bmatrix} =
        \begin{bmatrix}
            z 
        \end{bmatrix} \xrightarrow{\sigma} 
        \begin{bmatrix}
            \hat{y}
        \end{bmatrix} \approx
        \begin{bmatrix}
            y
        \end{bmatrix}
        $$        
    * 최적화 방법 : 경사하강법(Gradient descent)
        * 손실함수가 줄어드는 방향으로 $\beta, b$를 수정을 반복 수행
        * 손실함수에 대한 $\beta, b$의 편미분이 사용됨
            $$
            \begin{align*} 
            & \frac{\partial L}{\partial \beta} = \frac{\partial L}{\partial \hat{y}} 
                                                \frac{\partial \hat{y}}{\partial z}
                                                \frac{\partial z}{\partial \beta},  
            \qquad \frac{\partial L}{\partial b} = \frac{\partial L}{\partial \hat{y}} 
                                                \frac{\partial \hat{y}}{\partial z}
                                                \frac{\partial z}{\partial b}  \\
            \end{align*}
            $$
            * $\frac{\partial L}{\partial \hat{y}}$ : 손실함수에 따라 달라짐
            * $\frac{\partial \hat{y}}{\partial z}$ : 액티베이션 함수에 따라 달라짐
            * $\frac{\partial z}{\partial \beta}, \frac{\partial z}{\partial b}$: 다중선형회귀는 항상 동일
        * 각 편미분을 구해보면
            * $\frac{\partial L}{\partial \hat{y}}$ 은 손실함수별로 달라지나 형태는 동일함
                $$
                \frac{\partial L}{\partial \hat{y}} =                 
                \begin{bmatrix}
                    \frac{\partial L}{\partial \hat{y}_1} \\
                    \frac{\partial L}{\partial \hat{y}_2} \\
                    \vdots \\
                    \frac{\partial L}{\partial \hat{y}_m} \\
                \end{bmatrix}, \qquad \frac{\partial L}{\partial \hat{y}_i} \text{ 는 i번째 예측값이 변할 때 Loss의 변화율을 의미 }
                $$
            * $\frac{\partial \hat{y}}{\partial z}$ 은 activaion function 별로 달라지나 형태는 동일함
                $$
                \frac{\partial \hat{y}}{\partial z} =
                \begin{bmatrix}
                    \frac{\partial \hat{y}_1}{\partial z_1} & 0 & \cdots & 0\\
                    0 & \frac{\partial \hat{y}_2}{\partial z_2}  & \cdots & 0\\
                    \vdots & \vdots & \ddots & \vdots \\
                    0 & 0 & \cdots & \frac{\partial \hat{y}_m}{\partial z_m} \\
                \end{bmatrix}, \qquad \frac{\partial \hat{y}_i}{\partial z_i}  \text{ 는 i번째 선형변환 값이 변할 때 i번째 예측값의 변화율을 의미 }
                $$
                * $\frac{\partial \hat{y}_i}{\partial z_j} = 0, \text{ when } i \neq j$ \
                activation funtion은 element-wise 함수이기 때문에 위치가 다른 변수끼리 영향을 주지 않음
                * 코드로 실행할 때는 $\frac{\partial \hat{y}}{\partial z}$ 를 대각선 성분만으로 vector로 생성하고 element-wise product로 처리
                ```python
                # error = dL/dy_hat, (m,1) shape
                dy_hat_dz = - ( y - y_hat ) # (m,1) vector로 처리
                dL_dz = error * dy_hat_dz   # *는 element-wise product 
                ```
            * $\frac{\partial z}{\partial \beta}, \frac{\partial z}{\partial b}$ 는 다중선형회귀에서는 항상 동일함
                * $z = X\beta + b$
                $$
                \begin{bmatrix}
                    x_{11} & x_{12} & \cdots & x_{1n} \\
                    x_{21} & x_{22} & \cdots & x_{2n} \\
                    \vdots & \vdots & \vdots & \vdots \\
                    \vdots & \vdots & \vdots & \vdots \\
                    \vdots & \vdots & \vdots & \vdots \\
                    \vdots & \vdots & \vdots & \vdots \\
                    x_{m1} & x_{m2} & \cdots & x_{mn} \\
                \end{bmatrix} @
                \begin{bmatrix}
                    \beta_1 \\
                    \beta_2 \\
                    \vdots \\
                    \beta_n \\
                \end{bmatrix} +
                \begin{bmatrix}
                    b 
                \end{bmatrix} =
                \begin{bmatrix}
                    z_1 \\
                    z_2 \\
                    \vdots \\
                    \vdots \\
                    \vdots \\
                    \vdots \\
                    z_m \\
                \end{bmatrix} 
                $$
                * 다음과 같은 형식
                $$
                \begin{align*} 
                & \frac{\partial z}{\partial \beta} =
                \begin{bmatrix}
                    \frac{\partial z_1}{\partial \beta_1} & \frac{\partial z_2}{\partial \beta_1} & \cdots &\frac{\partial z_m}{\partial \beta_1} \\
                    \frac{\partial z_1}{\partial \beta_2} & \frac{\partial z_2}{\partial \beta_2} & \cdots & \frac{\partial z_m}{\partial \beta_2} \\
                    \vdots & \vdots & \vdots & \vdots \\
                    \frac{\partial z_1}{\partial \beta_n} & \frac{\partial z_2}{\partial \beta_n} & \cdots & \frac{\partial z_m}{\partial \beta_n}\\
                \end{bmatrix} = X^T,\quad & \frac{\partial z_i}{\partial \beta_j} 
                \text{ 는 j번째 beta가 변할때 zi가 얼마만큼 변하는지 의미 } \\
                & \frac{\partial z}{\partial b} = 
                \begin{bmatrix}
                    \frac{\partial z_1}{\partial b}\\
                    \frac{\partial z_2}{\partial b}\\
                    \vdots \\
                    \frac{\partial z_m}{\partial b}\\
                \end{bmatrix} =                 
                \begin{bmatrix}
                    1\\ 1\\ \vdots \\  1\\
                \end{bmatrix}, \quad & \frac{\partial z_i}{\partial b} 
                \text{ 는 b가 변할 때 zi가 얼마만큼 변하는지 의미 } \\
                \end{align*}
                $$
                * python 코드로 보면 
                    ```python
                    # error = dL/dz (m, 1) shape
                    beta_grad = np.transpose(X) @ error
                    b_grad = np.transpose(np.ones(m,1)) @ error = np.sum(error)
                    ```
* 손실함수로 MSE를 사용할 때 다중선형회귀 구조는 아래와 같음
    * 위에서 정리한 일반적인 다중선형회의 구조에 대입하여 해석하면 아래와 같음
        * Linear function : 변화 없이 동일,  $z = X\beta + b$
        * activation function : 사용하지 않음으로 $\hat{y} = z$
        * loss function : $MSE = \frac{1}{m}\lVert y - \hat{y} \rVert^2 $

* 손실함수와 그레디언트
    * 벡터 미분 성질 기본 성질
        $$
        \begin{align*} 
        & \nabla_\beta(a^T\beta) = a, \nabla_\beta(\beta^Ta) = a \\
        & \nabla_x(x^Tx) = 2x\\
        & \nabla_\beta(\beta^TA\beta) = (A+A^T)\beta, \quad 2A\beta \text{(when A is symetric)} \\
        & \nabla_\beta(\beta^T A^TA\beta) = 2A^TA\beta, \quad \because A^TA \text{ is symetric} \\
        \end{align*}
        $$
    * 손실함수
        $$
        \begin{align*} 
        MSE & = \frac{1}{m}\lVert y - \hat{y} \rVert^2 \\
        & = \frac{1}{m} (y - \hat{y})^T(y -\hat{y}) \\
        & = \frac{1}{m}(y^Ty - y^T\hat{y} - \hat{y}^Ty + \hat{y}^T\hat{y}) \\
        & = \frac{1}{m}(y^Ty -2y^T\hat{y} + \hat{y}^T\hat{y}) \qquad (\because \hat{y}^Ty \text{ is scalar, so } = y^T\hat{y} )
        \end{align*}
        $$
    * 그레디언트 of MSE
        $$
        \begin{align*} 
        \nabla_{\hat{y}} L 
        & = \nabla_{\hat{y}} \big( \frac{1}{m}\lVert y - \hat{y} \rVert^2  \big) \\
        & = \nabla_{\hat{y}} \big( \frac{1}{m}(y^Ty -2y^T\hat{y} + \hat{y}^T\hat{y}) \big)\\
        & = \frac{1}{m}(0 -2y + 2\hat{y}), \qquad (\because \text{위 벡터 미분 성질 이용 } ) \\
        & = -\frac{2}{m}(y - \hat{y}) \\
        \\
        \nabla_{z} \hat{y} 
        & = ones(), \text {(m,1) shape}\\
        \end{align*}
        $$
    * 최종 $\nabla_\beta L, \nabla_b L$
        $$
        \begin{align*} 
        \nabla_\beta L
        & = \frac{\partial L}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial z} \frac{\partial z}{\partial \beta} \\
        & = X^T @ ( ones() * (-\frac{2}{m}(y - \hat{y})) ) \\
        & = -\frac{2}{m} X^T @ (y - \hat{y}) \\
        \\
        \nabla_b L
        & = \frac{\partial L}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial z} \frac{\partial z}{\partial b} \\
        & = ones()^T @ \big( ones() * (-\frac{2}{m}(y - \hat{y}))\big) \\
        & = -\frac{2}{m} ones()^T @ (y - \hat{y}) \\
        & = -\frac{2}{m} sum(y - \hat{y}) \\
        & = -2mean(y - \hat{y}) \\
        \end{align*}
        $$ 