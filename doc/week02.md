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
* 

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
    * $ A=U\Sigma V^{T} $
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
    * 
* 

## 다중선형회귀 구조 분석
* 손실함수로 MSE를 사용할 때 다중선형회귀 구조는 아래와 같음
    * 데이터 $ X=(m,n), y=(m,1)$가 주어질 때 
    * $\hat{y} = X\beta + b, \qquad \hat{y}=(m,1), X=(m,n), \beta=(n,1), b=(1,1)  $
    * 이를 시각적으로 표현하면 
        * training
            * 주어진 데이터 $ X, y$로 training과정을 거쳐 $\beta, b$ 구하기
            * $MSE = \frac{1}{m}\lVert y - \hat{y} \rVert^2 $ 가 가장 작아지도록 $\beta, b$ 를 최적화
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
            * $x\beta +b$로 예측하고 결과 $\hat{y}$은 real $y$에 근접할 것이라고 기대
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
            \hat{y}
        \end{bmatrix} \approx
        \begin{bmatrix}
            y
        \end{bmatrix}
        $$
* 손실함수와 그레디언트
    * 손실함수
    $$
    \begin{align*} 
    MSE & = \frac{1}{m}\lVert y - \hat{y} \rVert^2 \\
    & = \frac{1}{m}\lVert y - X\beta \rVert^2 \\
    & = \frac{1}{m} (y - X\beta)^T(y - X\beta) \\
    & = \frac{1}{m}(y^Ty - y^TX\beta - (X\beta)^Ty + (X\beta)^TX\beta) \\
    & = \frac{1}{m}(y^Ty -2y^TX\beta + \beta^TX^TX\beta) \qquad (\because (X\beta)^Ty \text{ is scalar, so } = y^TX\beta )
    \end{align*}
    $$
    * 벡터 미분 성질
        $$
        \begin{align*} 
        & \nabla_\beta(a^T\beta) = a, \nabla_\beta(\beta^Ta) = a \\
        & \nabla_x(x^Tx) = 2x\\
        & \nabla_\beta(\beta^TA\beta) = (A+A^T)\beta, \quad 2A\beta \text{(when A is symetric)} \\
        & \nabla_\beta(\beta^T A^TA\beta) = 2A^TA\beta, \quad \because A^TA \text{ is symetric} \\
        \end{align*}
        $$
    * 그레디언트 of MSE
        $$
        \begin{align*} 
        \nabla_\beta MSE & = \nabla_\beta \big( \frac{1}{m}\lVert y - \hat{y} \rVert^2  \big)\\
        & = \nabla_\beta \big(\frac{1}{m}(y^Ty -2y^TX\beta + \beta^TX^TX\beta) \big)\\
        & = \frac{1}{m}(0 -2y^TX + 2X^TX\beta), \qquad (\because \text{위 벡터 미분 성질 이용 } ) \\
        & = \frac{1}{m}(0 -2X^Ty + 2X^TX\beta),\qquad (\because y^TX = X^Ty )\\
        & = -\frac{2}{m} (X^Ty - X^TX\beta) \\
        & = -\frac{2}{m}X^T(y - X\beta) \\
        & = -\frac{2}{m}X^T(y - \hat{y})
        \end{align*}
        $$