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
* 차원 (dimention)
    * 라이브러리, 수학 차이
* inner prodoct
    * vector
    * matrix : 라이브러리, 수학 차이

## pseudo-inverse ( Moore-Penrose inverse ) - left, right
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
        * $A_2 = (I) R_1 Q_1 = (Q_1^TQ_1) R_1 Q_1 = Q_1^T (Q_1 R_1) Q_1 = Q_1^T A_1 Q_1 \rightarrow A_1 \approx A_2$

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