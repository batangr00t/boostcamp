# Week 2 - 주간학습정리 - AI Math

## 깨달은 내용
* 혼란스러운 용어 정리
* pseudo-inverse ( Moore-Penrose inverse ) - left, right
* eigendecomposition 조건
* PCA(Principal Component Analysis)
* SVD(Singular value Decomposition)

## 혼란스러운 용어 정리
* 차원 (dimention)
    * 라이브러리, 수학 차이
* inner prodoct
    * vector
    * matrix : 라이브러리, 수학 차이

## pseudo-inverse ( Moore-Penrose inverse ) - left, right
* 

## eigendecomposition 조건
* 행렬 A는 조건이 맞는 경우 아래와 같이 분해할 수 있음
    * $$A = V\Lambda V^{-1}$$
    * V : 고유벡터를 열벡터로 한 행렬
    * $\Lambda$ : 고유갑이 대각성분인 행렬
* 고유값 분해의 조건
    * 행렬 A가 square matrix 
        * eigen value, eigen vector 정의가 $Av=\lambda v$
        * $\Rightarrow$ 정의역과 공역에 속해있는 원소들의 차원이 동일해야 함
        * $\Rightarrow$ A가 nxn matrix
    * A의 모든 고유벡터가 선형독립
        * eigendecomposition이 성립하려면 $V^{-1}$가 존재해야 함
        * $\Leftrightarrow$ V가 full rank  
        * $\Leftrightarrow$ 모든 고유벡터가 선형독립

## PCA(Principal Component Analysis)
* squal

## SVD(Singular value Decomposition)
* 인버스에 활용
    * $ A=U\Sigma V^{T} $ 로 분해 될 때 
    * $ A^{+} = V\Sigma^{+}U^{T} $ 
* 조건별 인버스 구하기
  |  조건  |  방법  |  numpy | torch   |
  |-------|-------|--------|---------|
  |square matrix, full rank |inverse| np.linalg.inv(A) | torch.linalg.inv(A) |
  |full rank(row or column)|pseudo inverse<br>(Moore-Penrose inverse)| np.linalg.pinv(A) |  torch.linalg.pinv(A) |
  |square matrix일 필요 없음<br>singular도 됨<br>rank=0일 때도 됨|SVD| np.linalg.svd(A, full_matrices=False) | torch.linalg.svd(A, full_matrices=False)|