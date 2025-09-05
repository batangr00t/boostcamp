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
vector1 = torch.tensor([1,2,3])
print(vector1)       # tensor([1, 2, 3])
print(vector1.shape) # torch.Size([3])
print(vector1.ndim)  # 1
print(vector1.dtype) # torch.int64

vector2 = torch.randn([5])
print(vector2)       # tensor([ 0.6466, -0.5115,  1.9365,  0.9905, -0.0909])
print(vector2.shape) # torch.Size([5])
print(vector2.ndim)  # 1
print(vector2.dtype) # torch.float32

# 2-D tensor
matrix1 = torch.zeros(3, 4)
print(matrix1)
# tensor([[0., 0., 0., 0.],
#         [0., 0., 0., 0.],
#         [0., 0., 0., 0.]])
print(matrix1.size()) # torch.Size([3, 4])
print(matrix1.dim())  # 2
print(matrix1.dtype)  # torch.float32

matrix2 = torch.eye(4, dtype=torch.double)  # 4x4 단위행렬
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
* 