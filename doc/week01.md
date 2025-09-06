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
* 텐서 생성
    * zeros(), ones(), eye(), full(), rand(), randn(), randint()
    * shape 을 파라미터로 입력
        * 0-D : [], ()
        * 1-D : [2], (3), [2,], (3,)
        * 2-D : [3,4], (3,4)
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
* dim 의미
    * 텐서는 ndim 차원의 수만큼의 dim을 갖음
    * 가장 외각 배열의 차원을 dim-0으로 하여 가장 마지막 단계의 차원을 dim-(ndim-1)로 칭함
    * 가장 마지막 차원을 -1로도 칭함, 그 다음은 -2, ... 가장 외각 배열의 차원은 -ndim 으로 칭함
    * [3, 4, 5] shape의 tensor 의 경우
        * 0 차원, -3 차원 : 사이즈 3
        * 1 차원, -2 차원 : 사이즈 4
        * 2 차원, -1 차원 : 사이즈 5
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
        
        # dim-0차원 2개의 배열을 더함, 스칼라 탠서로 변하면서 차원이 사라짐 => 결과는 (3,4) shape의 탠서
        a.sum(dim=0)
        # tensor([[12, 11,  5, 13],
        #         [10, 10, 11,  8],
        #         [15,  9, 13, 15]])

        # dim-1차원 3개의 배열을 더함, 스칼라 텐서로 변하면서 해당 차원이 사라짐 => 결과는 (2,4) shape의 탠서
        a.sum(dim=1)
        #tensor([[21, 19, 11, 18],
        #        [16, 11, 18, 18]])

        # dim-2차원 4개의 배열을 더함, 스칼라 텐서로 변하면서 해당 차원이 사라짐 => 결과는 (2,3) shape의 탠서
        a.sum(dim=2)
        # tensor([[25, 15, 29],
        #         [16, 24, 23]])

        # dim-2차원 4개의 배열을 더함 => 결과는 (2,3) shape의 탠서
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
## StandardScaler 활용법
## 시그모이드 함수란?
## BCELoss(Binary Cross Entropy) 함수란?
## CrossEntropyLoss() 함수란?
## Linear 모델의 파라미터 초기화