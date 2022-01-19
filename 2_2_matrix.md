## 행렬이란
* 행렬(matrix)은 벡터를 원소로 가지는 2차원 배열이다.
$$
X=\begin{bmatrix}
1 & -2 & 3 \\
-7 & 8 & -9 \\
4 & -5 & -6 \\
\end{bmatrix}
$$
``` python
X = np.array([[1, -2, 3],
              [-7, 8, -9],
              [4, -5, 6]])
```
$$
X = \begin{bmatrix}X_{1} \\ X_{2} \\ \vdots \\ X_{n}\end{bmatrix} = 
\begin{bmatrix}
x_{11} & x_{12} & \cdots & x_{1m} \\
x_{21} & x_{22} & \cdots & x_{2m} \\
\vdots & \vdots & \cdots & \vdots \\
x_{n1} & x_{n2} & \cdots & x_{nm} \\
\end{bmatrix} = (x_{ij})
$$

* 행렬은 **여러 점들**을 나타낸다.
* 행렬의 행벡터 $x_{i}$는 **$i$번째 데이터**를 의미한다.
* 행렬의 $x_{ij}$는 **$i$번째 데이터의 $j$번째 변수의 값**을 말한다.

<br />

## 행렬 곱셈
* 행렬 곱셈은 **$i$번째 행벡터와 $j$번째 열벡터 사이의 내적**을 성분으로 가지는 행렬을 계산한다.
* 행렬 곱은 **$X$의 열의 개수와 $Y$의 행의 개수**가 같아야 한다.
$$
X = \begin{bmatrix}
x_{11} & x_{12} & \cdots & x_{1m} \\
x_{21} & x_{22} & \cdots & x_{2m} \\
\vdots & \vdots & \cdots & \vdots \\
x_{n1} & x_{n2} & \cdots & x_{nm} \\
\end{bmatrix} \quad
Y = \begin{bmatrix}
y_{11} & y_{12} & \cdots & y_{1m} \\
y_{21} & y_{22} & \cdots & y_{2m} \\
\vdots & \vdots & \cdots & \vdots \\
y_{n1} & y_{n2} & \cdots & y_{nm} \\
\end{bmatrix} \quad
XY = (\sum_{k} x_{ik}y_{kj})
$$
``` python
X = np.array([[1, -2, 3],
              [7, 5, 0],
              [-2, -1, 2]])
Y = np.array([[0, -1],
              [1, -1],
              [-2, 1]])

print(X @ Y)
# array([[-8, 6],
#        [5, 2],
#        [-5, 1]])
```

<br />

## 행렬의 내적
* 넘파이의 ```np.inner```는 **$i$번째 행벡터와 $j$번째 행벡터 사이의 내적**을 성분으로 가지는 행렬을 계산한다.
* 행렬의 곱이 **행벡터와 열벡터의 내적**이라면 ```np.inner```은 **행벡터와 행벡터의 내적**이다.
``` python
X = np.array([[1, -2, 3],
              [7, 5, 0],
              [-2, -1, 2]])
Y = np.array([[0, 1, -1],
              [1, -1, 0]])

print(np.inner(X, Y))
# array([[-5, 3],
#        [5, 2],
#        [-3, -1]])
```

<br />

## 행렬의 다른 의미
* 행렬은 **벡터공간에서 사용되는 연산자**이다.
* 행렬곱을 통해 벡터를 **다른 차원의 공간**으로 보낼 수 있다.
* 행렬곱을 통하여 **패턴을 추출**할 수 있고 **데이터를 압축**할 수 있다.
* 선형변환(linear transform)이라고도 한다.
$$
z_{\color{Red}i} = \sum_{\color{Blue}j}a_{\color{Red}i\color{Blue}j}x_{\color{Blue}j} \qquad
Z = AX
$$
$$
\begin{bmatrix}z_{1} \\ z_{2} \\ \vdots \\ z_{n}\end{bmatrix} = 
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1m} \\
a_{21} & a_{22} & \cdots & a_{2m} \\
\vdots & \vdots & \cdots & \vdots \\
a_{n1} & a_{n2} & \cdots & a_{nm} \\
\end{bmatrix}
\begin{bmatrix}x_{1} \\ x_{2} \\ \vdots \\ x_{n}\end{bmatrix}
$$
* m차원 공간에서 $A$를 통하여 n차원 공간으로 변환.

<br />

## 역행렬
* 어떤 행렬 $A$의 연산을 거꾸로 되돌리는 행렬을 **역행렬**이라 한다.
* 행과 열 숫자가 같고, 행렬식(determinant)이 0이 아닌 경우에만 계산할 수 있다.
$$AA^{-1} = A^{-1}A = I$$
``` python
X = np.arange(9).reshape(3, 3)
np.linalg.inv(X)
```

* 역행렬을 계산할 수 없을 때는 **유사역행렬(pseudo-inverse)** 또는 **무어-펜로즈(Moore-Penrose)** 역행렬 $A^{+}$을 이용한다.
$$n \geq m \; 인 경우 \quad A^{+} = (A^{T}A)^{-1}A^{T} \\
n \leq m \; 인 경우 \quad A^{+} = A^{T}(AA^{T})^{-1}$$
``` python
X = np.arange(6).reshape(3, 2)
np.linalg.pinv(X)
```

<br />

#### 연립방정식 풀기
* ```np.linalg.pinv```를 이용하여 연립방정식의 해를 구할 수 있다.
$$
a_{11}x{1} + a_{12}x_{2} + \cdots + a_{1m}x_{m} = b_{1} \\
a_{12}x{1} + a_{22}x_{2} + \cdots + a_{2m}x_{m} = b_{1} \\
\vdots \\
a_{n1}x{1} + a_{n2}x_{2} + \cdots + a_{nm}x_{m} = b_{1}
$$
를 행렬로 표현하면
$$
Ax = b
$$
이고, $n \leq m$이면 무어-펜로즈 역행렬을 이용하여 해를 하나 구할 수 있다.
$$ x \; = \; A^{+}b \; = \; A^{T}(AA^{T})^{-1}b$$

<br />

#### 선형회귀분석
* ```np.linalg.pinv```를 이용하여 데이터를 선형모델(linear model)로 해석하는 **선형회귀식**을 찾을 수 있다.
$$
\begin{bmatrix}
\cdots & x_{1} & \cdots \\
\cdots & x_{2} & \cdots \\
& \vdots & \\
\cdots & x_{n} & \cdots \\
\end{bmatrix}
\begin{bmatrix}\beta_{1} \\ \beta_{2} \\ \vdots \\ \beta_{m} \end{bmatrix} = 
\begin{bmatrix}y_{1} \\ y_{2} \\ \vdots \\ y_{m} \end{bmatrix}
$$
이며, 선형회귀분석은 연립방정식과 달리 행이 더 크므로 방정식을 푸는건 불가능하다. ($n \geq m$)
따라서 무어-펜로즈 역행렬을 이용하여 $y$에 근접하는 $\hat{y}$를 찾을 수 있다.
$$
X\beta = \hat{y} \approx y \\
\beta \; = \; X^{+}y \; = \; (X^{T}X)^{-1}X^{T}y
$$
``` python
# Scikit Learn을 이용한 회귀분석
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
y_test = model.predict(x_test)

# Moore-Penrose 역행렬
beta = np.linalg.pinv(X) @ y
y_test = np.append(x_test) @ beta
```