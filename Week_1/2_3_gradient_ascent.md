## 미분
* 미분이란 **변수의 움직임에 따른 함수값의 변화를 측정하기 위한 도구**이다.
* 미분은 함수 $f$의 주어진 점 $(x, f(x))$에서의 접선의 기울기를 구한다.
``` python
import sympy as sym
from sympy.abc import x

symm.diff(sym.poly(x**2 + 2*x + 3), x)
# Poly(2*x + 2, x, domain='ZZ')
```
* 한 점에서 접선의 기울기를 알면 어느 방향으로 점을 움직여야 함수값이 **증가**하는지/**감소**하는지 알 수 있다.
* **증가**시키고 싶다면 미분값을 **더하고**, **감소**시키고 싶다면 미분값을 **뺀다.**

<br />

## 경사하강법
* 미분값을 빼면 경사하강법(gradient descent)이라 하며, 함수의 **극소값**의 위치를 구할 때 사용한다.
``` python
def func(val):
  fun = sym.poly(x**2 + 2*x + 3)
  return fun.subs(x, val), fun

def func_gradient(fun, val):
  _, function = fun(val)
  diff = sym.diff(function, x)
  return diff.subs(x, val), diff

def gradient_descent(fun, init_point, lr_rate=1e-2, epsilon=1e-5):
  cnt = 0
  val = init_point
  diff, _ = func_gradient(fun, init_point)
  while np.abs(diff) > epsilon:
    val = val - lr_rate * diff
    diff, _ = func_gradient(fun, val)
    cnt += 1

  print("함수: {}, 연산횟수: {}, 최소점: ({}, {})".format(fun(val)[1], cnt, val, fun(val)[0]))

gradient_descent(fun=func, init_point=np.random.uniform(-2, 2))
# 함수: Poly(x**2 + 2*x + 3, x, domain='ZZ'), 연산횟수: 636, 최소점: (-0.999995047967832, 2.000000000002452)
```

<br />

##### 벡터가 입력인 다변수 함수의 경우엔 편미분을 사용한다.
$$
\partial_{x_{i}}f(x) = \lim_{h \to 0}{{f(x+he_{i}) - f(x)} \over h}
$$
예)
$f(x, y) = x^2 + 2xy + 3 + cos(x + 2y)$
$\partial_{x}f(x, y) = 2x + 2y - sin(x + 2y)$
``` python
import sympy as sym
from sympy.abc import x, y

sym.diff(sym.poly(x**2 + 2*x*y + 3) + sym.cos(x + 2*y), x)
# 2*x + 2*y - sin(x + 2*y)
```

각 변수 별로 편미분을 계산한 **그래디언트(gradient) 벡터**를 이용하여 경사하강법에 사용할 수 있다.
$$\nabla f = (\partial_{x_{1}}f, \; \partial_{x_{2}}f, \; \cdots, \; \partial_{x_{d}}f)$$
예)
$f(x,y)=x^2+2y^2$
$- \nabla f = -(2x, 4y)$

<br />

## 경사하강법으로 선형회귀 계수 구하기
* 선형회귀의 목적식은 $\begin{Vmatrix}y-X\beta\end{Vmatrix}_{2}$ 이고, 이를 최소화하는 $\beta$를 찾아야 하므로 다음과 같은 그레디언트 벡터를 구해야 한다.
$$
\nabla_{\beta}\begin{Vmatrix}y-X\beta\end{Vmatrix}_{2} = (\partial_{\beta_{1}}\begin{Vmatrix}y-X\beta\end{Vmatrix}_{2}, \; \cdots, \; \partial_{\beta_{d}}\begin{Vmatrix}y-X\beta\end{Vmatrix}_{2})
$$
이때, 
$$
\partial_{\beta_{k}}\begin{Vmatrix}y-X\beta\end{Vmatrix}_{2} \; = \; 
\partial_{\beta_{k}}{\left\{ {1 \over n} \sum_{i=1}^n \left(y_{i} - \sum_{j=1}^d X_{ij}\beta_{j}\right)^2\right\}}^{1 \over 2} \; = \; 
-{{X_{k}^T(y-X\beta)} \over {n\begin{Vmatrix}y-X\beta\end{Vmatrix}_{2}}}
$$
이므로,
$$
\nabla_{\beta}\begin{Vmatrix}y-X\beta\end{Vmatrix}_{2} \; = \;
-{{X^T(y-X\beta)} \over {n\begin{Vmatrix}y-X\beta\end{Vmatrix}_{2}}}
$$
이다. 이 때, $\begin{Vmatrix}y-X\beta\end{Vmatrix}_{2}$ 대신 $\begin{Vmatrix}y-X\beta\end{Vmatrix}_{2}^{2}$ 최소화하면 아래와 같이 식이 더 간단해진다.
$$
\nabla_{\beta}\begin{Vmatrix}y-X\beta\end{Vmatrix}_{2} \; = \;
-{2 \over n}X^T(y-X\beta)
$$
이제 목적식을 최소화하는 $\beta$를 구하는 경사하강법 알고리즘은 다음과 같다.
$$\beta^{(t+1)} = \beta^{(t)} + {2\lambda \over n}X^T(y - X\beta^{(t)})$$

``` python
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

beta_gd = [0., 0., 0.]  # [1, 2, 3] 이 정답
X_ = np.array([np.append(x, [1]) for x in X])  # intercept 항 추가

for t in range(5000):
  error = y - X_ @ beta_gd
  grad = -np.transpose(X_) @ error
  beta_gd = beta_gd - 0.01 * grad

beta_gd  # [1.00000367, 1.99999949, 2.99999516]
```
* 경사하강법은 미분가능하고 볼록한 함수에 대하여 적절한 학습률과 학습횟수를 선택했을 때 수렴이 보장되어 있다.
* 하지만, 비선형회귀 문제의 경우 목적식이 볼록하지 않을 수 있으므로 수렴이 항상 보장되지는 않는다.

<br />

## 확률적 경사하강법
* 확률적 경사하강법(stochastic gradient descent)은 모든 데이터를 사용해서 업데이트 하는 것이 아닌 데이터 한개 또는 일부 활용하여 업데이트한다.
* 볼록이 아닌(non-convex) 목적식은 SGD를 통하여 최적화할 수 있다.
* SGD는 데이터의 일부를 가지고 패러미터를 업데이트하기 때문에 연산자원을 좀 더 효율적으로 활용하는데 도움이 된다.

<br />

* 경사하강법은 전체데이터 $\mathcal{D}=(X,y)$ 를 가지고 목적식의 그레디언트 벡터인 $\nabla_{\theta}L(\mathcal{D}, \theta)$ 를 계산한다.
* SGD는 미니배치 $\mathcal{D}_{(b)}=(X_{(b)},y_{(b)}) \subset \mathcal{D}$ 를 가지고 그레디언트 벡터를 계산한다.
* 미니배치는 확률적으로 선택하므로 목적식 모양이 바뀌게 된다.
* SGD는 볼록이 아닌 목적식에도 사용 가능하므로 경사하강법보다 머신러닝 학습에 더 효율적이다.