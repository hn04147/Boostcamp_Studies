#### 벡터란
* 벡터란 숫자를 원소로 가지는 ```list``` 또는 ```array```이다.
* 벡터는 공간에서 한 점을 나타낸다.
* 벡터는 원점으로부터 상대적 위치를 표현한다.
$$
\underset{열벡터}{X} = \begin{bmatrix}
X_{1} \\
X_{2} \\
\vdots \\
X_{d} \\
\end{bmatrix},\quad
\underset{행벡터}{X^{T}} = \begin{bmatrix}
X_{1},\; X_{2},\; \cdots,\; X_{3} \\
\end{bmatrix}
$$

#### 벡터의 노름(norm)
* 벡터의 노름(norm)은 **원점에서부터의 거리**를 말한다.
* $L_{1}$ 노름은 각 성분의 **변화량의 절대값**을 모두 더함 - **맨해튼 거리**
* $L_{2}$ 노름은 피타고라스 정리를 이용하여 거리를 구함 - **유클리드 거리**

$$
X = \begin{bmatrix}
X_{1} \\
X_{2} \\
\vdots \\
X_{d} \\
\end{bmatrix} \quad
\begin{Vmatrix} X \end{Vmatrix}_{1}=\sum_{i=1}^d \begin{vmatrix} X_{i} \end{vmatrix} \quad
\begin{Vmatrix} X \end{Vmatrix}_{2}=\sqrt{\sum_{i=1}^d {\begin{vmatrix} {X_{i}} \end{vmatrix}}^2}
$$

``` python
def l1_norm(x):
  return np.sum(np.abs(x))

def l2_norm(x):
  return np.sqrt(np.sum(x * x))
```

#### 두 벡터 사이의 각도
$$
cos \theta = {{\begin{Vmatrix} x \end{Vmatrix}_{2}^2 + \begin{Vmatrix} y \end{Vmatrix}_{2}^2 - \begin{Vmatrix} x-y \end{Vmatrix}_{2}^2} \over {2 \begin{Vmatrix} x \end{Vmatrix}_{2} \begin{Vmatrix} y \end{Vmatrix}_{2}}}
$$

* 내적
$$
\left\langle x, \, y \right\rangle = \sum_{i=1}^d x_{i} y_{i} = \begin{Vmatrix} x \end{Vmatrix}_{2} \begin{Vmatrix} y \end{Vmatrix}_{2} cos\theta
$$