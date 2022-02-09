동일한 구조를 갖지만 네트워크 파라미터와 다르게 학습되는 인코더와 디코더가 쌓여있는 구조

* n개의 단어가 어떻게 인코더에서 한번에 처리가 되는지?
* 인코더와 디코더 사이에 어떤 정보를 주고 받는지?
* 디코더가 어떻게 generate 할 수 있는지?

Self-Attention은 어떤 역할을 하냐?
- n개의 단어가 주어지면 n개의 벡터를 찾아준다
- $x_1$부터 $x_2$까지 n개의 단어가 주어지고, $z_1$부터 $z_n$까지 n개의 찾아야 하는 벡터가 있을 때, $i$번째 벡터를 $z_i$로 바꿀 때 나머지 $n-1$개의 $x$ 벡터를 같이 고려한다
- 서로 단어에 대해서 dependency가 있다.

Encoding 과정
1. 입력한 단어마다 ```query vector```, ```key vector```, ```value vector```를 만들게 된다.
2. ```score```를 구한다.
    - $i$번째 단어에 대한 ```score```는 내가 인코딩을 하고자 하는 단어의 query vector와 나머지 n개의 단어에 대한 key vector를 구하여 내적한다. 즉 이 두 벡터가 얼마나 align이 잘 되어있는지 보고, i번째 단어가 나머지 단어들과 얼마나 연관이 있고 유사도가 있는지 정하게 된다.
3. ```score```를 normalize 및 softmax한다.
4. 이렇게 나온 ```score``` 값과 ```value vector```를 곱하여 ```encoding vector```를 구한다.

Transformer는 입력이 고정되면 출력이 고정되는 것이 아니라, 입력이 고정되도, 내 옆에 있는 다른 단어들에 따라서 출력이 바뀐다. 그래서 훨씬 더 많이 표현할 수 있고, flexible한 네트워크를 표현할 수 있다.

하지만 RNN과는 다르게 입력된 단어들을 한 번에 처리해야 되기 때문에 한계가 존재한다.
(RNN은 n개의 단어가 입력되면 느리더라도 n번 돌리면 출력이 나오기 때문)

**Multi-headed attention(MHA)** 는 한 개의 ```embedded vector```에 대해서 여러개의 ```query vector```, ```key vector```, ```value vector```를 만든다.
- 한 개의 ```embedded vector```에 대해서 여러개의 ```encoded vector```가 나오게 된다.
- 입력과 출력의 차원을 맞춰줘야 한다.

각각의 ```embedded vector```는 sequential 하게 입력 되었지만 independent 하기 때문에, positional encoding이 필요하다.

