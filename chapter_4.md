# [ch 4. Numerical Computation](https://www.deeplearningbook.org/contents/numerical.html)

### 4.1 Overflow and Underflow
 - **underflow**란 0에 가까운 숫자가 반올림으로 인해 0이 되었을 때 발생하는 numerical error의 한 형태이다
 - **overflow**란 숫자의 크기가 아주 커서 $inf$ 또는 $-inf$로 approximate 되었을 때 발생하는 numerical error의 한 형태이다.
 - 다음과 같은 형태로 $\operatorname{softmax}$를 출력 함수로 사용할 때, 
    $$
    \operatorname{softmax}(\boldsymbol{x})_{i}=\frac{\exp \left(x_{i}\right)}{\sum_{j=1}^{n} \exp \left(x_{j}\right)}
    $$
    exponetial을 계산하기 때문에 숫자가 너무 작아지는 underflow나 숫자가 너무 커지는 overflow가 발생할 수 있는데 이를 해결하기 위해 아주 간단하게 
    ```math
    \operatorname{softmax}(\boldsymbol{z})_i \ (z=x-\max _{i} x_{i}) 

    ```
    다음과 같이 처리하면 된다. 이와 유사한 numerical tricks은 [여기](https://www.deeplearningbook.org/slides/04_numerical.pdf)를 참고하면 된다.

### 4.2 Poor conditionaing
 - matrix에 관련된 calculation이 Deepleaning의 기본적이기 때문에 다룬 내용으로 가장 대표적인 문제인 matrix inversion에 다룬 부분이다.
 - 함수 $f$는 다음과 같고, $A$가 symmetric matrix라고 할때(교재에서는 언급하지 않았지만 eigenvalue decomposition을 해야함으로 symmetric한 square matrix여야 한다),
    $$
        f(\boldsymbol{x})=\boldsymbol{A}^{-1} \boldsymbol{x} \ ( \boldsymbol{A} \in \mathbb{R}^{n \times n} )
    $$
    이고, 다음 값을 **condition number**라고 부른다.

    $$
        \max _{i, j}\left|\frac{\lambda_{i}}{\lambda_{j}}\right|
    $$
    condition number가 너무 큰 경우 $A$의 inverse를 구할때 계산해야하는 범위가 넓어진다는 insight를 얻을 수 있다.
    또한, condition value가 작을 경우 계산이 안정적으로 수행될 수 있다는 것을 알 수 있다.

### 4.3 Gradient-Based Optimization 
 - machine learning문제에서 우리가 계산해야하는 값은 $\boldsymbol{w}$(weight)값인데 non-convex한 네트워크에서 $\boldsymbol{w}$값을 찾아나가야하는 문제이기 때문에 한번에 계산할 수 없다. 따라서 iterative하게 계산해 나가야한다. 이때 사용하는 방식이 Gradient 방식으로 계산해 나가는 방식이다
 - convex optimization 문제의 경우 수식과 constraint가 주어지고 이를 이용해 $\boldsymbol{w}$를 찾아가야하는 방식인 반면, machine learning 문제의 경우 데이터 셋이 주어지는것이 전부이다. 그렇기 때문에 내가 쓸 Cost function을 잘 디자인해야 한다.
 - Cost(Loss) function을 고른 순간부터는 optimization 문제가 된다. 이 Cost function을 바탕으로 W를 찾아가는 과정을 우리는 **Learning**이라고 부른다.  
 - 학습으로 나온 결과가 못봤던 데이터 셋(Validation or Test dataset)에 대해서도 좋은 결과가 나와야 하는데 이를 **Generalization**이라고 부른다.
 - **Saddle point**: 미분해서 0이 나오는 지점으로 어떤 방향으로는 convex하고 또 다른 방향으로는 concave해서 minimum도 maximum 아닌 희한한 점이다. 딥러닝에서는 차원이 높기 때문에 우리가 local minuimum이라고 부르는 지점은 보통 minimum이나 maximum이 아닌 saddle point이다. 즉 dimension이 너무 커서 optimization이 어렵다는 이야기를 하고싶기 때문에 교재에 넣어놓은 것이다.
 - **Learning rate**: 학습을 시킬 때 사용하는 hyperparameter로 기본적으로 아주 작게 잡아야 한다.
 - **Gradient**: vector indput이고 scalar output일 때, $f(\boldsymbol{w})$에 대해 $\boldsymbol{w}$로 미분한 것을 의미한다. 
 - **Jaccobian matrix**: vector indput이고 vector output일 때, $\boldsymbol{f}: \mathbb{R}^{m} \rightarrow \mathbb{R}^{n}$ 인 $f$에 대해 $\boldsymbol{J} \in \mathbb{R}^{n \times m}$ 인  $J_{i, j}=\frac{\partial}{\partial x_{i}} f(\boldsymbol{x})_{i}$ 를 의미한다. 
 - **Hassian matrix**: Jacoobian을 한번 더 미분한 matrix로 $\boldsymbol{H}(f)(\boldsymbol{x})_{i, j}=\frac{\partial^{2}}{\partial x_{i} \partial x_{j}} f(\boldsymbol{x})$ 이다. 또한, Hassian이 작다는 것은 curverture가 천천히 변한다는 의미이다 
 - 함수를 approximation 하는 방식으로 보통 linear approximation과 second-order approximation을 사용하는데 second-order approximation을 사용하기 위해서는 Hassian function이 필요하다. 일반적으로 세상에 있는 smooth한 function들은 어떤 값의 근처에서 second-order approximation을 하면 거의 에러가 없다는 assumption이 있다. 그래서 optimization동네에서는 second-order approximation을 하는 newton's method를 자주 사용했었는데, 딥러닝에서는 dimension이 너무 크기 때문에 전체 dimension에 대해 만족시키는것이 너무 어려워 잘 안된다. approximation이 안맞는 경우를 줄이기 위해서는 $\epsilon$값을 작게 만들어야 하는데, 그렇게 하면 optimization이 너무 느려진다. 따라서 Hassian을 사용하면 optimization이 잘 안되기 때문에 Gradient Descent를 그냥 사용한다. 
 - **Newton's method**: second-order로 그냥 approximation하는 것이 아니라 gradient방향으로 약간 보정해서 $\boldsymbol{x}^{*}=\boldsymbol{x}^{(0)}-\boldsymbol{H}(f)\left(\boldsymbol{x}^{(0)}\right)^{-1} \nabla_{\boldsymbol{x}} f\left(\boldsymbol{x}^{(0)}\right)$ 로 표현된다. 설명하면 한점 $\boldsymbol{x}$ 에서 $\nabla_{\boldsymbol{x}} f(\boldsymbol{x})$를 계산하여 $-\nabla_{\boldsymbol{x}} f(\boldsymbol{x})$ 방향으로 가면 되는데 Hassian을 이용해서 2차함수에 맞게끔 방향을 약간 보정한다. 식을 보면, 우리가 Newton's method를 사용하기 위해서는 matrix inverse를 해야하는데, 요즘 linear algebra가 워낙 발달해서 inverse에 대해 어떻게든 근사하고 계산양을 조금 줄일 수 있는 테크닉이 존재한다 하더라도 $\epsilon$을 정하기 어렵다거나, W가 계속 jumping하면서 학습이 잘 되지 않는등 여러가지 문제로 딥러닝에서는 잘 사용하지 않는다. 또 다른 이유는 GD(Gradient Descent)같은 경우 sample의 사이즈를 줄이는 minibatch를 사용해도 학습이 잘 되지만 hassian 같은 경우에는 복잡하기 때문에 sample 사이즈를 늘려주어야 한다. 그러다보면 여러가지 문제가 생겨서 잘 쓰지 않는다.
 - **Lipschitz continuous**: continuous function을 define하는 방식 중 하나로 미분값이 얼마 이상은 커지지 않는다는 것을 말해주는 이야기이다. 수식으로는 Lipschitz constant $\mathcal{L}$ : $\forall \boldsymbol{x}, \forall \boldsymbol{y},|f(\boldsymbol{x})-f(\boldsymbol{y})| \leq \mathcal{L}\|\boldsymbol{x}-\boldsymbol{y}\|_{2}$ 로 표현된다. 이것을 사용하는 이유는 그냥 이렇게 가정을 해서 theorytic한 연구를 하면서 증명을 할때 미분값이 infinite하지 않고 constant값으로 제한되었다는 property를 사용하기 위해서이다. 예를들어, Lipschitz를 사용하는 이유는 딥러닝 학습을 수행할 때 $\boldsymbol{w}$가 너무 커지면 학습이 잘 되지 않는다는 것은 emperical하게 알려진 사실인데, Lipschitz를 사용함으로써 $\boldsymbol{w}$값이 너무 커지지 않음을 증명할 수 있다.

### 4.4 Constrained Optimization
 - **Lagrangian**: contraint가 있는 문제를 푸는 것은 어렵기 때문에 contraint를 없애기 위해 사용하는 기법으로 KKT(Karush-Kuhn-Tucker) condition을 만족했을 때 Optimal하다.
