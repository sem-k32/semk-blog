---
date: 2025-10-16
authors:
  - sem_k
categories:
  - RL
readtime: 10
---

# Car rental problem

I'd like to present to you the problem I stumbled upon in the R. Sutton and A. Barto [book](http://incompleteideas.net/book/the-book-2nd.html) about reinforcement learning and really liked. You manage two locations for car rental and try to maximize your profit by moving available cars around locations. It can be seen as a very close model to the real-world situation. It is also very convenient to show how to approach and solve such problems in terms of the RL.

<!-- more -->

Original text:

!!! quote "Car rental problem"

    Jack manages two locations for a nationwide car rental company. Each day, some number of customers arrive at each location to rent cars.
    If Jack has a car available, he rents it out and is credited 10 dollars by the national company.
    If he is out of cars at that location, then the business is lost. Cars become available for
    renting the day after they are returned. To help ensure that cars are available where
    they are needed, Jack can move them between the two locations overnight, at a cost of
    2 dollars per car moved. We assume that the number of cars requested and returned at each
    location are Poisson($\lambda$) random variables. Suppose $\lambda$ is 3 and 4 for rental requests at the first and second locations and 3 and 2 for returns. To simplify the problem slightly, we assume that there can be no more than 20 cars at each location (any additional cars are returned to the nationwide company, and thus disappear from the problem) and a
    maximum of five cars can be moved from one location to the other in one night. We take
    the discount rate to be $\gamma$ = 0.9 and formulate this as a continuing finite MDP, where
    the time steps are days, the state is the number of cars at each location at the end of
    the day, and the actions are the net numbers of cars moved between the two locations
    overnight.

The notebook for the article is available at [![Python](https://img.shields.io/badge/Github-8A2BE2?logo=github)]().

## Mathematical formulation

Let's rewrite the problem in RL terms. Then we will use the *value iteration* to find optimal value function $V(s)$ and policy $\pi(s)$.

Our state here is $s = (n_0, n_1)$ for cars at location 0 and location 1 at the end of some day. Then we take action $a \in -5, \ldots, 5$ and at the start of the next day we have $(n_0 - a, n_1 + a)$. Note that some actions are pointless or incorrect for some states, e.g. $s = (2, 1)$ and $a = 5$. For these cases we will add a terminal dummy state with zero reward. All pointless/incorrect $(s, a)$ pairs will transition to it with probability one.

After, we have some realization of poisson rental requests $(\text{req}_0, \text{req}_1)$ and returns $(\text{ret}_0, \text{ret}_1)$ for each location. Note that the reasonable values for rental requests are $0, \ldots, n_0 - a$ and $0, \ldots, n_1 + a$. That's why we can truncate poisson random variables to their maximum values. The probability of all values greater adds to the probability of the maximum value. Having the request values, the reasonable values for returns are $0, \ldots, 20 - (n_0 - a - \text{req}_0)$ and $0, \ldots, 20 - (n_1 + a - \text{ret}_1)$. So returns are truncated as well. The final state is $s' = (n_0 - a - \text{req}_0 + \text{ret}_0, n_1 + a - \text{req}_1 + \text{ret}_1) $. To compute probability of such a transition we use independence of the requests and returns so their mutual probability is a product of individual probabilities. Overall, we can compute $p(s' | s, a)$ by fixing $(s, a)$ and iterating over reasonable realizations of $(\text{req}_0, \text{req}_1)$ and $(\text{ret}_0, \text{ret}_1)$, adding $\mathbb{P}(\text{req}_0, \text{req}_1, \text{ret}_0, \text{ret}_1)$ to corresponding $s'$.

Reward $r(s, a, s')$ here is a random variable for fixed arguments because it depends on the random requests and returns. To use it in the dynamical programming we need to average it over valid $(\text{req}_0, \text{req}_1, \text{ret}_0, \text{ret}_1)$ for given $(s, a, s')$. We can do it during the same iteration when computing $p(s' | s, a)$. We init $r(s, a, s')$ with zeros. Then, for fixed $(s, a)$ and realization of the requests and returns the reward is uniquely determined. We multiply it with the probabilty of such realization and add to $r(s, a, s')$. After all iterations we have completed $p(s' | s, a)$ and averaged $r(s, a, s')$ but unnormed. The normalization constant for averaging is exactly $p(s' | s, a)$.

To compute optimal value function and policy we will use value iteration algorithm which basically solves the following equation iteratively:

$$
    v(s) = \max_a \sum_{s',r} p(s',r|s,a) \left[ r + \gamma v(s') \right].
$$

The optimal policy is then obtained as

$$
    \pi(s) = \arg\max_a \sum_{s',r} p(s',r|s,a) \left[ r + \gamma v(s') \right].
$$

## Implementation details

!!! note
    The source code is available at [`components.py`](components.py). Here we outline the main parts of the computations.

We init five dimensional tensors for probability and reward:

``` py linenums="1"
  n_actions = 2 * max_cars_moved + 1
  # dims go (n0, n1, a, n0', n1')
  reward_t = np.zeros(
      [max_cars_location + 1] * 2 + [n_actions] + [max_cars_location + 1] * 2
  )
  # init p(s' | s, a), dims go (n0, n1, a, n0', n1')
  prob_t = np.zeros(
      [max_cars_location + 1] * 2 + [n_actions] + [max_cars_location + 1] * 2
  )
```

We then iterate over all starting states and actions, then over all reasonable rental requests. In the inner-most cycle the reasonable return requests are easily determined as well as their probabilities.

``` py linenums="1"
  for (n0, n1) in product(
    range(0, conf.max_cars_location + 1), 
    range(0, conf.max_cars_location + 1)
  ):
    for cars_moved in range(-conf.max_cars_moved, conf.max_cars_moved + 1):
      # num of cars at the start of the day
      n0_start = n0 - cars_moved
      n1_start = n1 + cars_moved
      for (req0, req1) in product(range(0, n0_start + 1), range(0, n1_start + 1)):
        # cars left after requests
        cars_left0 = n0_start - req0
        cars_left1 = n1_start - req1
        # maximum number of cars that can be returned
        max_returns0 = conf.max_cars_location - cars_left0
        max_returns1 = conf.max_cars_location - cars_left1
        # probabilities for possible requests
        prob_req0 = truncate_distr_probs(lambda_req_0, n0_start)
        prob_req1 = truncate_distr_probs(lambda_req_0, n1_start)
```

We also compute probabilities of the truncated poisson variables for the retuns. Their mutual probability can be computed via `#!python np.outer`. The reward is constant in the inner-most loop because it depends only on the action and the rental requests. Finally, we add up probaility and mean reward to all possible states we transition to under current realizations of the poisson variables.

``` py linenums="1"
  # probabilities for possible returns
  prob_ret0 = truncate_distr_probs(lambda_ret_0, max_returns0)
  prob_ret1 = truncate_distr_probs(lambda_ret_1, max_returns1)
  # compute mutual probabilites for returns
  prob_ret = np.outer(prob_ret0, prob_ret1)
  # reward for current realization
  reward = (req0 + req1) * rent_price - \
                  np.abs(cars_moved) * conf.move_price

  # add a fraction to the expected reward for (s, a) = fixed
  # and s' - varied
  reward_t[
      n0, n1, act_indx, cars_left0:, cars_left1:
  ] += reward * (prob_req0[req0] * prob_req1[req1] * prob_ret)
  # add a fraction to the probability for (s, a) = fixed and s' - varied
  prob_t[
      n0, n1, act_indx, cars_left0:, cars_left1:
  ] += prob_req0[req0] * prob_req1[req1] * prob_ret
```

After all loops we normalize average reward as was said:

``` py linenums="1"
reward_t[prob_t != 0.] = reward_t[prob_t != 0.] / prob_t[prob_t != 0.]
```

!!! question "Can we do without loops?"
    Here we use nested loops as the most intuitive approach. We can't just use vectorized operations because these operations highly depend on the inital state, action and requests. Only in the inner-most loop we vectorize operations with returns. Can you think of a faster solution?

In the value iteration we flatten state dims so our tensors will have three dimensions $(s, a, s')$. It is convenient to use `#!python np.einsum` to compute mean values over probability transition function:

``` py
from scipy import linalg


def value_iteration(
    reward_t: np.ndarray,
    prob_t: np.ndarray,
    gamma: float,
    v0: np.ndarray,
    tol: float = 1e-3,
    max_steps: int = 1000
) -> np.ndarray:
    # compute avereged over s' reward r(s, a, s')
    reward_expect = np.einsum(
        "iaj,iaj->ia",
        prob_t, reward_t
    )
    v_prev = v0
    for _ in range(max_steps):
        v_expect = np.einsum(
            "iaj,j->ia",
            prob_t, v_prev
        )
        v = np.max(reward_expect + gamma * v_expect, axis=-1)

        delta = linalg.norm(v - v_prev, np.inf)
        if delta < tol:
            break
        v_prev = v
    
    return v
```

## Results

The found optimal value function and optimal policy are illustrated below. The results match with [book's illustrations](https://raw.githubusercontent.com/ShangtongZhang/reinforcement-learning-an-introduction/master/images/figure_4_2.png).

```plotly
{"file_path": "posts/car_rental/sutton/assets/value.json"}
```

```plotly
{"file_path": "posts/car_rental/sutton/assets/policy.json"}
```

In the optimal policy we see the reasonable tendency to balance number of cars in the locations. Because the rental request rate in the location 1 is higher we are more willing to move cars there and less willing to move it in other direction.

## Advanced problem

The book has also modified version of the task.

!!! quote Advance car rental problem
    One of Jack’s employees at the first location
    rides a bus home each night and lives near the second location. She is happy to shuttle
    one car to the second location for free. Each additional car still costs 2 dollars, as do all cars
    moved in the other direction. In addition, Jack has limited parking space at each location.
    If more than 10 cars are kept overnight at a location (after any moving of cars), then an
    additional cost of 4 dollars must be incurred to use a second parking lot (independent of how
    many cars are kept there).

To solve it we only need to take into account the new details when computing $r(s, a, s')$. All the rest stay the same. Updated computation of the tensors is located at [`components_adv.py`](./components_adv.py). The results are illustrated below.

```plotly
{"file_path": "posts/car_rental/sutton/assets/value_adv.json"}
```

```plotly
{"file_path": "posts/car_rental/sutton/assets/policy_adv.json"}
```

Here we see how we adapt to the new environment. For example, we use the free ride to balance number of cars at the locations when previously it was too costly. We have also become very selective in the number of cars moved to pay for additional parking only in one location. The value function has become unlinear.

## Relevant links

- [Sutton and Barto book's site](http://incompleteideas.net/book/the-book-2nd.html)
- [Other problem's implementation in Python](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter04/car_rental.py)
