---
date: 2025-10-16
authors:
  - sem_k
categories:
  - RL
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

## Results

## Advanced problem

## Relevant links

- [Sutton and Barto book's site](http://incompleteideas.net/book/the-book-2nd.html)
- [Other problem's implementation in Python](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter04/car_rental.py)
