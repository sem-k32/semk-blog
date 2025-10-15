from itertools import product
from pipe import select
from omegaconf import DictConfig

from scipy import stats
from scipy import linalg
import numpy as np

from rich.progress import track

def truncated_poisson_probs(distr: stats.rv_discrete, max_val: int) -> np.ndarray:
    probs = np.array(list(
        range(0, max_val + 1) |
        select(lambda x: distr.pmf(x))
    ))
    probs[-1] += distr.sf(max_val)
    return probs


def reward_prob_tensor(conf: DictConfig) -> np.ndarray:
    n_states = (conf.max_cars_location + 1) ** 2
    n_actions = 2 * conf.max_cars_moved + 1
    act_to_indx = {a:i for i, a in enumerate(range(-conf.max_cars_moved, conf.max_cars_moved + 1))}
    indx_to_act = {i:a for i, a in enumerate(range(-conf.max_cars_moved, conf.max_cars_moved + 1))}

    # init r(s, a, s')
    reward_t = np.zeros([conf.max_cars_location + 1] * 2 + [n_actions] + [conf.max_cars_location + 1] * 2)
    # init p(s' | s, a), dims go (s, a, s')
    prob_t = np.zeros([conf.max_cars_location + 1] * 2 + [n_actions] + [conf.max_cars_location + 1] * 2)

    for (n0, n1) in track(
        product(
            range(0, conf.max_cars_location + 1), range(0, conf.max_cars_location + 1)
        ),
        description="Computing reward and prob tensors",
        total=(conf.max_cars_location + 1)**2
    ):
        # cars moved from 0 to 1
        for cars_moved in range(-conf.max_cars_moved, conf.max_cars_moved + 1):
            if (n0 - cars_moved < 0 or n1 + cars_moved < 0) or\
                (n0 - cars_moved > conf.max_cars_location) or\
                (n1 + cars_moved > conf.max_cars_location):
                continue

            act_indx = act_to_indx[cars_moved]
            # num of cars at the start of the day
            n0_start = n0 - cars_moved
            n1_start = n1 + cars_moved
            # probabilities for possible requests
            prob_req0 = truncated_poisson_probs(stats.poisson(conf.request[0]), n0_start)
            prob_req1 = truncated_poisson_probs(stats.poisson(conf.request[1]), n1_start)
            for (req0, req1) in product(
                range(0, n0_start + 1), range(0, n1_start + 1)
            ):
                # cars left after requests
                cars_left0 = n0_start - req0
                cars_left1 = n1_start - req1
                # maximum number of cars that can be returned
                max_returns0 = conf.max_cars_location - cars_left0
                max_returns1 = conf.max_cars_location - cars_left1
                # probabilities for possible returns
                prob_ret0 = truncated_poisson_probs(stats.poisson(conf.ret[0]), max_returns0)
                prob_ret1 = truncated_poisson_probs(stats.poisson(conf.ret[1]), max_returns1)
                # compute mutual probabilites
                prob_ret = np.outer(prob_ret0, prob_ret1)
                # reward for current realization of rand vars
                reward = (req0 + req1) * conf.rent_price - \
                                np.abs(cars_moved) * conf.move_price

                # add fraction to expected reward for s, a = fixed and s' - varied
                reward_t[
                    n0, n1, act_indx, cars_left0:, cars_left1:
                ] += reward * (prob_req0[req0] * prob_req1[req1] * prob_ret)
                # add fraction to the probability for s, a = fixed and s' - varied
                prob_t[
                    n0, n1, act_indx, cars_left0:, cars_left1:
                ] += prob_req0[req0] * prob_req1[req1] * prob_ret

    # compute mean r(s, a, s')
    reward_t[prob_t != 0.] = reward_t[prob_t != 0.] / prob_t[prob_t != 0.]
    
    # we must add dummy state for incorrect actions
    reward_t, prob_t = list(
        [reward_t, prob_t] |
        select(lambda t: t.reshape([n_states, n_actions, n_states])) |
        select(lambda t:
               np.pad(
                    t,
                    ((0, 1), (0, 0), (0, 1)),
                    mode="constant",
                    constant_values=0.
                )
        )
    )
    prob_t[-1, :, -1] = 1.
    for (n0, n1) in track(
        product(
            range(0, conf.max_cars_location + 1), range(0, conf.max_cars_location + 1)
        ),
        description="Computing dummy state",
        total=(conf.max_cars_location + 1)**2
    ):
        s = n0 * (conf.max_cars_location + 1) + n1
        # cars moved from 0 to 1
        for cars_moved in range(-conf.max_cars_moved, conf.max_cars_moved + 1):
            act_indx = act_to_indx[cars_moved]
            if (n0 - cars_moved < 0 or n1 + cars_moved < 0) or\
                (n0 - cars_moved > conf.max_cars_location) or\
                (n1 + cars_moved > conf.max_cars_location):
                    prob_t[s, act_indx, -1] = 1.

    return reward_t, prob_t


def value_iteration(
    reward_t: np.ndarray,
    prob_t: np.ndarray,
    gamma: float,
    v0: np.ndarray,
    tol: float = 1e-3,
    max_steps: int = 1000
) -> np.ndarray:
    v_prev = v0
    reward_expect = np.einsum(
        "iaj,iaj->ia",
        prob_t, reward_t
    )
    for _ in track(range(max_steps), "Value Iteration"):
        v_expect = np.einsum(
            "iaj,j->ia",
            prob_t, v_prev
        )
        v = np.max(reward_expect + gamma * v_expect, axis=-1)

        delta = linalg.norm(v - v_prev, np.inf)
        if delta < tol:
            return v
        v_prev = v
    
    print("Max steps reached, cur approximation error: ", delta)
    return v


def get_optimal_policy(
    reward_t: np.ndarray,
    prob_t: np.ndarray,
    gamma: float,
    v: np.ndarray,
) -> np.ndarray:
    max_cars_moved = (reward_t.shape[1] - 1) // 2
    reward_expect = np.einsum(
        "iaj,iaj->ia",
        prob_t, reward_t
    )
    v_expect = np.einsum(
        "iaj,j->ia",
        prob_t, v
    )
    policy = np.argmax(reward_expect + gamma * v_expect, axis=-1)
    # transform action index to action itself
    policy -= max_cars_moved
    return policy
