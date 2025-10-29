"""Utils to get rental problem representation in terms of p(s'|s, a) and r(s, a, s')
    and dynamic programming algorithms.
"""
from itertools import product
# lib for functional programming
from pipe import select
from omegaconf import DictConfig

from scipy import stats
from scipy import linalg
import numpy as np

from rich.progress import track


def truncate_distr_probs(distr: stats.rv_discrete, max_val: int) -> np.ndarray:
    """Truncates distribution of discrete variable to max_val. Applied here for 
        poisson distr.
    """
    probs = np.array(list(
        range(0, max_val + 1) |
        select(distr.pmf)
    ))
    probs[-1] += distr.sf(max_val)
    return probs


def build_reward_prob_tensors(conf: DictConfig) -> tuple:
    """Constructs p(s'|s, a) and averaged r(s, a, s') for rental problem.
        We add dummy state to state space to be redirected to for
        incorrect or pointless (s, a) pairs.

    Args:
        conf (DictConfig): rental problem config

    Returns:
        tuple(np.ndarray): (r(s, a, s'), p(s'|s, a))
    """
    n_states = (conf.max_cars_location + 1) ** 2
    n_actions = 2 * conf.max_cars_moved + 1
    # mapping from num of cars moved to action's nonnegative index
    act_to_indx = {a:i for i, a in enumerate(range(-conf.max_cars_moved, conf.max_cars_moved + 1))}

    # init r(s, a, s')
    reward_t = np.zeros(
        [conf.max_cars_location + 1] * 2 + [n_actions] + [conf.max_cars_location + 1] * 2
    )
    # init p(s' | s, a), dims go (s, a, s')
    prob_t = np.zeros(
        [conf.max_cars_location + 1] * 2 + [n_actions] + [conf.max_cars_location + 1] * 2
    )

    # iterate over s
    for (n0, n1) in track(
        product(
            range(0, conf.max_cars_location + 1), range(0, conf.max_cars_location + 1)
        ),
        description="Computing reward and prob tensors",
        total=(conf.max_cars_location + 1)**2
    ):
        # cars moved from 0 location to 1 location
        for cars_moved in range(-conf.max_cars_moved, conf.max_cars_moved + 1):
            # incorrect or pointless (s, a) are skipped for now
            if (n0 - cars_moved < 0 or n1 + cars_moved < 0) or\
                (n0 - cars_moved > conf.max_cars_location) or\
                (n1 + cars_moved > conf.max_cars_location):
                continue

            act_indx = act_to_indx[cars_moved]
            # num of cars at the start of the day
            n0_start = n0 - cars_moved
            n1_start = n1 + cars_moved
            # probabilities for possible requests
            prob_req0 = truncate_distr_probs(stats.poisson(conf.request[0]), n0_start)
            prob_req1 = truncate_distr_probs(stats.poisson(conf.request[1]), n1_start)
            # iterate over all possible requests for locations
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
                prob_ret0 = truncate_distr_probs(stats.poisson(conf.ret[0]), max_returns0)
                prob_ret1 = truncate_distr_probs(stats.poisson(conf.ret[1]), max_returns1)
                # compute mutual probabilites
                prob_ret = np.outer(prob_ret0, prob_ret1)
                # reward for current realization
                reward = (req0 + req1) * conf.rent_price - \
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

    # compute mean r(s, a, s') by norming on probability of p(s' | s, a)
    # do it only where p(s' | s, a) != 0
    reward_t[prob_t != 0.] = reward_t[prob_t != 0.] / prob_t[prob_t != 0.]

    # we now flatten state space and add dummy state for incorrect/pointless (s, a)
    def flat_and_pad(x: np.ndarray):
        x = x.reshape([n_states, n_actions, n_states])
        x = np.pad(x, ((0, 1), (0, 0), (0, 1)), mode="constant", constant_values=0.)
        return x
    reward_t, prob_t = list([reward_t, prob_t] | select(flat_and_pad))
    # dummy state is terminal
    prob_t[-1, :, -1] = 1.

    # redirect all incorrect/pointless (s, a) to dummy state
    for (n0, n1) in track(
        product(
            range(0, conf.max_cars_location + 1), range(0, conf.max_cars_location + 1)
        ),
        description="Computing dummy state",
        total=(conf.max_cars_location + 1)**2
    ):
        # recompute state in flattened state space
        s = n0 * (conf.max_cars_location + 1) + n1
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
    """ Value iteration algorithm for flattened state space
    """
    # compute avereged over s' reward r(s, a, s')
    reward_expect = np.einsum(
        "iaj,iaj->ia",
        prob_t, reward_t
    )
    v_prev = v0
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

    print("Max steps reached, current approximation error: ", delta)
    return v


def get_optimal_policy(
    reward_t: np.ndarray,
    prob_t: np.ndarray,
    gamma: float,
    v: np.ndarray,
) -> np.ndarray:
    """ Computes greedy policy from the given value function
    """
    reward_expect = np.einsum(
        "iaj,iaj->ia",
        prob_t, reward_t
    )
    v_expect = np.einsum(
        "iaj,j->ia",
        prob_t, v
    )
    policy = np.argmax(reward_expect + gamma * v_expect, axis=-1)
    return policy
