"""Full data gathering, learning and planning pipeline
"""
from ndr.structs import Anti
from envs.ndr_blocks import NDRBlocksEnv, noiseoutcome
from envs.pybullet_blocks import PybulletBlocksEnv
from ndr.learn import run_main_search
from ndr.planning import find_policy
from ndr.utils import run_policy
from collections import defaultdict
from termcolor import colored
import pickle
import os
import numpy as np


def collect_training_data(env, outfile):
    """Load or generate training data
    """
    if os.path.exists(outfile):
        with open(outfile, 'rb') as f:
            transition_dataset = pickle.load(f)
        num_transitions = sum(len(v) for v in transition_dataset.values())
        print("Loaded {} transitions for {} actions.".format(num_transitions, 
            len(transition_dataset)))
    else:
        print("Collecting transition data... ", end='')
        transition_dataset = collect_transition_dataset(env)
        num_transitions = sum(len(v) for v in transition_dataset.values())
        print("collected {} transitions for {} actions.".format(num_transitions, 
            len(transition_dataset)))
        with open(outfile, 'wb') as f:
            pickle.dump(transition_dataset, f)
        print("Dumped dataset to {}.".format(outfile))
    return transition_dataset

def collect_transition_dataset(env, num_trials=100, num_transitions_per_problem=10, 
                               policy=None, actions="all"):
    """Collect transitions (state, action, effect) for the given actions
    Make sure that no more than 50% of outcomes per action are null.
    """
    total_counts = defaultdict(int)
    num_no_effects = defaultdict(int)

    if policy is None:
        policy = lambda s : env.action_space.sample()
    transitions = defaultdict(list)
    for trial in range(num_trials):
        done = True
        for _ in range(num_transitions_per_problem):
            if done:
                obs, _ = env.reset()
            action = policy(obs)
            next_obs, _, done, _ = env.step(action)
            effects = construct_effects(obs, next_obs)
            null_effect = len(effects) == 0 or noiseoutcome() in effects
            keep_transition = (actions == "all" or action.predicate in actions) and \
                (not null_effect or (num_no_effects[action.predicate] < \
                    total_counts[action.predicate]/2.+1))
            if keep_transition:
                total_counts[action.predicate] += 1
                if null_effect:
                    num_no_effects[action.predicate] += 1
                transition = (obs, action, effects)
                transitions[action.predicate].append(transition)
            obs = next_obs

    return transitions

def construct_effects(obs, next_obs):
    """Convert a next observation into effects
    """
    # This is just for debugging environments where noise outcomes are simulated
    if noiseoutcome() in next_obs:
        return { noiseoutcome() }
    effects = set()
    for lit in next_obs - obs:
        effects.add(lit)
    for lit in obs - next_obs:
        effects.add(Anti(lit))
    return effects

def learn_rule_set(training_data, outfile):
    """Main learning step
    """
    if os.path.exists(outfile):
        with open(outfile, 'rb') as f:
            rules = pickle.load(f)
        num_rules = sum(len(v) for v in rules.values())
        print("Loaded {} rules for {} actions.".format(num_rules, len(rules)))
    else:
        print("Learning rules... ")
        rules = run_main_search(training_data)
        num_rules = sum(len(v) for v in rules.values())
        print("Loaded {} rules for {} actions.".format(num_rules, len(rules)))
        with open(outfile, 'wb') as f:
            pickle.dump(rules, f)
        print("Dumped rules to {}.".format(outfile))
    print_rule_set(rules)
    return rules

def print_rule_set(rule_set):
    for action_predicate in sorted(rule_set):
        print(colored(action_predicate, attrs=['bold']))
        for rule in rule_set[action_predicate]:
            print(rule)

def run_test_suite(test_env_cls, test_outfile, num_problems=10, seed_start=10000,
                   num_trials_per_problem=1, render=True, verbose=False):
    all_returns = []
    for seed in range(seed_start, seed_start+num_problems):
        seed_returns = []
        for trial in range(num_trials_per_problem):
            env = test_env_cls()
            env.seed(seed)
            initial_state, debug_info = env.reset()
            goal = debug_info["goal"]
            policy = find_policy("ff_replan", initial_state, goal, env.operators, env.action_space, env.observation_space)
            total_returns = 0
            outdir = '/tmp/ndrblocks{}_{}/'.format(seed, trial)
            if render:
                os.makedirs(outdir, exist_ok=True)
            returns = run_policy(env, policy, verbose=verbose, render=render, check_reward=False, 
                outdir=outdir)
            seed_returns.append(returns)
        all_returns.append(seed_returns)
    print("Average returns:", np.mean(all_returns))
    return all_returns


def main():
    seed = 0

    training_env = NDRBlocksEnv()
    training_env.seed(seed)
    data_outfile = "data/{}_training_data.pkl".format(training_env.__class__.__name__)
    training_data = collect_training_data(training_env, data_outfile)

    rule_set_outfile = "data/{}_rule_set.pkl".format(training_env.__class__.__name__)
    rule_set = learn_rule_set(training_data, rule_set_outfile)

    test_env_cls = NDRBlocksEnv
    test_outfile = "data/{}_test_results.pkl".format(test_env_cls.__name__)
    test_results = run_test_suite(test_env_cls, test_outfile, render=False)

    print("Test results:")
    print(test_results)


if __name__ == "__main__":
    main()
