"""Main file for NDR learning
"""
from ndr.structs import Anti, NDR, ground_literal, LiteralConjunction
from ndr.inference import find_satisfying_assignments
from envs.ndr_blocks import NDRBlocksEnv, noiseoutcome
from collections import defaultdict
from termcolor import colored
import heapq as hq
import numpy as np


def collect_transition_dataset(num_problems, num_transitions_per_problem, policy=None):
    env = NDRBlocksEnv()
    assert num_problems <= env.num_problems
    if policy is None:
        policy = lambda s : env.action_space.sample()
    transitions = defaultdict(list)
    for problem_idx in range(num_problems):
        env.fix_problem_index(problem_idx)
        done = True
        for _ in range(num_transitions_per_problem):
            if done:
                obs, _ = env.reset()
            action = policy(obs)
            next_obs, _, done, _ = env.step(action)
            effects = construct_effects(obs, next_obs)
            transition = (obs, action, effects)
            obs = next_obs
            transitions[action.predicate].append(transition)
    return transitions

def construct_effects(obs, next_obs):
    effects = set()
    for lit in next_obs - obs:
        effects.add(lit)
    for lit in obs - next_obs:
        effects.add(Anti(lit))
    return effects

def get_variable_names(num_vars, start_at=0):
    return ["X{}".format(i) for i in range(start_at, start_at+num_vars)]

def find_assignments_for_ndr(ndr, state, action):
    kb = state | { action }
    assert action.predicate == ndr.action.predicate
    conds = [ndr.action] + list(ndr.preconditions)
    return find_satisfying_assignments(kb, conds)

def score_action_rule_set(action_rule_set, transitions_for_action, p_min=1e-6, alpha=0.5):
    # Get per-NDR penalties
    ndr_idx_to_pen = {}

    # Calculate penalty for number of literals
    for idx, selected_ndr in enumerate(action_rule_set):
        pen = 0
        preconds = selected_ndr.preconditions
        if isinstance(preconds, LiteralConjunction):
            pen += len(preconds.literals)
        else:
            pen += 1
        for _, outcome in selected_ndr.effects:
            if isinstance(outcome, LiteralConjunction):
                pen += len(outcome.literals)
            else:
                pen += 1
        ndr_idx_to_pen[idx] = pen

    # Calculate transition likelihoods per example and accumulate score
    score = 0.
    for (state, action, effects) in transitions_for_action:
        selected_ndr_idx = None
        for idx, ndr in enumerate(action_rule_set):
            assignments = find_assignments_for_ndr(ndr, state, action)
            if len(assignments) == 1:
                selected_ndr_idx = idx
                break
        assert selected_ndr_idx is not None, "At least the default NDR should be selected"
        selected_ndr = action_rule_set[selected_ndr_idx]
        pen = ndr_idx_to_pen[selected_ndr_idx]
        # Calculate transition likelihood
        transition_likelihood = 0.
        for prob, outcome in selected_ndr.effects:
            if outcome == noiseoutcome():
                # c.f. equation 3 in paper
                transition_likelihood += p_min * prob
            else:
                grounded_outcome = {ground_literal(lit, assignments[0]) for lit in outcome}
                if grounded_outcome == effects:
                    transition_likelihood += prob
        # Add to score
        score += np.log(transition_likelihood) - alpha * pen

    return score

def create_default_rule_set(transition_dataset):
    rule_set = {}
    total_score = 0.
    for action_predicate, transitions_for_action in transition_dataset.items():
        variable_names = get_variable_names(action_predicate.arity)
        lifted_action = action_predicate(*variable_names)
        ndr = NDR(action=lifted_action, preconditions=[], effects=[(1.0, noiseoutcome())])
        action_rule_set = [ndr]
        rule_set[action_predicate] = action_rule_set
        total_score += score_action_rule_set(action_rule_set, transitions_for_action)
    return total_score, rule_set

def get_search_operators(transition_dataset):
    # TODO
    return []

def run_greedy_search(transition_dataset, max_node_expansions=1000, rng=None):
    if rng is None:
        rng = np.random.RandomState(seed=0)

    search_operators = get_search_operators(transition_dataset)

    score, default_rule_set = create_default_rule_set(transition_dataset)
    best_rule_set = default_rule_set
    best_score = score

    print("Starting greedy search with initial score", score)

    for n in range(max_node_expansions):
        rule_set = best_rule_set
        for search_operator in search_operators:
            scored_children = search_operator.get_children(rule_set)
            for score, child in scored_children:
                if score > best_score:
                    best_rule_set = child
                    best_score = score
                    print("New best score:", best_score)
                    print("New best rule set:", best_rule_set)

    return best_rule_set

def run_best_first_search(transition_dataset, max_node_expansions=1000, rng=None):
    if rng is None:
        rng = np.random.RandomState(seed=0)

    search_operators = get_search_operators(transition_dataset)

    score, default_rule_set = create_default_rule_set(transition_dataset)
    queue = []
    hq.heappush(queue, (0, 0, default_rule_set))
    best_rule_set = default_rule_set
    best_score = score

    print("Starting search with initial score", score)

    for n in range(max_node_expansions):
        if len(queue) == 0:
            break
        print("Expanding node {}/{}".format(n, max_node_expansions))
        _, _, rule_set = hq.heappop(queue)
        for search_operator in search_operators:
            scored_children = search_operator.get_children(rule_set)
            for score, child in scored_children:
                hq.heappush(queue, (score, rng.uniform(), child))
                if score > best_score:
                    best_rule_set = child
                    best_score = score
                    print("New best score:", best_score)
                    print("New best rule set:", best_rule_set)

    return best_rule_set

def print_rule_set(rule_set):
    for action_predicate in sorted(rule_set):
        print(colored(action_predicate, attrs=['bold']))
        print(rule_set[action_predicate])

def main():
    num_problems = 1
    num_transitions_per_problem = 10
    max_node_expansions = 10
    print("Collecting transition data... ", end='')
    transition_dataset = collect_transition_dataset(num_problems, num_transitions_per_problem)
    print("collected {} transition.".format(len(transition_dataset)))
    print("Running search...")
    rule_set = run_greedy_search(transition_dataset, max_node_expansions=max_node_expansions)
    print("Learned rule set:")
    print_rule_set(rule_set)

if __name__ == "__main__":
    main()
