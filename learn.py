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

def iter_variable_names():
    i = 0
    while True:
        yield "X{}".format(i)
        i += 1

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
        action_rule = create_default_rule_for_action(action_predicate)
        action_rule_set = [action_rule]
        rule_set[action_predicate] = action_rule_set
        total_score += score_action_rule_set(action_rule_set, transitions_for_action)
    return total_score, rule_set

def create_default_rule_for_action(action):
    variable_name_generator = iter_variable_names()
    variable_names = [next(variable_name_generator) for _ in range(action.arity)]
    lifted_action = action(*variable_names)
    return NDR(action=lifted_action, preconditions=[], effects=[(1.0, noiseoutcome())])

def covered_by_default_rule(transition, action_rule_set):
    # default rule is assumed to be last!
    for rule in action_rule_set[:-1]:
        if rule_covers_transition(rule, transition):
            return False
    return True

def rule_covers_transition(rule, transition):
    state, action, _ = transition
    assignments = find_assignments_for_ndr(rule, state, action)
    if len(assignments) == 1:
        return True
    return False

def induce_outcomes(rule):
    assert rule.effects is None
    # modify the rule in place
    import ipdb; ipdb.set_trace()

## Operators
def create_explain_examples_operator(transition_dataset):
    def explain_examples_for_action(action, action_rule_set):
        transitions_for_action = transition_dataset[action]
        default_rule = create_default_rule_for_action(action)
        returned_rule_sets = []
        for transition in transitions_for_action:
            if not covered_by_default_rule(transition, action_rule_set):
                continue
            s, a, effs = transition
            # Step 1: Create a new rule
            new_rule = NDR(action=None, preconditions=LiteralConjunction([]), effects=None)
            # Step 1.1: Create an action and context for r
            # Create new variables to represent the arguments of a
            variable_name_generator = iter_variable_names()
            # Use them to create a new action substition
            sigma = dict(zip(variable_name_generator, a.variables))
            sigma_inverse = {v : k for k, v in sigma.items()}
            # Set r's action
            new_rule.action = action(*[sigma_inverse[val] for val in a.variables])
            # Set r's context to be the conjunction literals that can be formed using
            # the variables
            for lit in s:
                if all(val in sigma_inverse for val in lit.variables):
                    lifted_lit = lit.predicate(*[sigma_inverse[val] for val in lit.variables])
                    new_rule.preconditions.literals.append(lifted_lit)
            # Step 1.2: Create deictic references for r
            # Collect the set of constants whose properties changed from s to s' but 
            # which are not in sigma
            changed_objects = set()
            for lit in effs:
                for val in lit.variables:
                    if val not in sigma_inverse:
                        changed_objects.add(val)
            for c in sorted(changed_objects):
                # Create a new variable and extend sigma to map v to c
                new_variable = next(variable_name_generator)
                sigma[new_variable] = c
                assert c not in sigma_inverse
                sigma_inverse[c] = new_variable
                # Create the conjunction of literals containing c, but lifted
                d = []
                for lit in s:
                    if c not in lit.variables:
                        continue
                    if all(val in sigma_inverse for val in lit.variables):
                        lifted_lit = lit.predicate(*[sigma_inverse[val] for val in lit.variables])
                        d.append(lifted_lit)
                # Check if d uniquely refers to c in s
                assignments_d = find_satisfying_assignments(s, d)
                assert len(assignments_d) >= 1
                # If so, add it to r
                if len(assignments_d) == 1:
                    new_rule.preconditions.literals.extend(d)
            # Step 1.3: Complete the rule
            # Call InduceOutComes to create the rule's outcomes.
            induce_outcomes(new_rule)
            assert new_rule.effects is not None
            # Step 2: Trim literals from r
            # Create a rule set R' containing r and the default rule
            # Greedily trim literals from r, ensuring that r still covers (s, a, s')
            # and filling in the outcomes using InduceOutcomes until R's score stops improving
            trim_candidates = list(new_rule.preconditions.literals)
            best_score = score_action_rule_set([new_rule, default_rule], transitions_for_action)
            while len(trim_candidates) > 0:
                lit = trim_candidates.pop()
                candidate_preconditions = LiteralConjunction(
                    [l for l in new_rule.preconditions.literals if l != lit])
                candidate_new_rule = NDR(action=new_rule.action, 
                    preconditions=candidate_preconditions,
                    effects=None)
                if not rule_covers_transition(candidate_new_rule, transition):
                    continue
                induce_outcomes(candidate_new_rule)
                score = score_action_rule_set([candidate_new_rule, default_rule], transitions_for_action)
                if score > best_score:
                    best_score = score
                    new_rule = candidate_new_rule
            # Step 3: Create a new rule set containing r
            # Create a new rule set R' = R
            deprecated_rules = []
            # Add r to R' and remove any rules in R' that cover any examples r covers
            for t in transitions:
                if not rule_covers_transition(new_rule, t):
                    continue
                for rule in action_rule_set:
                    if rule_covers_transition(rule, t):
                        deprecated_rules.append(rule)
            new_rule_set = [rule for rule in action_rule_set if rule not in deprecated_rules]
            new_rule_set.append(new_rule)
            new_rule_set.append(default_rule)
            # Recompute the parameters of the default rule
            induce_outcomes(default_rule)
            # Add R' to the return rule sets R_O
            returned_rule_sets.append(new_rule_set)
        return returned_rule_sets


    def explain_examples(rule_set):
        new_rule_set = {}
        for action, action_rule_set in rule_set.items():
            new_rule_set[action] = explain_examples_for_action(action, action_rule_set)
        return new_rule_set

    return explain_examples

def get_search_operators(transition_dataset):
    explain_examples = create_explain_examples_operator(transition_dataset)

    return [explain_examples]

## Search
def run_search(search_method, *args, **kwargs):
    if search_method == "greedy":
        return run_greedy_search(*args, **kwargs)
    if search_method == "best_first":
        return run_best_first_search(*args, **kwargs)
    raise NotImplementedError()

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
            scored_children = search_operator(rule_set)
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
    search_method = "greedy"
    print("Collecting transition data... ", end='')
    transition_dataset = collect_transition_dataset(num_problems, num_transitions_per_problem)
    print("collected transitions for {} actions.".format(len(transition_dataset)))
    print("Running search...")
    rule_set = run_search(search_method, transition_dataset, max_node_expansions=max_node_expansions)
    print("Learned rule set:")
    print_rule_set(rule_set)

if __name__ == "__main__":
    main()
