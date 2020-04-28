"""Main file for NDR learning
"""
from ndr.structs import Anti, NDR, ground_literal, LiteralConjunction
from ndr.inference import find_satisfying_assignments
from envs.ndr_blocks import NDRBlocksEnv, noiseoutcome
from collections import defaultdict
from termcolor import colored
from scipy.optimize import minimize
import heapq as hq
import numpy as np
import copy
import time


ALPHA = 30. # Weight on rule set size penalty
P_MIN = 1e-8 # Probability for an individual noisy outcome
DEBUG = False

class MultipleOutcomesPossible(Exception):
    pass


## Helper functions
def iter_variable_names():
    """Generate unique variable names
    """
    i = 0
    while True:
        yield "?x{}".format(i)
        i += 1

def find_assignments_for_ndr(ndr, state, action):
    """Find all possible assignments of variables to values given
    the action and preconditions of an ndr
    """
    kb = state | { action }
    assert action.predicate == ndr.action.predicate
    conds = [ndr.action] + list(ndr.preconditions)
    return find_satisfying_assignments(kb, conds)

def rule_covers_transition(rule, transition, check_changed_objects=True):
    """Check whether the action and preconditions cover the transition
    """
    state, action, effects = transition
    assignments = find_assignments_for_ndr(rule, state, action)
    # Only covers if there is a unique binding of the variables
    if len(assignments) == 1:
        assigned_vals = set(assignments[0].values())
        # Only covers if all the objects in the effects are bound to vars
        if check_changed_objects:
            for lit in effects:
                for val in lit.variables:
                    if val not in assigned_vals:
                        return False
        return True
    return False

def get_covered_transitions(rule, transitions_for_action, 
                            rule_is_default=False, action_rule_set=None,
                            check_changed_objects=True):
    """Collect the transitions that this rule covers
    """
    if rule_is_default:
        assert len(rule.preconditions) == 0
        assert action_rule_set is not None
    else:
        assert len(rule.preconditions) > 0

    covered_transitions = []
    for transition in transitions_for_action:
        if rule_is_default:
            if covered_by_default_rule(transition, action_rule_set):
                covered_transitions.append(transition)
        elif rule_covers_transition(rule, transition, 
            check_changed_objects=check_changed_objects):
            covered_transitions.append(transition)
    return covered_transitions

def print_rule_set(rule_set):
    for action_predicate in sorted(rule_set):
        print(colored(action_predicate, attrs=['bold']))
        for rule in rule_set[action_predicate]:
            print(rule)

def print_transition(transition):
    print("  State:", transition[0])
    print("  Action:", transition[1])
    print("  Effects:", transition[2])

## Scoring
def get_pen(rule):
    """Helper for scores. Counts number of literals in rule to penalize
    """
    pen = 0
    preconds = rule.preconditions
    pen += len(preconds)
    for _, outcome in rule.effects:
        pen += len(outcome)
    return pen

def get_effect_for_transition(rule, transition):
    """Find the (assumed unique) effect that holds for the (assumed covered) transition
    """
    state, action, effects = transition
    assignments = find_assignments_for_ndr(rule, state, action)
    assert len(assignments) == 1, "Rule assumed to cover transition"
    selected_outcome_idx = None
    noise_outcome_idx = None
    for i, (_, outcome) in enumerate(rule.effects):
        if noiseoutcome() in outcome:
            assert noise_outcome_idx is None
            noise_outcome_idx = i
        else:
            grounded_outcome = {ground_literal(lit, assignments[0]) for lit in outcome}
            if sorted(grounded_outcome) == sorted(effects):
                if selected_outcome_idx is not None:
                    raise MultipleOutcomesPossible()
                selected_outcome_idx = i
    if selected_outcome_idx is not None:
        return selected_outcome_idx
    assert noise_outcome_idx is not None
    return noise_outcome_idx

def get_transition_likelihood(transition, rule, p_min=P_MIN):
    """Calculate the likelihood of a transition for a rule that covers it
    """
    state, action, effects = transition
    assignments = find_assignments_for_ndr(rule, state, action)
    assert len(assignments) == 1, "Rule assumed to cover transition"
    transition_likelihood = 0.
    for prob, outcome in rule.effects:
        if noiseoutcome() in outcome:
            # c.f. equation 3 in paper
            transition_likelihood += p_min * prob
        else:
            grounded_outcome = {ground_literal(lit, assignments[0]) for lit in outcome}
            if sorted(grounded_outcome) == sorted(effects):
                transition_likelihood += prob
    return transition_likelihood

def score_action_rule_set(action_rule_set, transitions_for_action, p_min=P_MIN, alpha=ALPHA):
    """Score a full rule set for an action
    """
    score = 0.

    # Calculate penalty for number of literals
    for rule in action_rule_set:
        pen = get_pen(rule)
        score += - alpha * pen

    # Calculate transition likelihoods per example and accumulate score
    for (state, action, effects) in transitions_for_action:
        # Figure out which rule covers the transition
        selected_ndr_idx = None
        for idx, ndr in enumerate(action_rule_set):
            assignments = find_assignments_for_ndr(ndr, state, action)
            if len(assignments) == 1:
                selected_ndr_idx = idx
                break
        assert selected_ndr_idx is not None, "At least the default NDR should be selected"
        selected_ndr = action_rule_set[selected_ndr_idx]
        # Calculate transition likelihood
        transition_likelihood = get_transition_likelihood((state, action, effects), 
            selected_ndr, p_min=p_min)
        # Add to score
        if transition_likelihood == 0.:
            return -10e8
        score += np.log(transition_likelihood)

    return score

def score_rule(rule, transitions_for_rule, p_min=P_MIN, alpha=ALPHA, compute_penalty=True):
    """Score a single rule on examples that it covers
    """
    # Calculate penalty for number of literals
    score = 0
    if compute_penalty:
        pen = get_pen(rule)
        score += - alpha * pen

    # Calculate transition likelihoods per example and accumulate score
    for transition in transitions_for_rule:
        # Calculate transition likelihood
        transition_likelihood = get_transition_likelihood(transition, rule, p_min=p_min)
        # Add to score
        if transition_likelihood == 0.:
            return -10e8
        score += np.log(transition_likelihood)

    return score

## Init rule set
def initialize_from_rule_set(rule_set, transitions_for_action):
    init_rule_set = []
    # Just fit the transition probabilities
    for ndr in rule_set:
        ndr_copy = copy.deepcopy(ndr)
        rule_is_default = (len(ndr.preconditions) == 0)
        covered_transitions = get_covered_transitions(ndr_copy, transitions_for_action,
            rule_is_default=rule_is_default, action_rule_set=rule_set,
            check_changed_objects=False)
        learn_parameters(ndr_copy, covered_transitions)
        init_rule_set.append(ndr_copy)
    score = score_action_rule_set(init_rule_set, transitions_for_action)
    return score, init_rule_set


## Default rules
def create_default_rule_set_for_action(action, transitions_for_action):
    """Helper for create default rule set. One default rule for action.
    """
    variable_name_generator = iter_variable_names()
    variable_names = [next(variable_name_generator) for _ in range(action.arity)]
    lifted_action = action(*variable_names)
    ndr = NDR(action=lifted_action, preconditions=LiteralConjunction([]), effects=None)
    induce_outcomes(ndr, transitions_for_action, rule_is_default=True,
        action_rule_set=[])
    score = score_action_rule_set([ndr], transitions_for_action)
    return score, [ndr]

def covered_by_default_rule(transition, action_rule_set):
    """Check whether the transition is covered by any non-default rules
    """
    # default rule is assumed to be last!
    for rule in action_rule_set[:-1]:
        if rule_covers_transition(rule, transition, check_changed_objects=False):
            return False
    return True

def get_unique_transitions(transitions):
    """Filter out transitions that are literally identical
    """
    unique_transitions = []
    seen_hashes = set()
    for s, a, e in transitions:
        hashed = (frozenset(s), a, frozenset(e))
        if hashed not in seen_hashes:
            unique_transitions.append((s, a, e))
        seen_hashes.add(hashed)
    return unique_transitions

## Learn parameters
def learn_parameters(rule, covered_transitions, maxiter=100):
    """Learn effect probabilities given the rest of a rule
    """
    # First check whether all of the rule effects are mutually exclusive.
    # If so, we can compute analytically!
    try:
        return learn_params_analytically(rule, covered_transitions)
    except MultipleOutcomesPossible:
        pass

    def update_rule(x):
        for i, p in enumerate(x):
            _, eff = rule.effects[i]
            rule.effects[i] = (p, eff)

    # Set up the loss
    def loss(x):
        update_rule(x)
        return -1. * score_rule(rule, covered_transitions, compute_penalty=False)
    # Set up init x
    x0 = [1./len(rule.effects) for _ in rule.effects]
    # Run optimization
    cons = [{'type': 'eq', 'fun' : lambda x: sum(x) - 1. }]
    bounds=[(0, 1) for i in range(len(x0))]
    result = minimize(loss, x0, method='SLSQP', constraints=tuple(cons), bounds=bounds,
        options={'disp' : False, 'maxiter' : maxiter})
    params = result.x
    assert all((0 <= p <= 1.) for p in params), "Optimization does not obey bounds"
    update_rule(params)

def learn_params_analytically(rule, covered_transitions):
    """Assuming effects are mutually exclusive, find best params"""
    effect_counts = [0. for _ in rule.effects]
    for transition in covered_transitions:
        idx = get_effect_for_transition(rule, transition)
        effect_counts[idx] += 1
    params = np.array(effect_counts) / np.sum(effect_counts)
    for i, p in enumerate(params):
        _, eff = rule.effects[i]
        rule.effects[i] = (p, eff)

## Induce outcomes
def create_induce_outcome_add_operator(rule, covered_transitions):
    """Pick a pair of non-contradictory outcomes and conjoin them
       (making sure not to conjoin with noiseoutcome)
    """
    def get_children(effects):
        for i, (p_i, effect_i) in enumerate(effects[:-1]):
            if noiseoutcome() in effect_i:
                continue
            for k, (p_j, effect_j) in enumerate(effects[i+1:]):
                j = k + (i+1)
                if noiseoutcome() in effect_j:
                    continue
                # Check for contradiction
                contradiction = False
                for lit_i in effect_i:
                    if contradiction:
                        break
                    for lit_j in effect_j:
                        if Anti(lit_i.predicate) == lit_j:
                            contradiction = True
                            break
                if not contradiction:
                    combined_effects = sorted(set(effect_i) | set(effect_j))
                    combined_p = p_i + p_j
                    remaining_effects = []
                    for m, effect in enumerate(effects):
                        if m in [i, j]:
                            continue
                        remaining_effects.append(effect)
                    remaining_effects.append((combined_p, combined_effects))
                    # Search for better parameters
                    # print("add remaining_effects:", remaining_effects)
                    rule.effects = remaining_effects
                    learn_parameters(rule, covered_transitions)
                    score = score_rule(rule, covered_transitions)
                    yield score, [eff for eff in rule.effects]
        return
    return get_children

def create_induce_outcome_remove_operator(rule, covered_transitions):
    """Drop an outcome (not the noise one though!)
    """
    def get_children(effects):
        for i, (p_i, effect_i) in enumerate(effects):
            if noiseoutcome() in effect_i:
                continue
            remaining_effects = []
            for j, effect in enumerate(effects):
                if i == j:
                    continue
                remaining_effects.append(effect)
                # Search for better parameters
            # print("remove remaining_effects:", remaining_effects)
            rule.effects = remaining_effects
            learn_parameters(rule, covered_transitions)
            score = score_rule(rule, covered_transitions)
            yield score, [eff for eff in rule.effects]
        return
    return get_children

def create_induce_outcomes_operators(rule, covered_transitions):
    """Search operators for outcome induction
    """
    add_operator = create_induce_outcome_add_operator(rule, covered_transitions)
    remove_operator = create_induce_outcome_remove_operator(rule, covered_transitions)
    return [add_operator, remove_operator]

def induce_outcomes(rule, transitions_for_action, rule_is_default=False, action_rule_set=None,
                    max_node_expansions=100):
    """Induce outcomes for a rule

    Modifies the rule in place.
    """
    # Find all transitions covered by this rule (only uses the action and preconditions)
    covered_transitions = get_covered_transitions(rule, transitions_for_action,
        rule_is_default=rule_is_default, action_rule_set=action_rule_set)
    # Get all possible outcomes
    # For default rule, the only possible outcomes are noise and nothing
    if rule_is_default:
        all_possible_outcomes = { (noiseoutcome(),), tuple() }
    else:
        all_possible_outcomes = { (noiseoutcome(),) }
        for state, action, effects in covered_transitions:
            assignments = find_assignments_for_ndr(rule, state, action)
            assert len(assignments) == 1
            inverse_sigma = {v : k for k, v in assignments[0].items()}
            lifted_effects = set()
            for effect in effects:
                lifted_effect = effect.predicate(*[inverse_sigma[val] for val in effect.variables])
                lifted_effects.add(lifted_effect)
            all_possible_outcomes.add(tuple(sorted(lifted_effects)))
    # Initialize effects with uniform distribution over all possible outcomes
    num_possible_outcomes = len(all_possible_outcomes)
    init_state = [(1./num_possible_outcomes, list(outcome)) for outcome in sorted(all_possible_outcomes)]
    rule.effects = init_state
    # Search for better parameters
    learn_parameters(rule, covered_transitions)
    init_score = score_rule(rule, covered_transitions)
    # Search for better effects
    search_operators = create_induce_outcomes_operators(rule, covered_transitions)
    best_effects = run_greedy_search(search_operators, init_state, init_score,
        max_node_expansions=max_node_expansions)
    rule.effects = best_effects

## Main search operators
def create_explain_examples_operator(action, transitions_for_action):
    """Explain examples, the beefiest search operator

    Tries to follow the pseudocode in the paper as faithfully as possible
    """
    unique_transitions = get_unique_transitions(transitions_for_action)

    def explain_examples_for_action(action_rule_set):
        """The operator
        """
        for i, transition in enumerate(unique_transitions):
            if not DEBUG:
                print("Running explain examples for action {} {}/{}".format(action, i, 
                    len(unique_transitions)), end='\r')
                if i == len(unique_transitions) -1:
                    print()
            if DEBUG: print("Considering explaining example for transition")
            if DEBUG: print_transition(transition)
            # Only want to explain examples that are covered by the default rule
            # i.e., that are not covered by a regular rule
            if not covered_by_default_rule(transition, action_rule_set):
                continue
            s, a, effs = transition
            # Step 1: Create a new rule
            new_rule = NDR(action=None, preconditions=LiteralConjunction([]), effects=None)
            # Step 1.1: Create an action and context for r
            # Create new variables to represent the arguments of a
            variable_name_generator = iter_variable_names()
            # Use them to create a new action substition
            variables = [next(variable_name_generator) for _ in a.variables]
            sigma = dict(zip(variables, a.variables))
            sigma_inverse = {v : k for k, v in sigma.items()}
            # Set r's action
            new_rule.action = action(*[sigma_inverse[val] for val in a.variables])
            # Set r's context to be the conjunction literals that can be formed using
            # the variables
            for lit in s:
                if all(val in sigma_inverse for val in lit.variables):
                    lifted_lit = lit.predicate(*[sigma_inverse[val] for val in lit.variables])
                    new_rule.preconditions.literals.append(lifted_lit)
            if DEBUG: import ipdb; ipdb.set_trace()
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
                assignments_d = find_satisfying_assignments(s, new_rule.preconditions.literals+d)
                assert len(assignments_d) >= 1
                # If so, add it to r
                if len(assignments_d) == 1:
                    new_rule.preconditions.literals.extend(d)
            # Step 1.3: Complete the rule
            # Call InduceOutComes to create the rule's outcomes.
            # If preconditions are empty, don't enumerate; this should be covered by the default rule
            if len(new_rule.preconditions) == 0:
                continue
            induce_outcomes(new_rule, transitions_for_action)
            assert new_rule.effects is not None
            if DEBUG: import ipdb; ipdb.set_trace()
            # Step 2: Trim literals from r
            # Create a rule set R' containing r and the default rule
            # Greedily trim literals from r, ensuring that r still covers (s, a, s')
            # and filling in the outcomes using InduceOutcomes until R's score stops improving
            trim_candidates = list(new_rule.preconditions.literals)
            # Recompute the parameters of the default rule
            default_rule = copy.deepcopy(action_rule_set[-1])
            induce_outcomes(default_rule, transitions_for_action, 
                rule_is_default=True, action_rule_set=[new_rule, default_rule])
            best_score = score_action_rule_set([new_rule, default_rule], transitions_for_action)
            while len(trim_candidates) > 0:
                lit = trim_candidates.pop()
                candidate_preconditions = LiteralConjunction(
                    [l for l in new_rule.preconditions.literals if l != lit])
                # If preconditions are empty, don't enumerate; this should be covered by the default rule
                if len(candidate_preconditions) == 0:
                    break
                candidate_new_rule = NDR(action=new_rule.action, 
                    preconditions=candidate_preconditions,
                    effects=None)
                if not rule_covers_transition(candidate_new_rule, transition):
                    continue
                induce_outcomes(candidate_new_rule, transitions_for_action)
                # Recompute the parameters of the default rule
                induce_outcomes(default_rule, transitions_for_action, 
                    rule_is_default=True, action_rule_set=[candidate_new_rule, default_rule])
                score = score_action_rule_set([candidate_new_rule, default_rule], transitions_for_action)
                if score > best_score:
                    if DEBUG: import ipdb; ipdb.set_trace()
                    best_score = score
                    new_rule = candidate_new_rule
            # If preconditions are empty, don't enumerate; this should be covered by the default rule
            if len(new_rule.preconditions.literals) == 0:
                continue
            if DEBUG: import ipdb; ipdb.set_trace()
            # Step 3: Create a new rule set containing r
            # Create a new rule set R' = R
            deprecated_rules = set()
            # Add r to R' and remove any rules in R' that cover any examples r covers
            for t in transitions_for_action:
                if not rule_covers_transition(new_rule, t):
                    continue
                # leave out default rule
                for rule in action_rule_set[:-1]:
                    # If default, continue
                    if rule_covers_transition(rule, t):
                        deprecated_rules.add(rule)
            new_rule_set = [new_rule] + [rule for rule in action_rule_set if rule not in deprecated_rules]
            new_rule_set[-1] = default_rule
            # Recompute the parameters of the default rule
            induce_outcomes(default_rule, transitions_for_action, 
                rule_is_default=True, action_rule_set=new_rule_set)
            if DEBUG: import ipdb; ipdb.set_trace()
            # Add R' to the return rule sets R_O
            score = score_action_rule_set(new_rule_set, transitions_for_action)
            yield score, new_rule_set
        return

    return explain_examples_for_action

def get_search_operators(action, transitions_for_action):
    """Main search operators
    """
    explain_examples = create_explain_examples_operator(action, transitions_for_action)

    return [explain_examples]

## Search
def run_main_search(transition_dataset, max_node_expansions=1000, rng=None, 
                    search_method="greedy", init_rule_set=None):
    """Run the main search
    """
    rule_sets = {}

    for action, transitions_for_action in transition_dataset.items():
        print("Running search for action", action)

        search_operators = get_search_operators(action, transitions_for_action)
        if init_rule_set is not None:
            init_score, init_state = initialize_from_rule_set(init_rule_set[action], 
                transitions_for_action)
        else:
            init_score, init_state = create_default_rule_set_for_action(action, transitions_for_action)

        print("Initial rule set (score={}):".format(init_score))
        print_rule_set({action : init_state})

        if search_method == "greedy":
            action_rule_set = run_greedy_search(search_operators, init_state, init_score, 
                max_node_expansions=max_node_expansions, rng=rng, verbose=True)
        elif search_method == "best_first":
            action_rule_set = run_best_first_search(search_operators, init_state, init_score, 
                max_node_expansions=max_node_expansions, rng=rng, verbose=True)
        else:
            raise NotImplementedError()

        rule_sets[action] = action_rule_set

    return rule_sets

def run_greedy_search(search_operators, init_state, init_score,
                      max_node_expansions=1000, rng=None, verbose=False):
    """Greedy search
    """
    if rng is None:
        rng = np.random.RandomState(seed=0)

    best_score, state = init_score, init_state

    if verbose:
        print("Starting greedy search with initial score", best_score)

    for n in range(max_node_expansions):
        if verbose:
            print("Expanding node {}/{}".format(n, max_node_expansions))
        found_improvement = False
        for search_operator in search_operators:
            scored_children = search_operator(state)
            for score, child in scored_children:
                if verbose and DEBUG:
                    import ipdb; ipdb.set_trace()
                if score > best_score:
                    state = child
                    best_score = score
                    found_improvement = True
                    if verbose:
                        print("New best score:", best_score)
                        print("New best state:", state)
        if not found_improvement:
            break

    return state

def run_best_first_search(search_operators, init_state, init_score,
                          max_node_expansions=1000, rng=None, verbose=False):
    """Best first search
    """
    if rng is None:
        rng = np.random.RandomState(seed=0)

    best_score, best_state = init_score, init_state

    queue = []
    hq.heappush(queue, (0, 0, state))

    if verbose:
        print("Starting search with initial score", best_score)

    for n in range(max_node_expansions):
        if len(queue) == 0:
            break
        if verbose:
            print("Expanding node {}/{}".format(n, max_node_expansions))
        _, _, state = hq.heappop(queue)
        for search_operator in search_operators:
            scored_children = search_operator(state)
            for score, child in scored_children:
                hq.heappush(queue, (score, rng.uniform(), child))
                if score > best_score:
                    best_state = child
                    best_score = score
                    if verbose:
                        print("New best score:", best_score)
                        print("New best state:", best_rule_set)

    return best_state

def main():
    num_problems = 3
    num_transitions_per_problem = 250
    max_node_expansions = 10
    print("Collecting transition data... ", end='')
    # transition_dataset = collect_manual_transition_dataset()
    env = NDRBlocksEnv(seed=0)
    transition_dataset = collect_transition_dataset(env, num_problems, num_transitions_per_problem)
    print("collected transitions for {} actions.".format(len(transition_dataset)))
    print("Running search...")
    start_time = time.time()
    rule_set = run_main_search(transition_dataset, max_node_expansions=max_node_expansions)
    print("Learned rule set:")
    print_rule_set(rule_set)
    print("Total search time:", time.time() - start_time)

if __name__ == "__main__":
    main()
