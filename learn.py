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


ALPHA = 0.5 # Weight on rule set size penalty
P_MIN = 1e-8 # Probability for an individual noisy outcome
DEBUG = False

## Data collection
def collect_manual_transition_dataset():
    """For debugging
    """
    from envs.ndr_blocks import on, ontable, clear, handempty, pickup, holding
    transitions = defaultdict(list)

    state = {on("a", "b"), on("b", "c"), ontable("c"), clear("a"), handempty()}
    action = pickup("b")
    effects = set()
    transitions[action.predicate].append((state, action, effects))

    state = {on("a", "b"), on("b", "c"), ontable("c"), clear("a"), handempty()}
    action = pickup("c")
    effects = set()
    transitions[action.predicate].append((state, action, effects))

    state = {on("a", "b"), on("b", "c"), ontable("c"), clear("a"), handempty()}
    action = pickup("a")
    effects = {Anti(on("a", "b")), clear("b"), Anti(clear("a")), Anti(handempty()), 
        holding("a")}
    transitions[action.predicate].append((state, action, effects))

    state = {on("a", "b"), on("b", "c"), ontable("c"), clear("a"), handempty()}
    action = pickup("a")
    effects = {Anti(on("a", "b")), clear("b"), Anti(clear("a")), Anti(handempty()), 
        holding("a")}
    transitions[action.predicate].append((state, action, effects))

    return transitions

def collect_transition_dataset(env, num_problems, num_transitions_per_problem, policy=None, actions="all"):
    """Collect transitions (state, action, effect) for the given actions
    Make sure that no more than 50% of outcomes per action are null.
    """
    total_counts = defaultdict(int)
    num_no_effects = defaultdict(int)

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

## Helper functions
def iter_variable_names():
    """Generate unique variable names
    """
    i = 0
    while True:
        yield "X{}".format(i)
        i += 1

def find_assignments_for_ndr(ndr, state, action):
    """Find all possible assignments of variables to values given
    the action and preconditions of an ndr
    """
    kb = state | { action }
    assert action.predicate == ndr.action.predicate
    conds = [ndr.action] + list(ndr.preconditions.literals)
    return find_satisfying_assignments(kb, conds)

def rule_covers_transition(rule, transition):
    """Check whether the action and preconditions cover the transition
    """
    state, action, effects = transition
    assignments = find_assignments_for_ndr(rule, state, action)
    # Only covers if there is a unique binding of the variables
    if len(assignments) == 1:
        assigned_vals = set(assignments[0].values())
        # Only covers if all the objects in the effects are bound to vars
        for lit in effects:
            for val in lit.variables:
                if val not in assigned_vals:
                    return False
        return True
    return False

def get_covered_transitions(rule, transitions_for_action, rule_is_default=False, action_rule_set=None):
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
        elif rule_covers_transition(rule, transition):
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
            if grounded_outcome == effects:
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

## Default rules
def create_default_rule_set(transition_dataset):
    """Create the initial rule set, containing just a default rule
    """
    rule_set = {}
    total_score = 0.
    for action_predicate, transitions_for_action in transition_dataset.items():
        action_rule = create_default_rule_for_action(action_predicate, transitions_for_action)
        action_rule_set = [action_rule]
        rule_set[action_predicate] = action_rule_set
        total_score += score_action_rule_set(action_rule_set, transitions_for_action)
    return total_score, rule_set

def create_default_rule_for_action(action, transitions_for_action):
    """Helper for create default rule set. One default rule for action.
    """
    variable_name_generator = iter_variable_names()
    variable_names = [next(variable_name_generator) for _ in range(action.arity)]
    lifted_action = action(*variable_names)
    ndr = NDR(action=lifted_action, preconditions=LiteralConjunction([]), effects=None)
    induce_outcomes(ndr, transitions_for_action, rule_is_default=True,
        action_rule_set=[])
    return ndr

def covered_by_default_rule(transition, action_rule_set):
    """Check whether the transition is covered by any non-default rules
    """
    # default rule is assumed to be last!
    for rule in action_rule_set[:-1]:
        if rule_covers_transition(rule, transition):
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
    # Set up the loss
    def loss(x):
        for i, p in enumerate(x):
            _, eff = rule.effects[i]
            rule.effects[i] = (p, eff)
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
    return params


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
                j = k - (i+1)
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
def create_explain_examples_operator(transition_dataset):
    """Explain examples, the beefiest search operator

    Tries to follow the pseudocode in the paper as faithfully as possible
    """
    def explain_examples_for_action(action, action_rule_set):
        """Helper that explains examples for a single action
        """
        transitions_for_action = transition_dataset[action]
        unique_transitions = get_unique_transitions(transitions_for_action)
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
            yield new_rule_set
        return


    def explain_examples(rule_set):
        """The operator for explain examples
        """
        # First calculate scores for all actions
        action_to_score = {}
        for action, action_rule_set in rule_set.items():
            transitions_for_action = transition_dataset[action]
            score = score_action_rule_set(action_rule_set, transitions_for_action)
            action_to_score[action] = score

        # Get children
        for action, action_rule_set in rule_set.items():
            # Get base score for other actions
            base_score = sum([action_to_score[a] for a in rule_set if a != action])
            transitions_for_action = transition_dataset[action]
            for new_action_rule_set in explain_examples_for_action(action, action_rule_set):
                new_rule_set = {k : v for k, v in rule_set.items() }
                new_rule_set[action] = new_action_rule_set
                score = base_score + score_action_rule_set(new_action_rule_set, transitions_for_action)
                if DEBUG: import ipdb; ipdb.set_trace()
                yield score, new_rule_set

    return explain_examples

def get_search_operators(transition_dataset):
    """Main search operators
    """
    explain_examples = create_explain_examples_operator(transition_dataset)

    return [explain_examples]

## Search
def run_main_search(search_method, transition_dataset, max_node_expansions=1000, rng=None):
    """Run the main search
    """
    search_operators = get_search_operators(transition_dataset)
    init_score, init_state = create_default_rule_set(transition_dataset)

    print("Initial rule set:")
    print_rule_set(init_state)

    if search_method == "greedy":
        return run_greedy_search(search_operators, init_state, init_score, 
            max_node_expansions=max_node_expansions, rng=rng, verbose=True)
    if search_method == "best_first":
        return run_best_first_search(search_operators, init_state, init_score, 
            max_node_expansions=max_node_expansions, rng=rng, verbose=True)

    raise NotImplementedError()

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
    search_method = "greedy"
    print("Collecting transition data... ", end='')
    # transition_dataset = collect_manual_transition_dataset()
    env = NDRBlocksEnv(seed=0)
    transition_dataset = collect_transition_dataset(env, num_problems, num_transitions_per_problem)
    print("collected transitions for {} actions.".format(len(transition_dataset)))
    print("Running search...")
    start_time = time.time()
    rule_set = run_main_search(search_method, transition_dataset, max_node_expansions=max_node_expansions)
    print("Learned rule set:")
    print_rule_set(rule_set)
    print("Total search time:", time.time() - start_time)

if __name__ == "__main__":
    main()
