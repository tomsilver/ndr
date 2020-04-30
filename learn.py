"""Main file for NDR learning
"""
from ndr.ndrs import NDR, NDRSet, NOISE_OUTCOME, MultipleOutcomesPossible
from ndr.structs import Anti, ground_literal
from collections import defaultdict
from termcolor import colored
from scipy.optimize import minimize
import heapq as hq
import numpy as np
import copy
import time
import abc


ALPHA = 20. # Weight on rule set size penalty
P_MIN = 1e-8 # Probability for an individual noisy outcome
DEBUG = False

## Generic search
class SearchOperator:

    @abc.abstractmethod
    def get_children(self, node):
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
            scored_children = search_operator.get_children(state)
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
            scored_children = search_operator.get_children(state)
            for score, child in scored_children:
                hq.heappush(queue, (score, rng.uniform(), child))
                if score > best_score:
                    best_state = child
                    best_score = score
                    if verbose:
                        print("New best score:", best_score)
                        print("New best state:", best_rule_set)

    return best_state


## Helper functions
def iter_variable_names():
    """Generate unique variable names
    """
    i = 0
    while True:
        yield "?x{}".format(i)
        i += 1

def print_rule_set(rule_set):
    for action_predicate in sorted(rule_set):
        print(colored(action_predicate, attrs=['bold']))
        for rule in rule_set[action_predicate]:
            print(rule)

def print_transition(transition):
    print("  State:", transition[0])
    print("  Action:", transition[1])
    print("  Effects:", transition[2])

def get_unique_transitions(transitions):
    """Filter out transitions that are literally (pun) identical
    """
    unique_transitions = []
    seen_hashes = set()
    for s, a, e in transitions:
        hashed = (frozenset(s), a, frozenset(e))
        if hashed not in seen_hashes:
            unique_transitions.append((s, a, e))
        seen_hashes.add(hashed)
    return unique_transitions

## Scoring
def get_pen(rule):
    """Helper for scores. Counts number of literals in rule to penalize
    """
    pen = 0
    preconds = rule.preconditions
    pen += len(preconds)
    for effect in rule.effects:
        pen += len(effect)
    return pen

def get_transition_likelihood(transition, rule, p_min=P_MIN):
    """Calculate the likelihood of a transition for a rule that covers it
    """
    state, action, effects = transition
    sigma = rule.find_substitutions(state, action)
    assert sigma is not None, "Rule assumed to cover transition"
    transition_likelihood = 0.
    for prob, outcome in zip(rule.effect_probs, rule.effects):
        if NOISE_OUTCOME in outcome:
            # c.f. equation 3 in paper
            transition_likelihood += p_min * prob
        else:
            grounded_outcome = {ground_literal(lit, sigma) for lit in outcome}
            if sorted(grounded_outcome) == sorted(effects):
                transition_likelihood += prob
    return transition_likelihood

def score_action_rule_set(action_rule_set, transitions_for_action, p_min=P_MIN, alpha=ALPHA):
    """Score a full rule set for an action

    Parameters
    ----------
    action_rule_set : NDRSet
    transitions_for_action : [ (set, Literal, set) ]
        List of (state, action, effects).
    """
    score = 0.

    # Calculate penalty for number of literals
    for rule in action_rule_set:
        pen = get_pen(rule)
        score += - alpha * pen

    # Calculate transition likelihoods per example and accumulate score
    for transition in transitions_for_action:
        # Figure out which rule covers the transition
        selected_ndr = action_rule_set.find_rule(transition)
        # Calculate transition likelihood
        transition_likelihood = get_transition_likelihood(transition, 
            selected_ndr, p_min=p_min)
        # Terminate early if likelihood is -inf
        if transition_likelihood == 0.:
            return -10e8
        # Add to score
        score += np.log(transition_likelihood)

    return score

def score_rule(rule, transitions_for_rule, p_min=P_MIN, alpha=ALPHA, compute_penalty=True):
    """Score a single rule on examples that it covers

    Parameters
    ----------
    rule : NDR
    transitions_for_rule : [ (set, Literal, set) ]
        List of (state, action, effects).
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


## Learn parameters
def learn_parameters(rule, covered_transitions, maxiter=100):
    """Learn effect probabilities given the rest of a rule

    Parameters
    ----------
    rule : NDR
    covered_transitions : [(set, Literal, set)]
    """
    # First check whether all of the rule effects are mutually exclusive.
    # If so, we can compute analytically!
    try:
        return learn_params_analytically(rule, covered_transitions)
    except MultipleOutcomesPossible:
        pass

    # Set up the loss
    def loss(x):
        rule.effect_probs = x
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

    # Finish rule
    rule.effect_probs = params

def learn_params_analytically(rule, covered_transitions):
    """Assuming effects are mutually exclusive, find best params"""
    effect_counts = [0. for _ in rule.effects]
    for transition in covered_transitions:
        # Throws a caught error if there is no unique matching effect
        idx = rule.find_unique_matching_effect_index(transition)
        effect_counts[idx] += 1
    rule.effect_probs = np.array(effect_counts) / np.sum(effect_counts)


## Induce outcomes
class InduceOutcomesSearchOperator(SearchOperator):
    """Boilerplate for searching over effect distributions
    """
    def __init__(self, rule, covered_transitions):
        self._rule_copy = rule.copy() # feel free to modify in-place
        self._covered_transitions = covered_transitions

    def get_children(self, probs_and_effects):
        """Get new effects, get new probs and scores, then yield
        """
        _, effects = probs_and_effects
        for new_effects in self.get_child_effects(effects):
            new_probs = self.get_probs(new_effects)
            score = self.get_score(new_probs, new_effects)
            yield (new_probs, new_effects)

    def get_probs(self, effects):
        self._rule_copy.effects = effects
        learn_parameters(self._rule_copy, self._covered_transitions)

    def get_score(self, probs, effects):
        self._rule_copy.effect_probs = probs
        self._rule_copy.effects = effects
        return score_rule(self._rule_copy, self._covered_transitions)

    @abc.abstractmethod
    def get_child_effects(self, effects):
        raise NotImplementedError()


class InduceOutcomesAddOperator(InduceOutcomesSearchOperator):
    """Pick a pair of non-contradictory outcomes and conjoin them
       (making sure not to conjoin with noiseoutcome)
    """
    def get_child_effects(self, effects):
        for i in range(len(effects)-1):
            if NOISE_OUTCOME in effects[i]:
                continue
            for j in range(i+1, len(efects)):
                if NOISE_OUTCOME in effects[j]:
                    continue
                # Check for contradiction
                contradiction = False
                for lit_i in effects[i]:
                    if contradiction:
                        break
                    for lit_j in effects[j]:
                        if Anti(lit_i.predicate) == lit_j:
                            contradiction = True
                            break
                if contradiction:
                    continue
                # Create new set of effects that combines the two
                combined_effects = sorted(set(effects[i]) | set(effects[j]))
                # Get the other effects
                new_effects = []
                for k in range(len(effects))
                    if k in [i, j]:
                        continue
                    new_effects.append(effects[k])
                # Add the new effect
                new_effects.append(combined_effects)
                yield new_effects


class InduceOutcomesRemoveOperator(InduceOutcomesSearchOperator):
    """Drop an outcome (not the noise one though!)
    """
    def get_child_effects(effects):
        for i, effect_i in enumerate(effects):
            if NOISE_OUTCOME in effect_i:
                continue
            new_effects = [e for j, e in enumerate(effects) if j != i]
            yield new_effects


def create_induce_outcomes_operators(rule, covered_transitions):
    """Search operators for outcome induction
    """
    add_operator = InduceOutcomesAddOperator(rule, covered_transitions)
    remove_operator = InduceOutcomesRemoveOperator(rule, covered_transitions)
    return [add_operator, remove_operator]

def get_all_possible_outcomes(rule, covered_transitions):
    """Create initial outcomes as all possible ones
    """
    # For default rule, the only possible outcomes are noise and nothing
    if len(rule.preconditions) == 0:
        all_possible_outcomes = { (NOISE_OUTCOME,), tuple() }
    else:
        all_possible_outcomes = { (NOISE_OUTCOME,) }
        for state, action, effects in covered_transitions:
            sigma = rule.find_substitutions(state, action)
            assert sigma is not None
            inverse_sigma = {v : k for k, v in sigma.items()}
            lifted_effects = {ground_literal(e, inverse_sigma) for e in effects}
            all_possible_outcomes.add(tuple(sorted(lifted_effects)))
    return all_possible_outcomes

def induce_outcomes(rule, covered_transitions, max_node_expansions=100):
    """Induce outcomes for a rule

    Modifies the rule in place.
    """
    # Initialize effects with uniform distribution over all possible outcomes
    all_possible_outcomes = get_all_possible_outcomes(rule, covered_transitions)
    num_possible_outcomes = len(all_possible_outcomes)
    rule.effect_probs = [1./num_possible_outcomes] * num_possible_outcomes
    rule.effects = [list(outcome) for outcome in sorted(all_possible_outcomes)]
    # Search for better parameters
    learn_parameters(rule, covered_transitions)
    # Search for better effects
    init_state = (rule.effect_probs, rule.effects)
    init_score = score_rule(rule, covered_transitions)
    search_operators = create_induce_outcomes_operators(rule, covered_transitions)
    best_probs, best_effects = run_greedy_search(search_operators, init_state, init_score,
        max_node_expansions=max_node_expansions)
    rule.effect_probs = best_probs
    rule.effects = best_effects

## Main search operators
def create_default_rule_set(action, transitions_for_action):
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


class TrimPreconditionsSearchOperator(SearchOperator):
    """Helper for ExplainExamples step 2
    """
    def __init__(self, rule, transitions):
        self._rule = rule
        self._transitions = transitions

    def get_score(self, preconditions):
        """Get a score for a possible set of preconditions
        """
        rule = self._rule.copy()
        rule.preconditions = preconditions
        rule_set = NDRSet([rule])
        # Induce outcomes for both rules
        rule_transitions, default_transitions = \
            rule_set.partition_transitions(self.transitions)
        induce_outcomes(rule, rule_transitions)
        induce_outcomes(rule_set.default_ndr, default_transitions)
        return score_action_rule_set(rule_set, self.transitions)

    def get_children(self, remaining_preconditions):
        for i in range(len(remaining_preconditions)):
            child_preconditions = [remaining_preconditions[j] \
                for j in range(len(remaining_preconditions)) if i != j]
            score = self.get_score(child_preconditions)
            yield score, child_preconditions


class ExplainExamples(SearchOperator):
    """Explain examples, the beefiest search operator

    Tries to follow the pseudocode in the paper as faithfully as possible
    """
    def __init__(self, action, transitions_for_action):
        self.action = action
        self.transitions_for_action = transitions_for_action
        self.unique_transitions = get_unique_transitions(transitions_for_action)

    def _get_default_transitions(self, action_rule_set):
        """Get unique transitions that are covered by the default rule
        """
        default_transitions = []
        for transition in self.unique_transitions:
            covering_rule = action_rule_set.find_rule(transition)
            if covering_rule == action_rule_set.default_ndr:
                default_transitions.append(transition)
        return default_transitions

    def _initialize_new_rule(self, transition):
        """Step 1: Create a new rule
        """
        s, a, effs = transition
        new_rule = NDR(action=None, preconditions=[], effect_probs=[], effects=[])
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
                lifted_lit = ground_literal(lit, sigma_inverse)
                new_rule.preconditions.append(lifted_lit)
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
                    lifted_lit = ground_literal(lit, sigma_inverse)
                    d.append(lifted_lit)
            # Check if d uniquely refers to c in s
            new_rule_copy = new_rule.copy()
            new_rule_copy.preconditions.extend(d)
            if new_rule_copy.covers_transition(transition):
                new_rule.preconditions.extend(d)
        # Step 1.3: Complete the rule
        # Call InduceOutComes to create the rule's outcomes.
        covered_transitions = new_rule.get_covered_transitions(self.transitions_for_action)
        induce_outcomes(new_rule, covered_transitions)
        assert new_rule.effects is not None
        if DEBUG: import ipdb; ipdb.set_trace()
        return new_rule

    def _trim_preconditions(self, rule):
        """Step 2: Trim literals from the rule
        """
        # Create a rule set R' containing r and the default rule
        # Greedily trim literals from r, ensuring that r still covers (s, a, s')
        # and filling in the outcomes using InduceOutcomes until R's score stops improving
        op = TrimPreconditionsSearchOperator(rule, self.transitions_for_action)
        init_state = list(new_rule.preconditions)
        init_score = op.get_score(init_state)
        best_preconditions = run_greedy_search([op], init_state, init_score)
        rule.preconditions = best_preconditions
        if DEBUG: import ipdb; ipdb.set_trace()

    def _create_new_rule_set(self, old_rule_set, new_rule):
        """Step 3: Create a new rule set containing the new rule
        """
        # Create a new rule set R' = R
        new_rules = [new_rule]
        # Add r to R' and remove any rules in R' that cover any examples r covers
        for t in self.transitions_for_action:
            if not new_rule.covers_transition(t):
                continue
            # leave out default rule
            for rule in old_rule_set.ndrs:
                if not rule.covers_transition(t):
                    new_rules.append(rule)
        # New rule set
        new_rule_set = NDRSet(new_rules)
        # Recompute the parameters of the default rule
        default_rule = new_rule_set.default_ndr
        partitions = new_rule_set.partition_transitions(self.transitions_for_action)
        covered_transitions = partitions[-1]
        induce_outcomes(default_rule, covered_transitions)
        if DEBUG: import ipdb; ipdb.set_trace()
        return new_rule_set

    def get_children(self, action_rule_set):
        """The successor
        """
        # Get unique transitions that are covered by the default rule
        transitions = self._get_default_transitions(action_rule_set)

        for i, transition in enumerate(transitions):
            if not DEBUG:
                print("Running explain examples for action {} {}/{}".format(action, i, 
                    len(unique_transitions)), end='\r')
                if i == len(unique_transitions) -1:
                    print()
            if DEBUG: print("Considering explaining example for transition")
            if DEBUG: print_transition(transition)
            # Step 1: Create a new rule
            new_rule = self._initialize_new_rule(transition)
            # Step 2: Trim literals from r
            self._trim_preconditions(new_rule)
            # If preconditions are empty, don't enumerate; this should be covered by the default rule
            if len(new_rule.preconditions) == 0:
                continue
            # Step 3: Create a new rule set containing r
            new_rule_set = self._create_new_rule_set(action_rule_set, new_rule)
            # Add R' to the return rule sets R_O
            score = score_action_rule_set(new_rule_set, self.transitions_for_action)
            yield score, new_rule_set

# def create_drop_rules_operator(action, transitions_for_action):
#     """Search operator that drops one rule from the set
#     """
#     def drop_rules(action_rule_set):
#         print("Running drop rules")
#         # Don't drop the default rule
#         for i in range(len(action_rule_set)-1):
#             remaining_rules = [action_rule_set[j].copy() for j in range(len(action_rule_set)) \
#                 if i != j]
#             induce_outcomes(remaining_rules[-1], transitions_for_action,
#                     rule_is_default=True, action_rule_set=remaining_rules)
#             score = score_action_rule_set(remaining_rules, transitions_for_action)
#             yield score, remaining_rules
#     return drop_rules

# def create_drop_lits_operator(action, transitions_for_action):
#     """Search operator that drops one lit per rule from the set
#     """
#     def drop_lits(action_rule_set):
#         print("Running drop lits")
#         # Don't drop the default rule
#         for i in range(len(action_rule_set)-1):
#             num_preconds = len(action_rule_set[i].preconditions)
#             if num_preconds <= 1:
#                 continue
#             for drop_i in range(num_preconds):
#                 remaining_rules = [rule.copy() for rule in action_rule_set]
#                 del remaining_rules[i].preconditions.literals[drop_i]
#                 induce_outcomes(remaining_rules[i], transitions_for_action)
#                 induce_outcomes(remaining_rules[-1], transitions_for_action,
#                     rule_is_default=True, action_rule_set=remaining_rules)
#                 score = score_action_rule_set(remaining_rules, transitions_for_action)
#                 yield score, remaining_rules
#     return drop_lits

# def create_add_lits_operator(action, transitions_for_action):
#     """Search operator that adds one lit per rule from the set
#     """
#     # Get all possible lits to add
#     all_possible_additions = set()
#     unique_transitions = get_unique_transitions(transitions_for_action)

#     for i, transition in enumerate(unique_transitions):
#         s, a, effs = transition
#         variable_name_generator = iter_variable_names()
#         variables = [next(variable_name_generator) for _ in a.variables]
#         sigma = dict(zip(variables, a.variables))
#         sigma_inverse = {v : k for k, v in sigma.items()}
#         for lit in s:
#             if all(val in sigma_inverse for val in lit.variables):
#                 lifted_lit = lit.predicate(*[sigma_inverse[val] for val in lit.variables])
#                 all_possible_additions.add(lifted_lit)
#         changed_objects = set()
#         for lit in effs:
#             for val in lit.variables:
#                 if val not in sigma_inverse:
#                     changed_objects.add(val)
#         for c in sorted(changed_objects):
#             # Create a new variable and extend sigma to map v to c
#             new_variable = next(variable_name_generator)
#             sigma[new_variable] = c
#             assert c not in sigma_inverse
#             sigma_inverse[c] = new_variable
#             for lit in s:
#                 if c not in lit.variables:
#                     continue
#                 if all(val in sigma_inverse for val in lit.variables):
#                     lifted_lit = lit.predicate(*[sigma_inverse[val] for val in lit.variables])
#                     all_possible_additions.add(lifted_lit)

#     all_possible_additions = sorted(all_possible_additions)

#     def add_lits(action_rule_set):
#         print("Running add lits")
#         # Don't add to the default rule
#         for i in range(len(action_rule_set)-1):
#             for new_lit in all_possible_additions:
#                 remaining_rules = [rule.copy() for rule in action_rule_set]
#                 if new_lit in remaining_rules[i].preconditions:
#                     continue
#                 remaining_rules[i].preconditions.literals.append(new_lit)
#                 induce_outcomes(remaining_rules[i], transitions_for_action)
#                 induce_outcomes(remaining_rules[-1], transitions_for_action,
#                     rule_is_default=True, action_rule_set=remaining_rules)
#                 score = score_action_rule_set(remaining_rules, transitions_for_action)
#                 import ipdb; ipdb.set_trace()
#                 yield score, remaining_rules
#     return add_lits


def get_search_operators(action, transitions_for_action):
    """Main search operators
    """
    explain_examples = create_explain_examples_operator(action, transitions_for_action)
    # drop_rules = create_drop_rules_operator(action, transitions_for_action)
    # drop_lits = create_drop_lits_operator(action, transitions_for_action)
    # add_lits = create_add_lits_operator(action, transitions_for_action)

    return [
        explain_examples, 
        # add_lits, 
        # drop_rules,
        # drop_lits
    ]

## Main
def run_main_search(transition_dataset, max_node_expansions=1000, rng=None, 
                    search_method="greedy"):
    """Run the main search
    """
    rule_sets = {}

    for action, transitions_for_action in transition_dataset.items():
        print("Running search for action", action)

        search_operators = get_search_operators(action, transitions_for_action)
        init_score, init_state = create_default_rule_set(action, transitions_for_action)

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
