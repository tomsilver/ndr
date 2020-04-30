from pddlgym.structs import Predicate, ground_literal
from pddlgym.inference import find_satisfying_assignments
import numpy as np

### Noisy deictic rules
NOISE_OUTCOME = Predicate("noiseoutcome", 0, [])()

class MultipleOutcomesPossible(Exception):
    pass


class NDR:
    """A Noisy Deictic Rule has a lifted action, lifted preconditions, and a
       distribution over effects.

    Parameters
    ----------
    action : Literal
    preconditions : [ Literal ]
    effect_probs : np.ndarray
    effects : [Literal]
    """
    def __init__(self, action, preconditions, effect_probs, effects):
        self._action = action
        self._preconditions = preconditions
        self._effect_probs = effect_probs
        self._effects = effects

        assert isinstance(preconditions, list)
        assert len(effect_probs) == len(effects)

        # Exactly one effect should have the noise outcome
        if len(effects) > 0:
            assert sum([NOISE_OUTCOME in e for e in effects]) == 1

        self._reset_precondition_cache()
        self._reset_effect_cache()

    def __str__(self):
        effs_str = "\n        ".join(["{}: {}".format(p, eff) \
            for p, eff in zip(self.effect_probs, self.effects)])
        return """{}:
  Pre: {}
  Effs: {}""".format(self.action, self.preconditions, effs_str)

    def __repr__(self):
        return str(self)

    @property
    def action(self):
        return self._action
    
    @property
    def preconditions(self):
        return self._preconditions

    @property
    def effect_probs(self):
        return self._effect_probs

    @property
    def effects(self):
        return self._effects

    @action.setter
    def action(self, x):
        self._reset_precondition_cache()
        self._action = x

    @preconditions.setter
    def preconditions(self, x):
        self._reset_precondition_cache()
        self._preconditions = x

    @effect_probs.setter
    def effect_probs(self, x):
        # No need to reset any caches
        self._effect_probs = x

    @effects.setter
    def effects(self, x):
        self._reset_effect_cache()
        self._effects = x

    def _reset_precondition_cache(self):
        self._precondition_cache = {}
        self._reset_effect_cache()

    def _reset_effect_cache(self):
        self._effect_cache = {}

    def copy(self):
        """Create a new NDR. Literals are assumed immutable.
        """
        action = self.action
        preconditions = [p for p in self.preconditions]
        effect_probs = np.array(self.effect_probs)
        effects = [eff.copy() for eff in self.effects]
        return NDR(action, preconditions, effect_probs, effects)

    def find_substitutions(self, state, action):
        """Find a mapping from variables to objects in the state
        and action. If non-unique or none, return None.
        """
        cache_key = hash((frozenset(state), action))
        if cache_key not in self._precondition_cache:
            kb = state | { action }
            assert action.predicate == self.action.predicate
            conds = [self.action] + list(self.preconditions)
            assignments = find_satisfying_assignments(kb, conds)
            if len(assignments) != 1:
                result = None
            else:
                result = assignments[0]
            self._precondition_cache[cache_key] = result
        return self._precondition_cache[cache_key]

    def covers_transition(self, transition):
        """Check whether the action and preconditions cover the transition
        """
        state, action, effects = transition
        sigma = self.find_substitutions(state, action)
        return sigma is not None

    def get_covered_transitions(self, transitions):
        """Filter out only covered transitions
        """
        return [t for t in transitions if self.covers_transition(t)]

    def find_unique_matching_effect_index(self, transition):
        """Find the unique effect index that matches the transition.

        Note that the noise outcome always holds, but only return it
        if no other effects hold.

        Used for quickly learning effect probabilities.
        """
        state, action, effects = transition
        cache_key = hash((frozenset(state), action, frozenset(effects)))
        if cache_key not in self._effect_cache:
            sigma = self.find_substitutions(state, action)
            assert sigma is not None, "Rule assumed to cover transition"
            inverse_sigma = {v : k for k, v in sigma.items()}
            try:
                lifted_effects = {ground_literal(lit, inverse_sigma) for lit in effects}
            except KeyError:
                # Some object in the effects was not named in the rule
                lifted_effects = {NOISE_OUTCOME}
            selected_outcome_idx = None
            noise_outcome_idx = None
            for i, outcome in enumerate(self.effects):
                if NOISE_OUTCOME in outcome:
                    assert noise_outcome_idx is None
                    noise_outcome_idx = i
                else:
                    if sorted(lifted_effects) == sorted(outcome):
                        if selected_outcome_idx is not None:
                            raise MultipleOutcomesPossible()
                        selected_outcome_idx = i
            if selected_outcome_idx is not None:
                result = selected_outcome_idx
            else:
                assert noise_outcome_idx is not None
                result = noise_outcome_idx
            self._effect_cache[cache_key] = result
        return self._effect_cache[cache_key]

    def objects_are_referenced(self, state, action, objs):
        """Make sure that each object is uniquely referenced
        """
        sigma = self.find_substitutions(state, action)
        if sigma is None:
            return False
        return set(objs).issubset(set(sigma.values()))


class NDRSet:
    """A set of NDRs with a special default rule.

    Parameters
    ----------
    action : Literal
        The lifted action that all NDRs are about.
    ndrs : [ NDR ]
        The NDRs. Order does not matter.
    default_ndr : NDR or None
        If None, one is created. Only should be not
        None when an existing NDR is getting copied.
    """
    def __init__(self, action, ndrs, default_ndr=None):
        self.action = action
        self.ndrs = list(ndrs)
        if default_ndr is None:
            self.default_ndr = self._create_default_ndr(action)
        else:
            self.default_ndr = default_ndr

        # Cannot have empty preconds
        for ndr in ndrs:
            assert len(ndr.preconditions) > 0
            assert ndr.action == action
        assert self.default_ndr.action == action

    def __str__(self):
        s = "\n".join([str(r) for r in self])
        return s

    def __iter__(self):
        return iter(self.ndrs + [self.default_ndr])

    def __len__(self):
        return len(self.ndrs) + 1

    @staticmethod
    def _create_default_ndr(action):
        """Either nothing or noise happens by default
        """
        preconditions = []
        effect_probs = [0.5, 0.5]
        effects = [{ NOISE_OUTCOME }, set()]
        return NDR(action, preconditions, effect_probs, effects)

    def find_rule(self, transition):
        """Find the (assumed unique) rule that covers this transition
        """
        for ndr in self.ndrs:
            if ndr.covers_transition(transition):
                return ndr
        return self.default_ndr

    def partition_transitions(self, transitions):
        """Organize transitions by rule
        """
        rules = list(self)
        assert rules[-1] == self.default_ndr
        transitions_per_rule = [ [] for _ in rules ]
        for t in transitions:
            rule = self.find_rule(t)
            idx = rules.index(rule)
            transitions_per_rule[idx].append(t)
        return transitions_per_rule

    def copy(self):
        """Copy all NDRs in the set.
        """
        action = self.action
        ndrs = [ndr.copy() for ndr in self.ndrs]
        default_ndr = self.default_ndr.copy()
        return NDRSet(action, ndrs, default_ndr=default_ndr)
