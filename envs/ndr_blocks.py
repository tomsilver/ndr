"""A blocks environment written with hardcoded NDR rules.

Based on the environment described in ZPK.
"""
from .rendering.block_words import render as _render
from .rendering.block_words import get_objects_from_obs as get_piles
from ndr.structs import Predicate, Type, Anti, NDR, ground_literal
from .spaces import LiteralSpace
from ndr.inference import find_satisfying_assignments


import gym
import numpy as np


# Object types
block_type = Type("block")

# State predicates
on = Predicate("on", 2, [block_type, block_type])
ontable = Predicate("ontable", 1, [block_type])
holding = Predicate("holding", 1, [block_type])

# Derived predicates
above = Predicate("above", 2, [block_type, block_type])
clear = Predicate("clear", 1, [block_type])
handempty = Predicate("handempty", 0, [])

# Noise effect
noiseoutcome = Predicate("noiseoutcome", 0)

# Actions
pickup = Predicate("pickup", 1, [block_type])
puton = Predicate("puton", 1, [block_type])
putontable = Predicate("putontable", 1, [block_type])


class NDRBlocksEnv(gym.Env):
    action_predicates = [pickup, puton, putontable]

    def __init__(self, seed=0):
        self.action_space = LiteralSpace(self.action_predicates)
        self._rng = np.random.RandomState(seed)

        self._action_predicate_to_ndr_list = {
            pickup : [
                # If you try to pickup something while already holding something else,
                # you'll probably drop the thing that you're holding
                NDR(action=pickup("X"), 
                    preconditions={holding("Y")},
                    effects=[
                        (0.6, {Anti(holding("Y")), ontable("Y")}),
                        (0.3, set()),
                        (0.1, {noiseoutcome()}),
                    ],
                ),
                # If you try pickup something clear while it's on something else, you
                # probably will succeed
                NDR(action=pickup("X"), 
                    preconditions={on("X", "Y"), clear("X"), handempty()},
                    effects=[
                        (0.7, {holding("X"), Anti(on("X", "Y"))}),
                        (0.1, set()),
                        (0.2, {noiseoutcome()}),
                    ],
                ),
                # If you try pickup something clear while it's on the table, you
                # probably will succeed
                NDR(action=pickup("X"), 
                    preconditions={ontable("X"), clear("X"), handempty()},
                    effects=[
                        (0.8, {holding("X"), Anti(ontable("X"))}),
                        (0.1, set()),
                        (0.1, {noiseoutcome()}),
                    ],
                ),
                # Default rule
                NDR(action=pickup("X"), 
                    preconditions={},
                    effects=[
                        (0.6, set()),
                        (0.4, {noiseoutcome()}),
                    ],
                ),
            ],
            puton : [
                # If you try to puton something that is clear, it
                # probably will succeed
                NDR(action=puton("X"), 
                    preconditions={clear("X"), holding("Y")},
                    effects=[
                        (0.8, {Anti(holding("Y")), on("Y", "X")}),
                        (0.1, set()),
                        (0.1, {noiseoutcome()}),
                    ],
                ),
                # If you try to puton something that is in the middle, it
                # probably will result in stacking on the top of the stack
                NDR(action=puton("X"), 
                    preconditions={clear("Z"), above("Z", "X"), holding("Y")},
                    effects=[
                        (0.6, {Anti(holding("Y")), on("Y", "Z")}),
                        (0.1, set()),
                        (0.3, {noiseoutcome()}),
                    ],
                ),
                # Default rule
                NDR(action=puton("X"), 
                    preconditions={},
                    effects=[
                        (0.6, set()),
                        (0.4, {noiseoutcome()}),
                    ],
                ),
            ],
            putontable : [
                # If you try to putontable and you're holding something,
                # it will probably succeed
                NDR(action=putontable("X"), 
                    preconditions={holding("X")},
                    effects=[
                        (0.8, {Anti(holding("X")), ontable("X")}),
                        (0.1, set()),
                        (0.1, {noiseoutcome()}),
                    ],
                ),
                # Default rule
                NDR(action=putontable("X"), 
                    preconditions={},
                    effects=[
                        (0.6, set()),
                        (0.4, {noiseoutcome()}),
                    ],
                ),
            ],
        }

    def _get_derived_literals(self, state):
        piles, holding = get_piles(state)

        derived_lits = set()
        # derive above
        for pile in piles:
            for i, block_i in enumerate(pile[:-1]):
                for block_j in pile[i+1:]:
                    derived_lits.add(above(block_j, block_i))

        # derive clear
        for pile in piles:
            if len(pile) == 0:
                continue
            derived_lits.add(clear(pile[-1]))

        # derive handempty
        if holding is None:
            derived_lits.add(handempty())

        return derived_lits

    def reset(self):
        self._state = { ontable("A"), ontable("B"), ontable("C") }
        self._problem_objects = [block_type("A"), block_type("B"), block_type("C")]
        self.action_space.update(self._problem_objects)
        return self._get_observation(), {}

    def step(self, action):
        ndr_list = self._action_predicate_to_ndr_list[action.predicate]
        full_state = self._get_full_state()
        effects = self._sample_effects(ndr_list, full_state, action, self._rng)
        self._state = self._execute_effects(self._state, effects)
        return self._get_observation(), 0., False, {}

    def render(self, *args, **kwargs):
        obs = self._get_observation()
        return _render(obs, *args, **kwargs)

    def _get_observation(self):
        return self._get_full_state()

    def _get_full_state(self):
        obs = self._state.copy()
        derived_lits = self._get_derived_literals(self._state)
        return obs | derived_lits

    @staticmethod
    def _sample_effects(ndr_list, full_state, action, rng):
        kb = full_state | { action }
        for ndr in ndr_list:
            assert action.predicate == ndr.action.predicate
            conds = [ndr.action] + list(ndr.preconditions)
            assignments = find_satisfying_assignments(kb, conds)
            # Successful rule application
            if len(assignments) == 1:
                # Sample an effect set
                probs, effs = zip(*ndr.effects)
                idx = rng.choice(len(probs), p=probs)
                selected_effs = effs[idx]
                # Ground it
                grounded_effects = set()
                for lifted_effect in selected_effs:
                    effect = ground_literal(lifted_effect, assignments[0])
                    grounded_effects.add(effect)
                return grounded_effects
            elif len(assignments) > 1:
                raise Exception("Multiple rules applied to one state aciton.")
        raise Exception("No NDRs (including the default one?!) applied")

    @staticmethod
    def _execute_effects(state, effects):
        new_state = { lit for lit in state }

        # First do negative effects
        for effect in effects:
            # Negative effect
            if effect.is_anti:
                literal = effect.inverted_anti
                if literal in new_state:
                    new_state.remove(literal)
        # Now positive effects
        for effect in effects:
            if not effect.is_anti:
                new_state.add(effect)

        return new_state
