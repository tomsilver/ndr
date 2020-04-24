"""A blocks environment written with hardcoded NDR rules.

Based on the environment described in ZPK.
"""
from .rendering.block_words import render as _render
from .rendering.block_words import get_objects_from_obs as get_piles
from ndr.structs import Predicate, LiteralConjunction, Type, Anti, NDR, ground_literal
from .spaces import LiteralSpace, LiteralSetSpace
from ndr.inference import find_satisfying_assignments


import gym
import numpy as np


# Object types
block_type = Type("block")

# State predicates
on = Predicate("on", 2, [block_type, block_type])
ontable = Predicate("ontable", 1, [block_type])
holding = Predicate("holding", 1, [block_type])
clear = Predicate("clear", 1, [block_type])
handempty = Predicate("handempty", 0, [])

# Noise effect
noiseoutcome = Predicate("noiseoutcome", 0, [])

# Actions
pickup = Predicate("pickup", 1, [block_type])
puton = Predicate("puton", 1, [block_type])
putontable = Predicate("putontable", 0, [])


class NDRBlocksEnv(gym.Env):
    action_predicates = [pickup, puton, putontable]
    observation_predicates = [on, ontable, holding, clear, handempty, noiseoutcome]

    def __init__(self, seed=0):
        self.action_space = LiteralSpace(self.action_predicates)
        self.action_space.seed(seed)
        self.observation_space = LiteralSetSpace(set(self.observation_predicates))
        self._rng = np.random.RandomState(seed)

        self.operators = {
            # pickup : [
            #     # If you try to pickup something while already holding something else,
            #     # you'll probably drop the thing that you're holding
            #     NDR(action=pickup("?x"), 
            #         preconditions={holding("?y")},
            #         effects=[
            #             (1., {Anti(holding("?y")), ontable("?y"), handempty(), clear("?y")}),
            #             (0., set()),
            #             (0., {noiseoutcome()}),
            #         ],
            #     ),
            #     # If you try pickup something clear while it's on something else, you
            #     # probably will succeed
            #     NDR(action=pickup("?x"), 
            #         preconditions={on("?x", "?y"), clear("?x"), handempty()},
            #         effects=[
            #             (1., {holding("?x"), Anti(on("?x", "?y")), clear("?y"), 
            #                    Anti(handempty()), Anti(clear("?x"))}),
            #             (0., set()),
            #             (0., {noiseoutcome()}),
            #         ],
            #     ),
            #     # If you try pickup something clear while it's on the table, you
            #     # probably will succeed
            #     NDR(action=pickup("?x"), 
            #         preconditions={ontable("?x"), clear("?x"), handempty()},
            #         effects=[
            #             (1., {holding("?x"), Anti(ontable("?x")), Anti(handempty()), 
            #                    Anti(clear("?x"))}),
            #             (0., set()),
            #             (0., {noiseoutcome()}),
            #         ],
            #     ),
            #     # Default rule
            #     NDR(action=pickup("?x"), 
            #         preconditions={},
            #         effects=[
            #             (1., set()),
            #             (0., {noiseoutcome()}),
            #         ],
            #     ),
            # ],
            pickup : [
                # If you try to pickup something while already holding something else,
                # you'll probably drop the thing that you're holding
                NDR(action=pickup("?x"), 
                    preconditions={holding("?y")},
                    effects=[
                        (0.6, {Anti(holding("?y")), ontable("?y"), handempty(), clear("?y")}),
                        (0.3, set()),
                        (0.1, {noiseoutcome()}),
                    ],
                ),
                # If you try pickup something clear while it's on something else, you
                # probably will succeed
                NDR(action=pickup("?x"), 
                    preconditions={on("?x", "?y"), clear("?x"), handempty()},
                    effects=[
                        (0.7, {holding("?x"), Anti(on("?x", "?y")), clear("?y"), 
                               Anti(handempty()), Anti(clear("?x"))}),
                        (0.1, set()),
                        (0.2, {noiseoutcome()}),
                    ],
                ),
                # If you try pickup something clear while it's on the table, you
                # probably will succeed
                NDR(action=pickup("?x"), 
                    preconditions={ontable("?x"), clear("?x"), handempty()},
                    effects=[
                        (0.8, {holding("?x"), Anti(ontable("?x")), Anti(handempty()), 
                               Anti(clear("?x"))}),
                        (0.1, set()),
                        (0.1, {noiseoutcome()}),
                    ],
                ),
                # Default rule
                NDR(action=pickup("?x"), 
                    preconditions={},
                    effects=[
                        (0.9, set()),
                        (0.1, {noiseoutcome()}),
                    ],
                ),
            ],
            puton : [
                # If you try to puton something that is clear, it
                # probably will succeed
                NDR(action=puton("?x"), 
                    preconditions={clear("?x"), holding("?y")},
                    effects=[
                        (0.8, {Anti(holding("?y")), on("?y", "?x"), handempty(), 
                               Anti(clear("?x")), clear("?y")}),
                        (0.1, set()),
                        (0.1, {noiseoutcome()}),
                    ],
                ),
                # Removing this case because I don't know how to handle derived predicates like this
                # If you try to puton something that is in the middle, it
                # probably will result in stacking on the top of the stack
                # NDR(action=puton("?x"), 
                #     preconditions={clear("?z"), above("?z", "?x"), holding("?y")},
                #     effects=[
                #         (0.6, {Anti(holding("?y")), on("?y", "?z")}),
                #         (0.1, set()),
                #         (0.3, {noiseoutcome()}),
                #     ],
                # ),
                # Default rule
                NDR(action=puton("?x"), 
                    preconditions={},
                    effects=[
                        (0.9, set()),
                        (0.1, {noiseoutcome()}),
                    ],
                ),
            ],
            putontable : [
                # If you try to putontable and you're holding something,
                # it will probably succeed
                NDR(action=putontable(), 
                    preconditions={holding("?x")},
                    effects=[
                        (0.8, {Anti(holding("?x")), ontable("?x"), clear("?x"), handempty()}),
                        (0.1, set()),
                        (0.1, {noiseoutcome()}),
                    ],
                ),
                # Default rule
                NDR(action=putontable(), 
                    preconditions={},
                    effects=[
                        (0.9, set()),
                        (0.1, {noiseoutcome()}),
                    ],
                ),
            ],
        }

        # (initial state, goal)
        self.problems = [
            ({ on("a", "b"), on("b", "c"), ontable("c"), clear("a"), 
               handempty() },
               LiteralConjunction([on("c", "a"), on("a", "b")])),
            ({ on("a", "b"), ontable("b"), ontable("c"), clear("a"), 
               clear("c"), handempty() },
               LiteralConjunction([on("c", "a"), on("a", "b")])),
            ({ ontable("a"), ontable("b"), ontable("c"), clear("a"), 
               clear("b"), clear("c"), handempty() },
               LiteralConjunction([on("c", "a"), on("a", "b")])),
        ]
        self.num_problems = len(self.problems)
        self._problem_idx = None

    def fix_problem_index(self, idx):
        self._problem_idx = idx

    def reset(self):
        if self._problem_idx is None:
            self._problem_idx = self._rng.choice(self.num_problems)
        self._state, self._goal = self.problems[self._problem_idx]
        self._problem_objects = sorted({ v for lit in self._state for v in lit.variables })
        self.action_space.update(self._problem_objects)
        return self._get_observation(), self._get_debug_info()

    def step(self, action):
        ndr_list = self.operators[action.predicate]
        full_state = self._get_full_state()
        effects = self._sample_effects(ndr_list, full_state, action, self._rng)
        self._state = self._execute_effects(self._state, effects)
        done = self._is_goal_reached() or noiseoutcome() in self._state
        reward = float(self._is_goal_reached())
        return self._get_observation(), reward, done, self._get_debug_info()

    def render(self, *args, **kwargs):
        obs = self._get_observation()
        return _render(obs, *args, **kwargs)

    def _get_observation(self):
        return self._get_full_state()

    def _get_debug_info(self):
        return { "goal" : self._goal }

    def _get_full_state(self, include_possible_actions_in_state=False):
        obs = self._state.copy()
        if include_possible_actions_in_state:
            obs |= self.action_space.all_ground_literals()
        if noiseoutcome() in obs:
            return { noiseoutcome() }
        return obs

    def _is_goal_reached(self):
        return self._goal.holds(self._state)

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
                import ipdb; ipdb.set_trace()
                raise Exception("Multiple rules applied to one state action.")
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


