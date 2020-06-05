from ndr.ndrs import *
from ndr.learn import *
from pddlgym.structs import Type, Anti
from ndr.main import *
from ndr.utils import nostdout
import gym
import pddlgym
import numpy as np


# Some shared stuff
block_type = Type("block")
Act0 = Predicate("act0" , 0, [])
Act01 = Predicate("act01" , 0, [])
Act1 = Predicate("act1" , 0, [block_type])
Red = Predicate("red", 1, [block_type])
Blue = Predicate("blue", 1, [block_type])
HandsFree = Predicate("handsfree", 0, [])

def test_ndr():
    def create_ndr():
        action = Act0()
        preconditions = [Red("?x"), HandsFree()]
        effect_probs = [0.8, 0.2]
        effects = [{Anti(HandsFree())}, {NOISE_OUTCOME}]
        return NDR(action, preconditions, effect_probs, effects)

    # Test copy
    ndr = create_ndr()
    ndr_copy = ndr.copy()
    ndr.preconditions.remove(HandsFree())
    assert HandsFree() not in ndr.preconditions
    assert HandsFree() in ndr_copy.preconditions
    del ndr.effects[0]
    assert len(ndr.effects) == 1
    assert len(ndr_copy.effects) == 2
    ndr.effect_probs[0] = 1.0
    assert ndr.effect_probs[0] == 1.0
    assert ndr_copy.effect_probs[0] == 0.8

    # Test find substitutions
    ndr = create_ndr()
    state = {Red("block0")}
    action = Act0()
    assert ndr.find_substitutions(state, action) == None
    state = {Red("block0"), HandsFree(), Blue("block1")}
    action = Act0()
    sigma = ndr.find_substitutions(state, action)
    assert sigma is not None
    assert len(sigma) == 1
    assert sigma[block_type("?x")] == block_type("block0")
    state = {Red("block0"), HandsFree(), Red("block1")}
    action = Act0()
    assert ndr.find_substitutions(state, action) == None

    # Test find_unique_matching_effect_index
    ndr = create_ndr()
    state = {Red("block0"), HandsFree(), Blue("block1")}
    action = Act0()
    effects = {Anti(HandsFree())}
    assert ndr.find_unique_matching_effect_index((state, action, effects)) == 0
    state = {Red("block0"), HandsFree(), Blue("block1")}
    action = Act0()
    effects = {Anti(HandsFree()), Blue("block0")}
    assert ndr.find_unique_matching_effect_index((state, action, effects)) == 1

    print("Test NDR passed.")

def test_ndr_set():
    def create_ndr_set():
        action = Act0()
        preconditions = [Red("?x"), HandsFree()]
        effect_probs = [0.8, 0.2]
        effects = [{Anti(HandsFree())}, {NOISE_OUTCOME}]
        ndr0 = NDR(action, preconditions, effect_probs, effects)
        preconditions = [Red("?x"), Blue("?x")]
        effect_probs = [0.5, 0.4, 0.1]
        effects = [{HandsFree()}, {Anti(Blue("?x"))}, {NOISE_OUTCOME}]
        ndr1 = NDR(action, preconditions, effect_probs, effects)
        return NDRSet(action, [ndr0, ndr1])

    # Test find rule
    ndr_set = create_ndr_set()
    state = {Red("block0")}
    action = Act0()
    assert ndr_set.find_rule((state, action, set())) == ndr_set.default_ndr
    state = {Red("block0"), HandsFree(), Blue("block1")}
    action = Act0()
    assert ndr_set.find_rule((state, action, set())) == ndr_set.ndrs[0]
    state = {Red("block0"), Blue("block0")}
    action = Act0()
    assert ndr_set.find_rule((state, action, set())) == ndr_set.ndrs[1]

    # Test partition transitions
    transitions = [
        ({Red("block0"), HandsFree(), Blue("block1")}, Act0(), set()),
        ({Red("block0"), Blue("block0")}, Act0(), set()),
        ({Red("block0"), Blue("block0"), Blue("block1")}, Act0(), set()),
        ({Red("block0")}, Act0(), set()),
    ]
    partitions = ndr_set.partition_transitions(transitions)
    assert partitions[0] == [transitions[0]]
    assert partitions[1] == transitions[1:3]
    assert partitions[2] == [transitions[3]]
    print("Test NDRSet passed.")

def test_planning():
    print("Running planning test with ground-truth NDRs")
    with nostdout():
        env = NDRBlocksEnv()
        env.seed(0)
        policy = find_policy("ff_replan", env.operators, env.action_space, env.observation_space)
        total_returns = 0
        for trial in range(10):
            returns = run_policy(env, policy, verbose=False, render=False, check_reward=False)
            total_returns += returns
    # print("Average returns:", total_returns/10.)
    assert total_returns == 6


def test_integration():
    seed = 0
    print("Running integration tests (this will take a long time)")

    # Test Hanoi
    with nostdout():
        training_env = gym.make("PDDLEnvHanoi-v0")
        training_env.seed(seed)
        training_data = collect_training_data(training_env,
            num_transitions_per_problem=10,
            max_transitions_per_action=500)
        training_env.close()
        rule_set = learn_rule_set(training_data)
        test_env = gym.make("PDDLEnvHanoiTest-v0")
        test_results = run_test_suite(rule_set, test_env, render=False, verbose=False,
            num_problems=5,
            max_num_steps=10000)
        test_env.close()
        assert np.sum(test_results) == 5
    print("Hanoi integration test passed.")

    # Test TSP
    with nostdout():
        training_env = gym.make("PDDLEnvTsp-v0")
        training_env.seed(seed)
        training_data = collect_training_data(training_env,
            max_num_trials=5000,
            num_transitions_per_problem=100,
            max_transitions_per_action=2500,)
        training_env.close()
        rule_set = learn_rule_set(training_data)
        test_env = gym.make("PDDLEnvTspTest-v0")
        test_results = run_test_suite(rule_set, test_env, render=False, verbose=False,
            num_problems=5,
            max_num_steps=10000)
        test_env.close()
        assert np.sum(test_results) == 5
    print("TSP integration test passed.")

    # Test Doors
    # Currently broken due to two preconditions involving non-referenced objects
    # with nostdout():
    #     training_env = gym.make("PDDLEnvDoors-v0")
    #     training_env.seed(seed)
    #     training_data = collect_training_data(training_env,
    #         num_transitions_per_problem=10,
    #         max_transitions_per_action=500)
    #     training_env.close()
    #     rule_set = learn_rule_set(training_data)
    #     test_env = gym.make("PDDLEnvDoorsTest-v0")
    #     test_results = run_test_suite(rule_set, test_env, render=False, verbose=False,
    #         num_problems=5,
    #         max_num_steps=10000)
    #     test_env.close()
    #     assert np.sum(test_results) == 5
    # print("TSP integration test passed.")

    # Test Rearrangement
    # Currently broken due to two preconditions involving non-referenced objects
    # with nostdout():
    #     training_env = gym.make("PDDLEnvRearrangement-v0")
    #     training_env.seed(seed)
    #     training_data = collect_training_data(training_env,
    #         num_transitions_per_problem=10,
    #         max_transitions_per_action=100)
    #     training_env.close()
    #     rule_set = learn_rule_set(training_data)
    #     test_env = gym.make("PDDLEnvRearrangement-v0")
    #     test_results = run_test_suite(rule_set, test_env, render=False, verbose=False,
    #         num_problems=5,
    #         max_num_steps=10000)
    #     test_env.close()
    #     assert np.sum(test_results) == 5
    # print("TSP integration test passed.")

    # Test NDRBlocks
    with nostdout():
        training_env = NDRBlocksEnv()
        training_env.seed(seed)
        training_data = collect_training_data(training_env)
        training_env.close()
        rule_set = learn_rule_set(training_data)
        test_env = NDRBlocksEnv()
        test_results = run_test_suite(rule_set, test_env, render=False, verbose=False,
            num_problems=100)
        test_env.close()
        assert 40 < np.sum(test_results) < 60
    print("NDRBlocks integration test passed.")

    # Test PybulletBlocksEnv
    with nostdout():
        training_env = PybulletBlocksEnv(use_gui=False)
        training_env.seed(seed)
        training_data = collect_training_data(training_env,
            max_num_trials=5000,
            num_transitions_per_problem=1,
            max_transitions_per_action=500)
        training_env.close()
        rule_set = learn_rule_set(training_data)
        test_env = PybulletBlocksEnv(use_gui=False)
        test_results = run_test_suite(rule_set, test_env, render=False, verbose=False)
        test_env.close()
        assert np.sum(test_results) == 8.0
    print("PybulletBlocksEnv integration test passed.")

    print("Integration tests passed.")


if __name__ == "__main__":
    import time
    start_time = time.time()
    test_ndr()
    test_ndr_set()
    test_planning()
    test_integration()
    print("Tests completed in {} seconds".format(time.time() - start_time))
