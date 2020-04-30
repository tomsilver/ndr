from ndr.ndrs import *
from ndr.learn import *
from ndr.structs import Type, Anti

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


if __name__ == "__main__":
    test_ndr()
    test_ndr_set()

