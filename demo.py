from planning import find_policy
from utils import run_random_agent_demo, run_plan, run_policy
from envs.ndr_blocks import NDRBlocksEnv, pickup, puton, putontable


def run_all(render=True, verbose=True):
    env = NDRBlocksEnv()
    initial_state, debug_info = env.reset()
    goal = debug_info["goal"]
    policy = find_policy("ff_replan", initial_state, goal, env.operators, env.action_space, env.observation_space)
    run_policy(env, policy, verbose=verbose, render=render)
    # plan = [pickup("A"), puton("B"), pickup("C"), puton("A")]
    # run_plan(env, plan, verbose=verbose, render=render)
    # run_random_agent_demo(env, verbose=verbose, seed=0, render=render)


if __name__ == '__main__':
    run_all()
