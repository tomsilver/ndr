from utils import run_random_agent_demo, run_plan
from envs.ndr_blocks import NDRBlocksEnv, pickup, puton, putontable


def run_all(render=True, verbose=True):
    env = NDRBlocksEnv()
    plan = [pickup("A"), puton("B"), pickup("C"), puton("A")]
    run_plan(env, plan, verbose=verbose, render=render, check_reward=False)
    # run_random_agent_demo(env, verbose=verbose, seed=0, render=render)


if __name__ == '__main__':
    run_all()
