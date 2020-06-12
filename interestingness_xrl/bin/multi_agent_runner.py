__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import argparse
import multiprocessing as mp
from interestingness_xrl.bin.agent_runner import run_trial
from interestingness_xrl.scenarios import AgentType

DEF_AGENT_TYPE = AgentType.Learning
DEF_TRIALS = 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multiple RL Agent runner')
    parser.add_argument('-a', '--agent', help='agent type', type=int, default=DEF_AGENT_TYPE)
    parser.add_argument('-o', '--output', help='directory in which to store results')
    parser.add_argument('-r', '--results', help='directory from which to load results')
    parser.add_argument('-c', '--config', help='path to config file')
    parser.add_argument('-t', '--trials', help='number of trials to run', type=int, default=DEF_TRIALS)
    parser.add_argument('-rv', '--record', help='record videos according to linear schedule', action='store_true')
    parser.add_argument('-v', '--verbose', help='print information to the console', action='store_true')
    parser.add_argument('-p', '--processes', type=int, default=0,
                        help='Number of parallel processes to use. 0 indicates all available CPUs.')
    args = parser.parse_args()

    # processes each environment in parallel or sequentially
    pool = mp.Pool(args.processes if args.processes > 0 else None)
    pool.map(run_trial, [argparse.Namespace(**dict({'trial': i}, **vars(args))) for i in range(args.trials)])
