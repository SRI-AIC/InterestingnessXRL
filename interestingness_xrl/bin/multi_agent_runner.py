__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import os
import argparse
import multiprocessing as mp
from interestingness_xrl.scenarios import AgentType

DEF_AGENT_TYPE = AgentType.Learning
DEF_TRIALS = 1


def run_trial_process(trial_num):
    # adds arguments
    proc_args = '-p -t {}'.format(trial_num)
    if args.agent is not None:
        proc_args += ' -a {}'.format(args.agent)
    if args.output is not None:
        proc_args += ' -o {}'.format(args.output)
    if args.results is not None:
        proc_args += ' -r {}'.format(args.results)
    if args.config is not None:
        proc_args += ' -c {}'.format(args.config)

    # run independent process
    os.system('python agent_runner.py {}'.format(proc_args))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multiple RL Agent runner')
    parser.add_argument('-a', '--agent', help='agent type', type=int, default=DEF_AGENT_TYPE)
    parser.add_argument('-o', '--output', help='directory in which to store results')
    parser.add_argument('-r', '--results', help='directory from which to load results')
    parser.add_argument('-c', '--config', help='path to config file')
    parser.add_argument('-t', '--trials', help='number of trials to run', type=int, default=DEF_TRIALS)
    parser.add_argument('-p', '--parallel', help='run trials in parallel', action='store_true')
    args = parser.parse_args()

    # tries to get num trials
    num_trials = args.trials

    # run trials sequentially or in parallel
    if args.parallel:

        print('Running {} trials in parallel (CPU count: {})'.format(num_trials, mp.cpu_count()))
        pool = mp.Pool(mp.cpu_count())

        # runs all trials in parallel
        for i in range(num_trials):
            pool.apply_async(run_trial_process, args=[i])

        # waits for jobs completion
        pool.close()
        pool.join()
    else:
        print('Running {} trials sequentially'.format(num_trials))
        for i in range(num_trials):
            run_trial_process(i)
