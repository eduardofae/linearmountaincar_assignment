N_REPETITIONS = 10
N_STEPS = 50_000
N_SUCCESSES = 10
N_AGENTS = 5

import gymnasium as gym
env = gym.make('MountainCar-v0')

from feature_extractors import RBFFeatureExtractor
fes = [
    RBFFeatureExtractor(env), # N_CENTERS 25
    RBFFeatureExtractor(env, n_centers=10),
    RBFFeatureExtractor(env, n_centers=15),
    RBFFeatureExtractor(env, n_centers=26),
    RBFFeatureExtractor(env, n_centers=37),
    RBFFeatureExtractor(env, n_centers=48)
]

from sarsa_linear_agent import LinearSarsaAgent
agents = {
    'LR=0.004'  : LinearSarsaAgent(env, fes[0], learning_rate=0.004),
    'LR=0.02'   : LinearSarsaAgent(env, fes[0], learning_rate=0.02),
    'LR=0.03'   : LinearSarsaAgent(env, fes[0], learning_rate=0.03),
    'LR=0.06'   : LinearSarsaAgent(env, fes[0], learning_rate=0.06),
    'LR=0.2'    : LinearSarsaAgent(env, fes[0], learning_rate=0.2),
    'CENTERS=10': LinearSarsaAgent(env, fes[1]), # LR 0.01
    'CENTERS=15': LinearSarsaAgent(env, fes[2]),
    'CENTERS=26': LinearSarsaAgent(env, fes[3]),
    'CENTERS=37': LinearSarsaAgent(env, fes[4]),
    'CENTERS=48': LinearSarsaAgent(env, fes[5])
}

def get_results_dict_layout():
    return {
        feature: []
        for feature in agents.keys()
    }

from tqdm import tqdm
def gen_results():
    results = get_results_dict_layout()
    for seed in tqdm(range(N_REPETITIONS), desc='Repetitions'):
        env.action_space.seed(seed)
        for key, agent in tqdm(agents.items(), desc='Agents'):
            _, _, steps_needed = agent.train_until_x_successes(N_STEPS, N_SUCCESSES, seed)
            results[key].append(steps_needed)
    return results

import json
from pathlib import Path
def save_results(results, path='../results/results.json'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as file:
        file.write(json.dumps(results))

import numpy as np
results = gen_results()
sorted_results = sorted(results.items(), key=lambda item: np.median(item[1]))
best_results = dict(sorted_results[:N_AGENTS])
save_results(best_results)