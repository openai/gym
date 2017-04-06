from gym.benchmarks.viewer.app import Trial

trial = Trial.from_training_dir('/tmp/AtariExploration40M/atariexploration40m_1487832585_bst_h_dqn_k20_nm/bst_h_dqn_k20_nm_1487832585_freewaynoframeskip-v3_0/gym/')

print(trial.score())
import ipdb; ipdb.set_trace()
