from gym.benchmarks.viewer.app import tasks_from_bmrun_path, Evaluation

evaluation_path = '/tmp/AtariExploration40M/atariexploration40m_1487832585_bst_h_dqn_k20_nm/bst_h_dqn_k20_nm_1487832585_freewaynoframeskip-v3_0/gym/'

bmrun_path = '/tmp/AtariExploration40M/atariexploration40m_1487832585_bst_h_dqn_k20_nm/'

def test_trial_loading():
    trial = Evaluation.from_training_dir(evaluation_path)

    score_results = trial.score()
    assert score_results == 22.

def test_tasks_from_bmrun_path():
    env_id_to_task = tasks_from_bmrun_path(bmrun_path)

