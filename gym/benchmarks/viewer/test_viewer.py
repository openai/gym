from gym.benchmarks.viewer.app import load_tasks_from_bmrun_path, EvaluationResource

evaluation_path = '/tmp/AtariExploration40M/atariexploration40m_1487832585_bst_h_dqn_k20_nm/bst_h_dqn_k20_nm_1487832585_freewaynoframeskip-v3_0/gym/'

bmrun_path = '/tmp/AtariExploration40M/atariexploration40m_1490254298_bst_h_dqndbl_k10_nm_mvo'

def test_trial_loading():
    trial = EvaluationResource.from_training_dir(evaluation_path)
    assert trial is not None


def test_tasks_from_bmrun_path():
    tasks = load_tasks_from_bmrun_path(bmrun_path)
    assert len(tasks.keys()) == 7


if __name__ == '__main__':
    test_tasks_from_bmrun_path()
