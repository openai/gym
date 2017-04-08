# Benchmark Viewer

View a benchmark
```
cd gym/benchmarks/viewer
./view_benchmark.py /tmp/Atari40M --open
```

Tests live in scoreboard directory, run them with
```
pytest ../../scoreboard/
```

In development, run with --debug to get a debugger
```
./view_benchmark.py /tmp/Atari40M --debug
```

# Data

The important information about the performance of each evaluation lives in the batch stats e.g. `openaigym.episode_batch.0.7.stats.json`

benchmark-specific data lives in `benchmark_run_data.yaml`
```
 ॐ  cat /tmp/Atari40M/atari40m_1484937002_deepq_double_prior/benchmark_run_data.yaml
id: deepq_double_prior
title: Double DQN with prioritized replay
github_user: nivwusquorum
repository: https://github.com/openai/rl-algs
commit: 1eae50479a76e1513edef9343683737fbfd103f1
commmand: python run atari_benchmark.py ...
```


# Integration tests

We want to maintain scoring parity with the previous benchmark. You can look at
```
/mnt/efs/richardchen/tinkerbell/atariexploration40m_1490254298_bst_h_dqndbl_k10_nm_mvo
```
which was uploaded to
http://gym.sci.openai-tech.com/benchmark_runs/bmrun_1mercg57Q3mulsPSVYJDw
