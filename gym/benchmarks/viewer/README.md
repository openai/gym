# Benchmark Viewer

View a benchmark
```
cd gym/benchmarks/viewer
./view_benchmark.py /tmp/Atari40M --open
```

Run tests with
```
pytest
```

In development, run with --debug to get a debugger
```
./view_benchmark.py /tmp/Atari40M --debug
```

# Integration tests

We want to maintain scoring parity with the previous benchmark. You can look at
```
/mnt/efs/richardchen/tinkerbell/atariexploration40m_1490254298_bst_h_dqndbl_k10_nm_mvo
```
which was uploaded to
http://gym.sci.openai-tech.com/benchmark_runs/bmrun_1mercg57Q3mulsPSVYJDw
