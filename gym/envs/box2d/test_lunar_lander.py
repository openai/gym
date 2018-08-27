from .lunar_lander import LunarLander, LunarLanderContinuous, heuristic

def test_lunar_lander():
    _test_lander(LunarLander())

def test_lunar_lander_continuous():
    _test_lander(LunarLanderContinuous())

def _test_lander(env):
    total_reward = 0
    steps = 0
    s = env.reset()
    while True:
        a = heuristic(env, s)
        s, r, done, info = env.step(a)
        still_open = env.render()
        if still_open==False: break
        total_reward += r
        if steps % 20 == 0 or done:
            print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        if done: break
    assert total_reward > 100


if __name__=="__main__":
    env = LunarLander()
    _test_lander(env)

