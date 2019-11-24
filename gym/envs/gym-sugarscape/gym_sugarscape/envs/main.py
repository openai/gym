from sugarscape_env import SugarscapeEnv
"""
Example scenario run of model
"""
x = SugarscapeEnv()
x.reset(10, 50) # 50 by 50 grid and 10 agents.
print(x.step('N'))
print(x.step('E'))
print(x.step('S'))
print(x.step('W'))
print(x.step('N'))
x.render()
