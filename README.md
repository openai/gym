# OpenAI Gym 

This is my personal fork of [OpenAI gym](https://github.com/openai/gym) that I maintain for variuos new environments, relevant for research in pixel-based control. These environments will include:
- [Carrot pushing environment](https://arxiv.org/pdf/2002.09093.pdf) for manipulating piles of objects 
- Gym environments wrapped around [Drake](https://drake.mit.edu/)
- Modifications to existing [gym](https://github.com/openai/gym) environments where observations are directly pixels.

Some of these environments will be exported to a pip package once they are mature.

# Setup 

This personal fork can be setup in the following way.

```
git clone git@github.com:hjsuh94/gym.git
cd gym
pip install -e .
``` 

If you are working on this repo then add the following lines:
```
cd gym 
git remote set-url origin git@github.com:hjsuh94/gym.git
git remote add upstream git@github.com:openai/gym.git
git remote set-url --push upstream no_push
```

# New Environments

- Carrot pushing environment. `gym.make('Carrot-v0')`

# Dependencies 
- `pyglet`
- `pymunk`
