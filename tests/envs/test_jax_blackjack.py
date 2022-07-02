import jax.numpy as jnp
from jax import random

import gym


def test_jax_blackjack_wrapped_rollout():
    # testing basic operation of the reset and step function wrappers
    env = gym.make("Jax-BlackJack-v0")
    obs = env.reset(seed=0)

    assert obs == (19, 10, 0)
    action = 1
    result = env.step(action)
    assert result == ((29, 10, 0), -1.0, True, {})

    obs = env.reset(seed=0)
    assert obs == (19, 10, 0)
    action = 0
    result = env.step(action)
    assert result == ((19, 10, 0), 1.0, True, {})


def test_jax_blackjack_sutton_barto():

    key = random.PRNGKey(0)

    player_hand = jnp.zeros(21)
    dealer_hand = jnp.zeros(21)
    player_hand = player_hand.at[0].set(1)
    player_hand = player_hand.at[1].set(10)
    dealer_hand = dealer_hand.at[0].set(1)
    dealer_hand = dealer_hand.at[1].set(10)
    dealer_cards = 2
    player_cards = 2
    double_natural_state = (dealer_hand, player_hand, dealer_cards, player_cards), key

    env = gym.make("Jax-BlackJack-v0")
    # for Jax-BlackJack-v0, sutton_and_barto flag is active, and natural flag is inactive (default env.make)
    _ = env.reset()

    env.unwrapped.state = double_natural_state
    action = 0
    obs, reward, done, info = env.step(action)
    assert done
    assert reward == 0

    env = gym.make("Jax-BlackJack-v0", sutton_and_barto=True, natural=False)
    #  sutton_and_barto flag is active, and natural flag is inactive
    _ = env.reset()

    env.unwrapped.state = double_natural_state
    action = 0
    obs, reward, done, info = env.step(action)
    assert done
    assert reward == 0

    env = gym.make("Jax-BlackJack-v0", sutton_and_barto=True, natural=True)
    # sutton_and_barto flag is active, and natural flag is active
    _ = env.reset()

    env.unwrapped.state = double_natural_state
    action = 0
    obs, reward, done, info = env.step(action)
    assert done
    assert reward == 0

    key = random.PRNGKey(
        6
    )  # we pick a seed and starting state that results in the dealer getting a non-natural 21 count (7,9,5)

    player_hand = jnp.zeros(21)
    dealer_hand = jnp.zeros(21)
    player_hand = player_hand.at[0].set(1)
    player_hand = player_hand.at[1].set(10)
    dealer_hand = dealer_hand.at[0].set(7)
    dealer_hand = dealer_hand.at[1].set(9)
    dealer_cards = 2
    player_cards = 2
    player_natural_dealer_not_natural = (
        dealer_hand,
        player_hand,
        dealer_cards,
        player_cards,
    ), key

    env = gym.make("Jax-BlackJack-v0")
    # for Jax-BlackJack-v0, sutton_and_barto flag is active, and natural flag is inactive (default make)
    _ = env.reset()

    env.unwrapped.state = player_natural_dealer_not_natural
    action = 0
    obs, reward, done, info = env.step(action)
    print(env.state)
    assert done
    assert reward == 1

    env = gym.make("Jax-BlackJack-v0", sutton_and_barto=True, natural=False)
    # sutton_and_barto flag is active, and natural flag is inactive
    _ = env.reset()

    env.unwrapped.state = player_natural_dealer_not_natural
    action = 0
    obs, reward, done, info = env.step(action)
    print(env.state)
    assert done
    assert reward == 1

    env = gym.make(
        "Jax-BlackJack-v0", sutton_and_barto=True, natural=True
    )  # sutton_and_barto flag is active, and natural flag is active
    _ = env.reset()

    env.unwrapped.state = player_natural_dealer_not_natural
    action = 0
    obs, reward, done, info = env.step(action)
    print(env.state)
    assert done
    assert reward == 1

    key = random.PRNGKey(6)
    # we pick a seed and starting state that results in the player getting a non-natural 21 count (7,9,5)

    player_hand = jnp.zeros(21)
    dealer_hand = jnp.zeros(21)
    player_hand = player_hand.at[0].set(7)
    player_hand = player_hand.at[1].set(9)
    dealer_hand = dealer_hand.at[0].set(1)
    dealer_hand = dealer_hand.at[1].set(10)
    dealer_cards = 2
    player_cards = 2
    player_natural_dealer_not_natural = (
        dealer_hand,
        player_hand,
        dealer_cards,
        player_cards,
    ), key

    env = gym.make("Jax-BlackJack-v0")
    # for Jax-BlackJack-v0, sutton_and_barto flag is active, and natural flag is inactive (default make)
    _ = env.reset()

    env.unwrapped.state = player_natural_dealer_not_natural
    action = 1
    obs, reward, done, info = env.step(action)
    action = 0
    obs, reward, done, info = env.step(action)
    print(env.state)
    assert done
    assert reward == 0

    env = gym.make("Jax-BlackJack-v0", sutton_and_barto=True, natural=False)
    # sutton_and_barto flag is active, and natural flag is inactive
    _ = env.reset()

    env.unwrapped.state = player_natural_dealer_not_natural
    action = 1
    obs, reward, done, info = env.step(action)
    action = 0
    obs, reward, done, info = env.step(action)

    print(env.state)
    assert done
    assert reward == 0

    env = gym.make("Jax-BlackJack-v0", sutton_and_barto=True, natural=True)
    # sutton_and_barto flag is active, and natural flag is active
    _ = env.reset()

    env.unwrapped.state = player_natural_dealer_not_natural
    action = 1
    obs, reward, done, info = env.step(action)
    action = 0
    obs, reward, done, info = env.step(action)

    print(env.state)
    assert done
    assert reward == 0


def test_jax_blackjack_natural():
    key = random.PRNGKey(6)
    # we pick a seed and starting state that results in the dealer getting a non-natural 21 count (7,9,5)

    player_hand = jnp.zeros(21)
    dealer_hand = jnp.zeros(21)
    player_hand = player_hand.at[0].set(1)
    player_hand = player_hand.at[1].set(10)
    dealer_hand = dealer_hand.at[0].set(1)
    dealer_hand = dealer_hand.at[1].set(10)
    dealer_cards = 2
    player_cards = 2
    player_natural_dealer_not_natural = (
        dealer_hand,
        player_hand,
        dealer_cards,
        player_cards,
    ), key

    env = gym.make("Jax-BlackJack-v0", sutton_and_barto=False, natural=True)
    # = sutton_and_barto flag is inactive, and natural flag is active
    _ = env.reset()

    env.unwrapped.state = player_natural_dealer_not_natural
    action = 0
    obs, reward, done, info = env.step(action)
    print(env.state)
    assert done
    assert reward == 0

    key = random.PRNGKey(6)
    # we pick a seed and starting state that results in the player getting a sub-21 count (6,6,5)

    player_hand = jnp.zeros(21)
    dealer_hand = jnp.zeros(21)
    player_hand = player_hand.at[0].set(1)
    player_hand = player_hand.at[1].set(10)
    dealer_hand = dealer_hand.at[0].set(6)
    dealer_hand = dealer_hand.at[1].set(6)
    dealer_cards = 2
    player_cards = 2
    player_natural_dealer_sub_21 = (
        dealer_hand,
        player_hand,
        dealer_cards,
        player_cards,
    ), key

    env = gym.make("Jax-BlackJack-v0", sutton_and_barto=False, natural=True)
    _ = env.reset()

    env.unwrapped.state = player_natural_dealer_sub_21
    action = 0
    obs, reward, done, info = env.step(action)
    print(env.state)
    assert done
    assert reward == 1.5
