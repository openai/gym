import pytest

pytest.importorskip("gym.envs.atari")

from gym.envs.registration import registry

from itertools import product


def test_ale_legacy_env_specs():
    versions = ["-v0", "-v4"]
    suffixes = ["", "NoFrameskip", "Deterministic"]
    obs_types = ["", "-ram"]
    games = [
        "adventure",
        "air_raid",
        "alien",
        "amidar",
        "assault",
        "asterix",
        "asteroids",
        "atlantis",
        "bank_heist",
        "battle_zone",
        "beam_rider",
        "berzerk",
        "bowling",
        "boxing",
        "breakout",
        "carnival",
        "centipede",
        "chopper_command",
        "crazy_climber",
        "defender",
        "demon_attack",
        "double_dunk",
        "elevator_action",
        "enduro",
        "fishing_derby",
        "freeway",
        "frostbite",
        "gopher",
        "gravitar",
        "hero",
        "ice_hockey",
        "jamesbond",
        "journey_escape",
        "kangaroo",
        "krull",
        "kung_fu_master",
        "montezuma_revenge",
        "ms_pacman",
        "name_this_game",
        "phoenix",
        "pitfall",
        "pong",
        "pooyan",
        "private_eye",
        "qbert",
        "riverraid",
        "road_runner",
        "robotank",
        "seaquest",
        "skiing",
        "solaris",
        "space_invaders",
        "star_gunner",
        "tennis",
        "time_pilot",
        "tutankham",
        "up_n_down",
        "venture",
        "video_pinball",
        "wizard_of_wor",
        "yars_revenge",
        "zaxxon",
    ]

    # Convert snake case to camel case
    games = list(map(lambda x: x.title().replace("_", ""), games))
    specs = list(map("".join, product(games, obs_types, suffixes, versions)))

    """
    defaults:
        repeat_action_probability = 0.0
        full_action_space = False
        frameskip = (2, 5)
        game = "Pong"
        obs_type = "ram"
        mode = None
        difficulty = None

    v0: repeat_action_probability = 0.25
    v4: inherits defaults

    -NoFrameskip: frameskip = 1
    -Deterministic: frameskip = 4 or 3 for space_invaders
    """
    for spec in specs:
        assert spec in registry.env_specs
        kwargs = registry.env_specs[spec]._kwargs

        # Assert necessary parameters are set
        assert "frameskip" in kwargs
        assert "game" in kwargs
        assert "obs_type" in kwargs
        assert "repeat_action_probability" in kwargs
        assert "full_action_space" in kwargs

        # Common defaults
        assert kwargs["full_action_space"] is False
        assert "mode" not in kwargs
        assert "difficulty" not in kwargs

        if "-ram" in spec:
            assert kwargs["obs_type"] == "ram"
        else:
            assert kwargs["obs_type"] == "rgb"

        if "NoFrameskip" in spec:
            assert kwargs["frameskip"] == 1
        elif "Deterministic" in spec:
            assert isinstance(kwargs["frameskip"], int)
            frameskip = 3 if "SpaceInvaders" in spec else 4
            assert kwargs["frameskip"] == frameskip
        else:
            assert isinstance(kwargs["frameskip"], tuple) and kwargs["frameskip"] == (
                2,
                5,
            )

        assert spec.endswith("v0") or spec.endswith("v4")
        if spec.endswith("v0"):
            assert kwargs["repeat_action_probability"] == 0.25
        elif spec.endswith("v4"):
            assert kwargs["repeat_action_probability"] == 0.0
