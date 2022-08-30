import numpy as np
import pytest

from gym.utils.env_checker import data_equivalence
from gym.utils.step_api_compatibility import (
    convert_to_done_step_api,
    convert_to_terminated_truncated_step_api,
)


@pytest.mark.parametrize(
    "is_vector_env, done_returns, expected_terminated, expected_truncated",
    (
        # Test each of the permutations for single environments with and without the old info
        (False, (0, 0, False, {"Test-info": True}), False, False),
        (False, (0, 0, False, {"TimeLimit.truncated": False}), False, False),
        (False, (0, 0, True, {}), True, False),
        (False, (0, 0, True, {"TimeLimit.truncated": True}), False, True),
        (False, (0, 0, True, {"Test-info": True}), True, False),
        # Test vectorise versions with both list and dict infos testing each permutation for sub-environments
        (
            True,
            (
                0,
                0,
                np.array([False, True, True]),
                [{}, {}, {"TimeLimit.truncated": True}],
            ),
            np.array([False, True, False]),
            np.array([False, False, True]),
        ),
        (
            True,
            (
                0,
                0,
                np.array([False, True, True]),
                {"TimeLimit.truncated": np.array([False, False, True])},
            ),
            np.array([False, True, False]),
            np.array([False, False, True]),
        ),
        # empty truncated info
        (
            True,
            (
                0,
                0,
                np.array([False, True]),
                {},
            ),
            np.array([False, True]),
            np.array([False, False]),
        ),
    ),
)
def test_to_done_step_api(
    is_vector_env, done_returns, expected_terminated, expected_truncated
):
    _, _, terminated, truncated, info = convert_to_terminated_truncated_step_api(
        done_returns, is_vector_env=is_vector_env
    )
    assert np.all(terminated == expected_terminated)
    assert np.all(truncated == expected_truncated)

    if is_vector_env is False:
        assert "TimeLimit.truncated" not in info
    elif isinstance(info, list):
        assert all("TimeLimit.truncated" not in sub_info for sub_info in info)
    else:  # isinstance(info, dict)
        assert "TimeLimit.truncated" not in info

    roundtripped_returns = convert_to_done_step_api(
        (0, 0, terminated, truncated, info), is_vector_env=is_vector_env
    )
    assert data_equivalence(done_returns, roundtripped_returns)


@pytest.mark.parametrize(
    "is_vector_env, terminated_truncated_returns, expected_done, expected_truncated",
    (
        (False, (0, 0, False, False, {"Test-info": True}), False, False),
        (False, (0, 0, True, False, {}), True, False),
        (False, (0, 0, False, True, {}), True, True),
        # (False, (), True, True),  # Not possible to encode in the old step api
        # Test vector dict info
        (
            True,
            (0, 0, np.array([False, True, False]), np.array([False, False, True]), {}),
            np.array([False, True, True]),
            np.array([False, False, True]),
        ),
        # Test vector dict info with no truncation
        (
            True,
            (0, 0, np.array([False, True]), np.array([False, False]), {}),
            np.array([False, True]),
            np.array([False, False]),
        ),
        # Test vector list info
        (
            True,
            (
                0,
                0,
                np.array([False, True, False]),
                np.array([False, False, True]),
                [{"Test-Info": True}, {}, {}],
            ),
            np.array([False, True, True]),
            np.array([False, False, True]),
        ),
    ),
)
def test_to_terminated_truncated_step_api(
    is_vector_env, terminated_truncated_returns, expected_done, expected_truncated
):
    _, _, done, info = convert_to_done_step_api(
        terminated_truncated_returns, is_vector_env=is_vector_env
    )
    assert np.all(done == expected_done)

    if is_vector_env is False:
        if expected_done:
            assert info["TimeLimit.truncated"] == expected_truncated
        else:
            assert "TimeLimit.truncated" not in info
    elif isinstance(info, list):
        for sub_info, env_done, env_truncated in zip(
            info, expected_done, expected_truncated
        ):
            if env_done:
                assert sub_info["TimeLimit.truncated"] == env_truncated
            else:
                assert "TimeLimit.truncated" not in sub_info
    else:  # isinstance(info, dict)
        if np.any(expected_done):
            assert np.all(info["TimeLimit.truncated"] == expected_truncated)
        else:
            assert "TimeLimit.truncated" not in info

    roundtripped_returns = convert_to_terminated_truncated_step_api(
        (0, 0, done, info), is_vector_env=is_vector_env
    )
    assert data_equivalence(terminated_truncated_returns, roundtripped_returns)


def test_edge_case():
    # When converting between the two-step APIs this is not possible in a single case
    #   terminated=True and truncated=True -> done=True and info={}
    # We cannot test this in test_to_terminated_truncated_step_api as the roundtripping test will fail
    _, _, done, info = convert_to_done_step_api((0, 0, True, True, {}))
    assert done is True
    assert info == {"TimeLimit.truncated": False}

    # Test with vector dict info
    _, _, done, info = convert_to_done_step_api(
        (0, 0, np.array([True]), np.array([True]), {}), is_vector_env=True
    )
    assert np.all(done)
    assert info == {"TimeLimit.truncated": np.array([False])}

    # Test with vector list info
    _, _, done, info = convert_to_done_step_api(
        (0, 0, np.array([True]), np.array([True]), [{"Test-Info": True}]),
        is_vector_env=True,
    )
    assert np.all(done)
    assert info == [{"Test-Info": True, "TimeLimit.truncated": False}]
