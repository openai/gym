# Gym Contribution Guidelines

At this time we are currently accepting the current forms of contributions:

- Bug reports (keep in mind that changing environment behavior should be minimized as that requires releasing a new version of the environment and makes results hard to compare across versions)
- Pull requests for bug fixes
- Documentation improvements

Notably, we are not accepting these forms of contributions:

- New environments
- New features

This may change in the future.
If you wish to make a Gym environment, follow the instructions in [Creating Environments](https://github.com/openai/gym/blob/master/docs/creating_environments.md).  When your environment works, you can make a PR to add it to the bottom of the [List of Environments](https://github.com/openai/gym/blob/master/docs/third_party_environments.md).


Edit July 27, 2021: Please see https://github.com/openai/gym/issues/2259 for new contributing standards

# Development
This section contains technical instructions & hints for the contributors.

## Type checking
The project uses `pyright` to check types. 
To type check locally, install `pyright` per official [instructions](https://github.com/microsoft/pyright#command-line). 
It's configuration lives within `pyproject.toml`. It includes list of included and excluded files currently supporting type checks.
To run `pyright` for the project, run the pre-commit process (`pre-commit run --all-files`) or `pyright --project=pyproject.toml`
Alternatively, pyright is a built-in feature of VSCode that will automatically provide type hinting.

### Adding typing to more modules and packages
If you would like to add typing to a module in the project, 
the list of included, excluded and strict files can be found in pyproject.toml (pyproject.toml -> [tool.pyright]). 
To run `pyright` for the project, run the pre-commit process (`pre-commit run --all-files`) or `pyright`

## Git hooks
The CI will run several checks on the new code pushed to the Gym repository. These checks can also be run locally without waiting for the CI by following the steps below:
1. [install `pre-commit`](https://pre-commit.com/#install),
2. Install the Git hooks by running `pre-commit install`.

Once those two steps are done, the Git hooks will be run automatically at every new commit. 
The Git hooks can also be run manually with `pre-commit run --all-files`, and if needed they can be skipped (not recommended) with `git commit --no-verify`. 
**Note:** you may have to run `pre-commit run --all-files` manually a couple of times to make it pass when you commit, as each formatting tool will first format the code and fail the first time but should pass the second time.

Additionally, for pull requests, the project runs a number of tests for the whole project using [pytest](https://docs.pytest.org/en/latest/getting-started.html#install-pytest).
These tests can be run locally with `pytest` in the root folder. 

## Docstrings
Pydocstyle has been added to the pre-commit process such that all new functions follow the (google docstring style)[https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html].
All new functions require either a short docstring, a single line explaining the purpose of a function
or a multiline docstring that documents each argument and the return type (if there is one) of the function.
In addition, new file and class require top docstrings that should outline the purpose of the file/class.
For classes, code block examples can be provided in the top docstring and not the constructor arguments.

To check your docstrings are correct, run `pre-commit run --al-files` or `pydocstyle --source --explain --convention=google`.
If all docstrings that fail, the source and reason for the failure is provided. 