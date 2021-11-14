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
To type check locally, install pyright per official [instructions](https://github.com/microsoft/pyright#command-line). 
It's configuration lives under a section of `pyproject.toml`. It includes list of files currently supporting type checks. Use `pyright` CLI command to launch type checks.

### Adding typing to more modules and packages

If you would like to add typing to a module in the project, 
add the path to the file(s) in the include section 
(pyproject.toml -> [tool.pyright] -> include). 
Then you can run `pyright` to see list of type problems in the newly added file, and fix them.

