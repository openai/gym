# Development
This file contains technical instructions & hints for the contributors.

## Type checking
The project uses `pyright` to check types. 
To type check locally, install pyright per official [instructions](https://github.com/microsoft/pyright#command-line). 
It's configuration lives under a section of `pyproject.toml`. It includes list of files currently supporting type checks. Use `pyright` CLI command to launch type checks.

### Adding typing to more modules and packages

If you would like to add typing to a module in the project, 
add the path to the file(s) in the include section 
(pyproject.toml -> [tool.pyright] -> include). 
Then you can run `pyright` to see list of type problems in the newly added file, and fix them.
