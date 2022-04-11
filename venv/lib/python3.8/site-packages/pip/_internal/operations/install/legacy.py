"""Legacy installation process, i.e. `setup.py install`.
"""

import logging
import os
from distutils.util import change_root

from pip._internal.utils.deprecation import deprecated
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import ensure_dir
from pip._internal.utils.setuptools_build import make_setuptools_install_args
from pip._internal.utils.subprocess import runner_with_spinner_message
from pip._internal.utils.temp_dir import TempDirectory
from pip._internal.utils.typing import MYPY_CHECK_RUNNING

if MYPY_CHECK_RUNNING:
    from typing import List, Optional, Sequence

    from pip._internal.models.scheme import Scheme
    from pip._internal.req.req_install import InstallRequirement


logger = logging.getLogger(__name__)


def install(
    install_req,  # type: InstallRequirement
    install_options,  # type: List[str]
    global_options,  # type: Sequence[str]
    root,  # type: Optional[str]
    home,  # type: Optional[str]
    prefix,  # type: Optional[str]
    use_user_site,  # type: bool
    pycompile,  # type: bool
    scheme,  # type: Scheme
):
    # type: (...) -> None
    # Extend the list of global and install options passed on to
    # the setup.py call with the ones from the requirements file.
    # Options specified in requirements file override those
    # specified on the command line, since the last option given
    # to setup.py is the one that is used.
    global_options = list(global_options) + \
        install_req.options.get('global_options', [])
    install_options = list(install_options) + \
        install_req.options.get('install_options', [])

    header_dir = scheme.headers

    with TempDirectory(kind="record") as temp_dir:
        record_filename = os.path.join(temp_dir.path, 'install-record.txt')
        install_args = make_setuptools_install_args(
            install_req.setup_py_path,
            global_options=global_options,
            install_options=install_options,
            record_filename=record_filename,
            root=root,
            prefix=prefix,
            header_dir=header_dir,
            home=home,
            use_user_site=use_user_site,
            no_user_config=install_req.isolated,
            pycompile=pycompile,
        )

        runner = runner_with_spinner_message(
            "Running setup.py install for {}".format(install_req.name)
        )
        with indent_log(), install_req.build_env:
            runner(
                cmd=install_args,
                cwd=install_req.unpacked_source_directory,
            )

        if not os.path.exists(record_filename):
            logger.debug('Record file %s not found', record_filename)
            return
        install_req.install_succeeded = True

        # We intentionally do not use any encoding to read the file because
        # setuptools writes the file using distutils.file_util.write_file,
        # which does not specify an encoding.
        with open(record_filename) as f:
            record_lines = f.read().splitlines()

    def prepend_root(path):
        # type: (str) -> str
        if root is None or not os.path.isabs(path):
            return path
        else:
            return change_root(root, path)

    for line in record_lines:
        directory = os.path.dirname(line)
        if directory.endswith('.egg-info'):
            egg_info_dir = prepend_root(directory)
            break
    else:
        deprecated(
            reason=(
                "{} did not indicate that it installed an "
                ".egg-info directory. Only setup.py projects "
                "generating .egg-info directories are supported."
            ).format(install_req),
            replacement=(
                "for maintainers: updating the setup.py of {0}. "
                "For users: contact the maintainers of {0} to let "
                "them know to update their setup.py.".format(
                    install_req.name
                )
            ),
            gone_in="20.2",
            issue=6998,
        )
        # FIXME: put the record somewhere
        return
    new_lines = []
    for line in record_lines:
        filename = line.strip()
        if os.path.isdir(filename):
            filename += os.path.sep
        new_lines.append(
            os.path.relpath(prepend_root(filename), egg_info_dir)
        )
    new_lines.sort()
    ensure_dir(egg_info_dir)
    inst_files_path = os.path.join(egg_info_dir, 'installed-files.txt')
    with open(inst_files_path, 'w') as f:
        f.write('\n'.join(new_lines) + '\n')
