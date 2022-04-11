"""
Requirements file parsing
"""

# The following comment should be removed at some point in the future.
# mypy: strict-optional=False

from __future__ import absolute_import

import optparse
import os
import re
import shlex
import sys

from pip._vendor.six.moves import filterfalse
from pip._vendor.six.moves.urllib import parse as urllib_parse

from pip._internal.cli import cmdoptions
from pip._internal.exceptions import (
    InstallationError,
    RequirementsFileParseError,
)
from pip._internal.models.search_scope import SearchScope
from pip._internal.req.constructors import (
    install_req_from_editable,
    install_req_from_line,
)
from pip._internal.utils.encoding import auto_decode
from pip._internal.utils.typing import MYPY_CHECK_RUNNING
from pip._internal.utils.urls import get_url_scheme

if MYPY_CHECK_RUNNING:
    from optparse import Values
    from typing import (
        Any, Callable, Iterator, List, NoReturn, Optional, Text, Tuple,
    )

    from pip._internal.req import InstallRequirement
    from pip._internal.cache import WheelCache
    from pip._internal.index.package_finder import PackageFinder
    from pip._internal.network.session import PipSession

    ReqFileLines = Iterator[Tuple[int, Text]]

    LineParser = Callable[[Text], Tuple[str, Values]]


__all__ = ['parse_requirements']

SCHEME_RE = re.compile(r'^(http|https|file):', re.I)
COMMENT_RE = re.compile(r'(^|\s+)#.*$')

# Matches environment variable-style values in '${MY_VARIABLE_1}' with the
# variable name consisting of only uppercase letters, digits or the '_'
# (underscore). This follows the POSIX standard defined in IEEE Std 1003.1,
# 2013 Edition.
ENV_VAR_RE = re.compile(r'(?P<var>\$\{(?P<name>[A-Z0-9_]+)\})')

SUPPORTED_OPTIONS = [
    cmdoptions.index_url,
    cmdoptions.extra_index_url,
    cmdoptions.no_index,
    cmdoptions.constraints,
    cmdoptions.requirements,
    cmdoptions.editable,
    cmdoptions.find_links,
    cmdoptions.no_binary,
    cmdoptions.only_binary,
    cmdoptions.require_hashes,
    cmdoptions.pre,
    cmdoptions.trusted_host,
    cmdoptions.always_unzip,  # Deprecated
]  # type: List[Callable[..., optparse.Option]]

# options to be passed to requirements
SUPPORTED_OPTIONS_REQ = [
    cmdoptions.install_options,
    cmdoptions.global_options,
    cmdoptions.hash,
]  # type: List[Callable[..., optparse.Option]]

# the 'dest' string values
SUPPORTED_OPTIONS_REQ_DEST = [str(o().dest) for o in SUPPORTED_OPTIONS_REQ]


class ParsedLine(object):
    def __init__(
        self,
        filename,  # type: str
        lineno,  # type: int
        comes_from,  # type: str
        args,  # type: str
        opts,  # type: Values
        constraint,  # type: bool
    ):
        # type: (...) -> None
        self.filename = filename
        self.lineno = lineno
        self.comes_from = comes_from
        self.args = args
        self.opts = opts
        self.constraint = constraint


def parse_requirements(
    filename,  # type: str
    session,  # type: PipSession
    finder=None,  # type: Optional[PackageFinder]
    comes_from=None,  # type: Optional[str]
    options=None,  # type: Optional[optparse.Values]
    constraint=False,  # type: bool
    wheel_cache=None,  # type: Optional[WheelCache]
    use_pep517=None  # type: Optional[bool]
):
    # type: (...) -> Iterator[InstallRequirement]
    """Parse a requirements file and yield InstallRequirement instances.

    :param filename:    Path or url of requirements file.
    :param session:     PipSession instance.
    :param finder:      Instance of pip.index.PackageFinder.
    :param comes_from:  Origin description of requirements.
    :param options:     cli options.
    :param constraint:  If true, parsing a constraint file rather than
        requirements file.
    :param wheel_cache: Instance of pip.wheel.WheelCache
    :param use_pep517:  Value of the --use-pep517 option.
    """
    skip_requirements_regex = (
        options.skip_requirements_regex if options else None
    )
    line_parser = get_line_parser(finder)
    parser = RequirementsFileParser(
        session, line_parser, comes_from, skip_requirements_regex
    )

    for parsed_line in parser.parse(filename, constraint):
        req = handle_line(
            parsed_line, finder, options, session, wheel_cache, use_pep517
        )
        if req is not None:
            yield req


def preprocess(content, skip_requirements_regex):
    # type: (Text, Optional[str]) -> ReqFileLines
    """Split, filter, and join lines, and return a line iterator

    :param content: the content of the requirements file
    :param options: cli options
    """
    lines_enum = enumerate(content.splitlines(), start=1)  # type: ReqFileLines
    lines_enum = join_lines(lines_enum)
    lines_enum = ignore_comments(lines_enum)
    if skip_requirements_regex:
        lines_enum = skip_regex(lines_enum, skip_requirements_regex)
    lines_enum = expand_env_variables(lines_enum)
    return lines_enum


def handle_line(
    line,  # type: ParsedLine
    finder=None,  # type: Optional[PackageFinder]
    options=None,  # type: Optional[optparse.Values]
    session=None,  # type: Optional[PipSession]
    wheel_cache=None,  # type: Optional[WheelCache]
    use_pep517=None,  # type: Optional[bool]
):
    # type: (...) -> Optional[InstallRequirement]
    """Handle a single parsed requirements line; This can result in
    creating/yielding requirements, or updating the finder.

    For lines that contain requirements, the only options that have an effect
    are from SUPPORTED_OPTIONS_REQ, and they are scoped to the
    requirement. Other options from SUPPORTED_OPTIONS may be present, but are
    ignored.

    For lines that do not contain requirements, the only options that have an
    effect are from SUPPORTED_OPTIONS. Options from SUPPORTED_OPTIONS_REQ may
    be present, but are ignored. These lines may contain multiple options
    (although our docs imply only one is supported), and all our parsed and
    affect the finder.
    """

    # preserve for the nested code path
    line_comes_from = '%s %s (line %s)' % (
        '-c' if line.constraint else '-r', line.filename, line.lineno,
    )

    # return a line requirement
    if line.args:
        isolated = options.isolated_mode if options else False
        if options:
            cmdoptions.check_install_build_global(options, line.opts)
        # get the options that apply to requirements
        req_options = {}
        for dest in SUPPORTED_OPTIONS_REQ_DEST:
            if dest in line.opts.__dict__ and line.opts.__dict__[dest]:
                req_options[dest] = line.opts.__dict__[dest]
        line_source = 'line {} of {}'.format(line.lineno, line.filename)
        return install_req_from_line(
            line.args,
            comes_from=line_comes_from,
            use_pep517=use_pep517,
            isolated=isolated,
            options=req_options,
            wheel_cache=wheel_cache,
            constraint=line.constraint,
            line_source=line_source,
        )

    # return an editable requirement
    elif line.opts.editables:
        isolated = options.isolated_mode if options else False
        return install_req_from_editable(
            line.opts.editables[0], comes_from=line_comes_from,
            use_pep517=use_pep517,
            constraint=line.constraint, isolated=isolated,
            wheel_cache=wheel_cache
        )

    # percolate hash-checking option upward
    elif line.opts.require_hashes:
        options.require_hashes = line.opts.require_hashes

    # set finder options
    elif finder:
        find_links = finder.find_links
        index_urls = finder.index_urls
        if line.opts.index_url:
            index_urls = [line.opts.index_url]
        if line.opts.no_index is True:
            index_urls = []
        if line.opts.extra_index_urls:
            index_urls.extend(line.opts.extra_index_urls)
        if line.opts.find_links:
            # FIXME: it would be nice to keep track of the source
            # of the find_links: support a find-links local path
            # relative to a requirements file.
            value = line.opts.find_links[0]
            req_dir = os.path.dirname(os.path.abspath(line.filename))
            relative_to_reqs_file = os.path.join(req_dir, value)
            if os.path.exists(relative_to_reqs_file):
                value = relative_to_reqs_file
            find_links.append(value)

        search_scope = SearchScope(
            find_links=find_links,
            index_urls=index_urls,
        )
        finder.search_scope = search_scope

        if line.opts.pre:
            finder.set_allow_all_prereleases()

        if session:
            for host in line.opts.trusted_hosts or []:
                source = 'line {} of {}'.format(line.lineno, line.filename)
                session.add_trusted_host(host, source=source)

    return None


class RequirementsFileParser(object):
    def __init__(
        self,
        session,  # type: PipSession
        line_parser,  # type: LineParser
        comes_from,  # type: str
        skip_requirements_regex,  # type: Optional[str]
    ):
        # type: (...) -> None
        self._session = session
        self._line_parser = line_parser
        self._comes_from = comes_from
        self._skip_requirements_regex = skip_requirements_regex

    def parse(self, filename, constraint):
        # type: (str, bool) -> Iterator[ParsedLine]
        """Parse a given file, yielding parsed lines.
        """
        for line in self._parse_and_recurse(filename, constraint):
            yield line

    def _parse_and_recurse(self, filename, constraint):
        # type: (str, bool) -> Iterator[ParsedLine]
        for line in self._parse_file(filename, constraint):
            if (
                not line.args and
                not line.opts.editables and
                (line.opts.requirements or line.opts.constraints)
            ):
                # parse a nested requirements file
                if line.opts.requirements:
                    req_path = line.opts.requirements[0]
                    nested_constraint = False
                else:
                    req_path = line.opts.constraints[0]
                    nested_constraint = True

                # original file is over http
                if SCHEME_RE.search(filename):
                    # do a url join so relative paths work
                    req_path = urllib_parse.urljoin(filename, req_path)
                # original file and nested file are paths
                elif not SCHEME_RE.search(req_path):
                    # do a join so relative paths work
                    req_path = os.path.join(
                        os.path.dirname(filename), req_path,
                    )

                for inner_line in self._parse_and_recurse(
                    req_path, nested_constraint,
                ):
                    yield inner_line
            else:
                yield line

    def _parse_file(self, filename, constraint):
        # type: (str, bool) -> Iterator[ParsedLine]
        _, content = get_file_content(
            filename, self._session, comes_from=self._comes_from
        )

        lines_enum = preprocess(content, self._skip_requirements_regex)

        for line_number, line in lines_enum:
            try:
                args_str, opts = self._line_parser(line)
            except OptionParsingError as e:
                # add offending line
                msg = 'Invalid requirement: %s\n%s' % (line, e.msg)
                raise RequirementsFileParseError(msg)

            yield ParsedLine(
                filename,
                line_number,
                self._comes_from,
                args_str,
                opts,
                constraint,
            )


def get_line_parser(finder):
    # type: (Optional[PackageFinder]) -> LineParser
    def parse_line(line):
        # type: (Text) -> Tuple[str, Values]
        # Build new parser for each line since it accumulates appendable
        # options.
        parser = build_parser()
        defaults = parser.get_default_values()
        defaults.index_url = None
        if finder:
            defaults.format_control = finder.format_control

        args_str, options_str = break_args_options(line)
        # Prior to 2.7.3, shlex cannot deal with unicode entries
        if sys.version_info < (2, 7, 3):
            # https://github.com/python/mypy/issues/1174
            options_str = options_str.encode('utf8')  # type: ignore

        # https://github.com/python/mypy/issues/1174
        opts, _ = parser.parse_args(
            shlex.split(options_str), defaults)  # type: ignore

        return args_str, opts

    return parse_line


def break_args_options(line):
    # type: (Text) -> Tuple[str, Text]
    """Break up the line into an args and options string.  We only want to shlex
    (and then optparse) the options, not the args.  args can contain markers
    which are corrupted by shlex.
    """
    tokens = line.split(' ')
    args = []
    options = tokens[:]
    for token in tokens:
        if token.startswith('-') or token.startswith('--'):
            break
        else:
            args.append(token)
            options.pop(0)
    return ' '.join(args), ' '.join(options)  # type: ignore


class OptionParsingError(Exception):
    def __init__(self, msg):
        # type: (str) -> None
        self.msg = msg


def build_parser():
    # type: () -> optparse.OptionParser
    """
    Return a parser for parsing requirement lines
    """
    parser = optparse.OptionParser(add_help_option=False)

    option_factories = SUPPORTED_OPTIONS + SUPPORTED_OPTIONS_REQ
    for option_factory in option_factories:
        option = option_factory()
        parser.add_option(option)

    # By default optparse sys.exits on parsing errors. We want to wrap
    # that in our own exception.
    def parser_exit(self, msg):
        # type: (Any, str) -> NoReturn
        raise OptionParsingError(msg)
    # NOTE: mypy disallows assigning to a method
    #       https://github.com/python/mypy/issues/2427
    parser.exit = parser_exit  # type: ignore

    return parser


def join_lines(lines_enum):
    # type: (ReqFileLines) -> ReqFileLines
    """Joins a line ending in '\' with the previous line (except when following
    comments).  The joined line takes on the index of the first line.
    """
    primary_line_number = None
    new_line = []  # type: List[Text]
    for line_number, line in lines_enum:
        if not line.endswith('\\') or COMMENT_RE.match(line):
            if COMMENT_RE.match(line):
                # this ensures comments are always matched later
                line = ' ' + line
            if new_line:
                new_line.append(line)
                yield primary_line_number, ''.join(new_line)
                new_line = []
            else:
                yield line_number, line
        else:
            if not new_line:
                primary_line_number = line_number
            new_line.append(line.strip('\\'))

    # last line contains \
    if new_line:
        yield primary_line_number, ''.join(new_line)

    # TODO: handle space after '\'.


def ignore_comments(lines_enum):
    # type: (ReqFileLines) -> ReqFileLines
    """
    Strips comments and filter empty lines.
    """
    for line_number, line in lines_enum:
        line = COMMENT_RE.sub('', line)
        line = line.strip()
        if line:
            yield line_number, line


def skip_regex(lines_enum, pattern):
    # type: (ReqFileLines, str) -> ReqFileLines
    """
    Skip lines that match the provided pattern

    Note: the regex pattern is only built once
    """
    matcher = re.compile(pattern)
    lines_enum = filterfalse(lambda e: matcher.search(e[1]), lines_enum)
    return lines_enum


def expand_env_variables(lines_enum):
    # type: (ReqFileLines) -> ReqFileLines
    """Replace all environment variables that can be retrieved via `os.getenv`.

    The only allowed format for environment variables defined in the
    requirement file is `${MY_VARIABLE_1}` to ensure two things:

    1. Strings that contain a `$` aren't accidentally (partially) expanded.
    2. Ensure consistency across platforms for requirement files.

    These points are the result of a discussion on the `github pull
    request #3514 <https://github.com/pypa/pip/pull/3514>`_.

    Valid characters in variable names follow the `POSIX standard
    <http://pubs.opengroup.org/onlinepubs/9699919799/>`_ and are limited
    to uppercase letter, digits and the `_` (underscore).
    """
    for line_number, line in lines_enum:
        for env_var, var_name in ENV_VAR_RE.findall(line):
            value = os.getenv(var_name)
            if not value:
                continue

            line = line.replace(env_var, value)

        yield line_number, line


def get_file_content(url, session, comes_from=None):
    # type: (str, PipSession, Optional[str]) -> Tuple[str, Text]
    """Gets the content of a file; it may be a filename, file: URL, or
    http: URL.  Returns (location, content).  Content is unicode.
    Respects # -*- coding: declarations on the retrieved files.

    :param url:         File path or url.
    :param session:     PipSession instance.
    :param comes_from:  Origin description of requirements.
    """
    scheme = get_url_scheme(url)

    if scheme in ['http', 'https']:
        # FIXME: catch some errors
        resp = session.get(url)
        resp.raise_for_status()
        return resp.url, resp.text

    elif scheme == 'file':
        if comes_from and comes_from.startswith('http'):
            raise InstallationError(
                'Requirements file %s references URL %s, which is local'
                % (comes_from, url))

        path = url.split(':', 1)[1]
        path = path.replace('\\', '/')
        match = _url_slash_drive_re.match(path)
        if match:
            path = match.group(1) + ':' + path.split('|', 1)[1]
        path = urllib_parse.unquote(path)
        if path.startswith('/'):
            path = '/' + path.lstrip('/')
        url = path

    try:
        with open(url, 'rb') as f:
            content = auto_decode(f.read())
    except IOError as exc:
        raise InstallationError(
            'Could not open requirements file: %s' % str(exc)
        )
    return url, content


_url_slash_drive_re = re.compile(r'/*([a-z])\|', re.I)
