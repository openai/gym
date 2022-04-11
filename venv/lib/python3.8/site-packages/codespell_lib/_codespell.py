# -*- coding: utf-8 -*-
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; version 2 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see
# http://www.gnu.org/licenses/old-licenses/gpl-2.0.html.
"""
Copyright (C) 2010-2011  Lucas De Marchi <lucas.de.marchi@gmail.com>
Copyright (C) 2011  ProFUSION embedded systems
"""

from __future__ import print_function

import argparse
import codecs
import configparser
import fnmatch
import os
import re
import sys
import textwrap

word_regex_def = u"[\\w\\-'’`]+"
# While we want to treat characters like ( or " as okay for a starting break,
# these may occur unescaped in URIs, and so we are more restrictive on the
# endpoint.  Emails are more restrictive, so the endpoint remains flexible.
uri_regex_def = (u"(\\b(?:https?|[ts]?ftp|file|git|smb)://[^\\s]+(?=$|\\s)|"
                 u"\\b[\\w.%+-]+@[\\w.-]+\\b)")
encodings = ('utf-8', 'iso-8859-1')
USAGE = """
\t%prog [OPTIONS] [file1 file2 ... fileN]
"""
VERSION = '2.1.0'

supported_languages_en = ('en', 'en_GB', 'en_US', 'en_CA', 'en_AU')
supported_languages = supported_languages_en

# Users might want to link this file into /usr/local/bin, so we resolve the
# symbolic link path to the real path if necessary.
_data_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
_builtin_dictionaries = (
    # name, desc, name, err in aspell, correction in aspell, \
    # err dictionary array, rep dictionary array
    # The arrays must contain the names of aspell dictionaries
    # The aspell tests here aren't the ideal state, but the None's are
    # realistic for obscure words
    ('clear', 'for unambiguous errors', '',
        False, None, supported_languages_en, None),
    ('rare', 'for rare but valid words', '_rare',
        None, None, None, None),
    ('informal', 'for making informal words more formal', '_informal',
        True, True, supported_languages_en, supported_languages_en),
    ('usage', 'for replacing phrasing with recommended terms', '_usage',
        None, None, None, None),
    ('code', 'for words common to code and/or mathematics that might be typos', '_code',  # noqa: E501
        None, None, None, None,),
    ('names', 'for valid proper names that might be typos', '_names',
        None, None, None, None,),
    ('en-GB_to_en-US', 'for corrections from en-GB to en-US', '_en-GB_to_en-US',  # noqa: E501
        True, True, ('en_GB',), ('en_US',)),
)
_builtin_default = 'clear,rare'

# docs say os.EX_USAGE et al. are only available on Unix systems, so to be safe
# we protect and just use the values they are on macOS and Linux
EX_OK = 0
EX_USAGE = 64
EX_DATAERR = 65

# OPTIONS:
#
# ARGUMENTS:
#    dict_filename       The file containing the dictionary of misspellings.
#                        If set to '-', it will be read from stdin
#    file1 .. fileN      Files to check spelling


class QuietLevels(object):
    NONE = 0
    ENCODING = 1
    BINARY_FILE = 2
    DISABLED_FIXES = 4
    NON_AUTOMATIC_FIXES = 8
    FIXES = 16


class GlobMatch(object):
    def __init__(self, pattern):
        if pattern:
            # Pattern might be a list of comma-delimited strings
            self.pattern_list = ','.join(pattern).split(',')
        else:
            self.pattern_list = None

    def match(self, filename):
        if self.pattern_list is None:
            return False

        for p in self.pattern_list:
            if fnmatch.fnmatch(filename, p):
                return True

        return False


class Misspelling(object):
    def __init__(self, data, fix, reason):
        self.data = data
        self.fix = fix
        self.reason = reason


class TermColors(object):
    def __init__(self):
        self.FILE = '\033[33m'
        self.WWORD = '\033[31m'
        self.FWORD = '\033[32m'
        self.DISABLE = '\033[0m'

    def disable(self):
        self.FILE = ''
        self.WWORD = ''
        self.FWORD = ''
        self.DISABLE = ''


class Summary(object):
    def __init__(self):
        self.summary = {}

    def update(self, wrongword):
        if wrongword in self.summary:
            self.summary[wrongword] += 1
        else:
            self.summary[wrongword] = 1

    def __str__(self):
        keys = list(self.summary.keys())
        keys.sort()

        return "\n".join(["{0}{1:{width}}".format(
                         key,
                         self.summary.get(key),
                         width=15 - len(key)) for key in keys])


class FileOpener(object):
    def __init__(self, use_chardet, quiet_level):
        self.use_chardet = use_chardet
        if use_chardet:
            self.init_chardet()
        self.quiet_level = quiet_level

    def init_chardet(self):
        try:
            from chardet.universaldetector import UniversalDetector
        except ImportError:
            raise ImportError("There's no chardet installed to import from. "
                              "Please, install it and check your PYTHONPATH "
                              "environment variable")

        self.encdetector = UniversalDetector()

    def open(self, filename):
        if self.use_chardet:
            return self.open_with_chardet(filename)
        else:
            return self.open_with_internal(filename)

    def open_with_chardet(self, filename):
        self.encdetector.reset()
        with codecs.open(filename, 'rb') as f:
            for line in f:
                self.encdetector.feed(line)
                if self.encdetector.done:
                    break
        self.encdetector.close()
        encoding = self.encdetector.result['encoding']

        try:
            f = codecs.open(filename, 'r', encoding=encoding)
        except UnicodeDecodeError:
            print("ERROR: Could not detect encoding: %s" % filename,
                  file=sys.stderr)
            raise
        except LookupError:
            print("ERROR: Don't know how to handle encoding %s: %s"
                  % (encoding, filename,), file=sys.stderr)
            raise
        else:
            lines = f.readlines()
            f.close()

        return lines, encoding

    def open_with_internal(self, filename):
        curr = 0
        while True:
            try:
                f = codecs.open(filename, 'r', encoding=encodings[curr])
            except UnicodeDecodeError:
                if not self.quiet_level & QuietLevels.ENCODING:
                    print("WARNING: Decoding file using encoding=%s failed: %s"
                          % (encodings[curr], filename,), file=sys.stderr)
                    try:
                        print("WARNING: Trying next encoding %s"
                              % encodings[curr + 1], file=sys.stderr)
                    except IndexError:
                        pass

                curr += 1
            else:
                lines = f.readlines()
                f.close()
                break
        if not lines:
            raise Exception('Unknown encoding')

        encoding = encodings[curr]

        return lines, encoding

# -.-:-.-:-.-:-.:-.-:-.-:-.-:-.-:-.:-.-:-.-:-.-:-.-:-.:-.-:-


# If someday this breaks, we can just switch to using RawTextHelpFormatter,
# but it has the disadvantage of not wrapping our long lines.

class NewlineHelpFormatter(argparse.HelpFormatter):
    """Help formatter that preserves newlines and deals with lists."""

    def _split_lines(self, text, width):
        parts = text.split('\n')
        out = list()
        for pi, part in enumerate(parts):
            # Eventually we could allow others...
            indent_start = '- '
            if part.startswith(indent_start):
                offset = len(indent_start)
            else:
                offset = 0
            part = part[offset:]
            part = self._whitespace_matcher.sub(' ', part).strip()
            parts = textwrap.wrap(part, width - offset)
            parts = [' ' * offset + p for p in parts]
            if offset:
                parts[0] = indent_start + parts[0][offset:]
            out.extend(parts)
        return out


def parse_options(args):
    parser = argparse.ArgumentParser(formatter_class=NewlineHelpFormatter)

    parser.set_defaults(colors=sys.stdout.isatty())
    parser.add_argument('--version', action='version', version=VERSION)

    parser.add_argument('-d', '--disable-colors',
                        action='store_false', dest='colors',
                        help='disable colors, even when printing to terminal '
                             '(always set for Windows)')
    parser.add_argument('-c', '--enable-colors',
                        action='store_true', dest='colors',
                        help='enable colors, even when not printing to '
                             'terminal')

    parser.add_argument('-w', '--write-changes',
                        action='store_true', default=False,
                        help='write changes in place if possible')

    parser.add_argument('-D', '--dictionary',
                        action='append',
                        help='custom dictionary file that contains spelling '
                             'corrections. If this flag is not specified or '
                             'equals "-" then the default dictionary is used. '
                             'This option can be specified multiple times.')
    builtin_opts = '\n- '.join([''] + [
        '%r %s' % (d[0], d[1]) for d in _builtin_dictionaries])
    parser.add_argument('--builtin',
                        dest='builtin', default=_builtin_default,
                        metavar='BUILTIN-LIST',
                        help='comma-separated list of builtin dictionaries '
                        'to include (when "-D -" or no "-D" is passed). '
                        'Current options are:' + builtin_opts + '\n'
                        'The default is %(default)r.')
    parser.add_argument('--ignore-regex',
                        action='store', type=str,
                        help='regular expression which is used to find '
                             'patterns to ignore by treating as whitespace. '
                             'When writing regular expressions, consider '
                             'ensuring there are boundary non-word chars, '
                             'e.g., "\\bmatch\\b". Defaults to '
                             'empty/disabled.')
    parser.add_argument('-I', '--ignore-words',
                        action='append', metavar='FILE',
                        help='file that contains words which will be ignored '
                             'by codespell. File must contain 1 word per line.'
                             ' Words are case sensitive based on how they are '
                             'written in the dictionary file')
    parser.add_argument('-L', '--ignore-words-list',
                        action='append', metavar='WORDS',
                        help='comma separated list of words to be ignored '
                             'by codespell. Words are case sensitive based on '
                             'how they are written in the dictionary file')
    parser.add_argument('--uri-ignore-words-list',
                        action='append', metavar='WORDS',
                        help='comma separated list of words to be ignored '
                             'by codespell in URIs and emails only. Words are '
                             'case sensitive based on how they are written in '
                             'the dictionary file. If set to "*", all '
                             'misspelling in URIs and emails will be ignored.')
    parser.add_argument('-r', '--regex',
                        action='store', type=str,
                        help='regular expression which is used to find words. '
                             'By default any alphanumeric character, the '
                             'underscore, the hyphen, and the apostrophe is '
                             'used to build words. This option cannot be '
                             'specified together with --write-changes.')
    parser.add_argument('--uri-regex',
                        action='store', type=str,
                        help='regular expression which is used to find URIs '
                             'and emails. A default expression is provided.')
    parser.add_argument('-s', '--summary',
                        action='store_true', default=False,
                        help='print summary of fixes')

    parser.add_argument('--count',
                        action='store_true', default=False,
                        help='print the number of errors as the last line of '
                             'stderr')

    parser.add_argument('-S', '--skip',
                        action='append',
                        help='comma-separated list of files to skip. It '
                             'accepts globs as well. E.g.: if you want '
                             'codespell to skip .eps and .txt files, '
                             'you\'d give "*.eps,*.txt" to this option.')

    parser.add_argument('-x', '--exclude-file', type=str, metavar='FILE',
                        help='FILE with lines that should not be checked for '
                             'errors or changed')

    parser.add_argument('-i', '--interactive',
                        action='store', type=int, default=0,
                        help='set interactive mode when writing changes:\n'
                             '- 0: no interactivity.\n'
                             '- 1: ask for confirmation.\n'
                             '- 2: ask user to choose one fix when more than one is available.\n'  # noqa: E501
                             '- 3: both 1 and 2')

    parser.add_argument('-q', '--quiet-level',
                        action='store', type=int, default=2,
                        help='bitmask that allows suppressing messages:\n'
                             '- 0: print all messages.\n'
                             '- 1: disable warnings about wrong encoding.\n'
                             '- 2: disable warnings about binary files.\n'
                             '- 4: omit warnings about automatic fixes that were disabled in the dictionary.\n'  # noqa: E501
                             '- 8: don\'t print anything for non-automatic fixes.\n'  # noqa: E501
                             '- 16: don\'t print the list of fixed files.\n'
                             'As usual with bitmasks, these levels can be '
                             'combined; e.g. use 3 for levels 1+2, 7 for '
                             '1+2+4, 23 for 1+2+4+16, etc. '
                             'The default mask is %(default)s.')

    parser.add_argument('-e', '--hard-encoding-detection',
                        action='store_true', default=False,
                        help='use chardet to detect the encoding of each '
                             'file. This can slow down codespell, but is more '
                             'reliable in detecting encodings other than '
                             'utf-8, iso8859-1, and ascii.')

    parser.add_argument('-f', '--check-filenames',
                        action='store_true', default=False,
                        help='check file names as well')

    parser.add_argument('-H', '--check-hidden',
                        action='store_true', default=False,
                        help='check hidden files and directories (those '
                             'starting with ".") as well.')
    parser.add_argument('-A', '--after-context', type=int, metavar='LINES',
                        help='print LINES of trailing context')
    parser.add_argument('-B', '--before-context', type=int, metavar='LINES',
                        help='print LINES of leading context')
    parser.add_argument('-C', '--context', type=int, metavar='LINES',
                        help='print LINES of surrounding context')
    parser.add_argument('--config', type=str,
                        help='path to config file.')

    parser.add_argument('files', nargs='*',
                        help='files or directories to check')

    # Parse command line options.
    options = parser.parse_args(list(args))

    # Load config files and look for ``codespell`` options.
    cfg_files = ['setup.cfg', '.codespellrc']
    if options.config:
        cfg_files.append(options.config)
    config = configparser.ConfigParser()
    config.read(cfg_files)

    if config.has_section('codespell'):
        # Build a "fake" argv list using option name and value.
        cfg_args = []
        for key in config['codespell']:
            # Add option as arg.
            cfg_args.append("--%s" % key)
            # If value is blank, skip.
            val = config['codespell'][key]
            if val != "":
                cfg_args.append(val)

        # Parse config file options.
        options = parser.parse_args(cfg_args)

        # Re-parse command line options to override config.
        options = parser.parse_args(list(args), namespace=options)

    if not options.files:
        options.files.append('.')

    return options, parser


def parse_ignore_words_option(ignore_words_option):
    ignore_words = set()
    if ignore_words_option:
        for comma_separated_words in ignore_words_option:
            for word in comma_separated_words.split(','):
                ignore_words.add(word.strip())
    return ignore_words


def build_exclude_hashes(filename, exclude_lines):
    with codecs.open(filename, 'r') as f:
        for line in f:
            exclude_lines.add(line)


def build_ignore_words(filename, ignore_words):
    with codecs.open(filename, mode='r', encoding='utf-8') as f:
        for line in f:
            ignore_words.add(line.strip())


def build_dict(filename, misspellings, ignore_words):
    with codecs.open(filename, mode='r', encoding='utf-8') as f:
        for line in f:
            [key, data] = line.split('->')
            # TODO for now, convert both to lower. Someday we can maybe add
            # support for fixing caps.
            key = key.lower()
            data = data.lower()
            if key in ignore_words:
                continue
            data = data.strip()
            fix = data.rfind(',')

            if fix < 0:
                fix = True
                reason = ''
            elif fix == (len(data) - 1):
                data = data[:fix]
                reason = ''
                fix = False
            else:
                reason = data[fix + 1:].strip()
                data = data[:fix]
                fix = False

            misspellings[key] = Misspelling(data, fix, reason)


def is_hidden(filename, check_hidden):
    bfilename = os.path.basename(filename)

    return bfilename not in ('', '.', '..') and \
        (not check_hidden and bfilename[0] == '.')


def is_text_file(filename):
    with open(filename, mode='rb') as f:
        s = f.read(1024)
    if b'\x00' in s:
        return False
    return True


def fix_case(word, fixword):
    if word == word.capitalize():
        return fixword.capitalize()
    elif word == word.upper():
        return fixword.upper()
    # they are both lower case
    # or we don't have any idea
    return fixword


def ask_for_word_fix(line, wrongword, misspelling, interactivity):
    if interactivity <= 0:
        return misspelling.fix, fix_case(wrongword, misspelling.data)

    if misspelling.fix and interactivity & 1:
        r = ''
        fixword = fix_case(wrongword, misspelling.data)
        while not r:
            print("%s\t%s ==> %s (Y/n) " % (line, wrongword, fixword), end='')
            r = sys.stdin.readline().strip().upper()
            if not r:
                r = 'Y'
            if r != 'Y' and r != 'N':
                print("Say 'y' or 'n'")
                r = ''

        if r == 'N':
            misspelling.fix = False
            misspelling.fixword = ''

    elif (interactivity & 2) and not misspelling.reason:
        # if it is not disabled, i.e. it just has more than one possible fix,
        # we ask the user which word to use

        r = ''
        opt = list(map(lambda x: x.strip(), misspelling.data.split(',')))
        while not r:
            print("%s Choose an option (blank for none): " % line, end='')
            for i in range(len(opt)):
                fixword = fix_case(wrongword, opt[i])
                print(" %d) %s" % (i, fixword), end='')
            print(": ", end='')
            sys.stdout.flush()

            n = sys.stdin.readline().strip()
            if not n:
                break

            try:
                n = int(n)
                r = opt[n]
            except (ValueError, IndexError):
                print("Not a valid option\n")

        if r:
            misspelling.fix = True
            misspelling.data = r

    return misspelling.fix, fix_case(wrongword, misspelling.data)


def print_context(lines, index, context):
    # context = (context_before, context_after)
    for i in range(index - context[0], index + context[1] + 1):
        if 0 <= i < len(lines):
            print('%s %s' % ('>' if i == index else ':', lines[i].rstrip()))


def extract_words(text, word_regex, ignore_word_regex):
    if ignore_word_regex:
        text = ignore_word_regex.sub(' ', text)
    return word_regex.findall(text)


def apply_uri_ignore_words(check_words, line, word_regex, ignore_word_regex,
                           uri_regex, uri_ignore_words):
    if not uri_ignore_words:
        return
    for uri in re.findall(uri_regex, line):
        for uri_word in extract_words(uri, word_regex,
                                      ignore_word_regex):
            if uri_word in uri_ignore_words:
                check_words.remove(uri_word)


def parse_file(filename, colors, summary, misspellings, exclude_lines,
               file_opener, word_regex, ignore_word_regex, uri_regex,
               uri_ignore_words, context, options):
    bad_count = 0
    lines = None
    changed = False
    encoding = encodings[0]  # if not defined, use UTF-8

    if filename == '-':
        f = sys.stdin
        lines = f.readlines()
    else:
        if options.check_filenames:
            for word in extract_words(filename, word_regex, ignore_word_regex):
                lword = word.lower()
                if lword not in misspellings:
                    continue
                fix = misspellings[lword].fix
                fixword = fix_case(word, misspellings[lword].data)

                if summary and fix:
                    summary.update(lword)

                cfilename = "%s%s%s" % (colors.FILE, filename, colors.DISABLE)
                cwrongword = "%s%s%s" % (colors.WWORD, word, colors.DISABLE)
                crightword = "%s%s%s" % (colors.FWORD, fixword, colors.DISABLE)

                if misspellings[lword].reason:
                    if options.quiet_level & QuietLevels.DISABLED_FIXES:
                        continue
                    creason = "  | %s%s%s" % (colors.FILE,
                                              misspellings[lword].reason,
                                              colors.DISABLE)
                else:
                    if options.quiet_level & QuietLevels.NON_AUTOMATIC_FIXES:
                        continue
                    creason = ''

                bad_count += 1

                print("%(FILENAME)s: %(WRONGWORD)s"
                      " ==> %(RIGHTWORD)s%(REASON)s"
                      % {'FILENAME': cfilename,
                         'WRONGWORD': cwrongword,
                         'RIGHTWORD': crightword, 'REASON': creason})

        # ignore irregular files
        if not os.path.isfile(filename):
            return bad_count

        text = is_text_file(filename)
        if not text:
            if not options.quiet_level & QuietLevels.BINARY_FILE:
                print("WARNING: Binary file: %s" % filename, file=sys.stderr)
            return bad_count
        try:
            lines, encoding = file_opener.open(filename)
        except Exception:
            return bad_count

    for i, line in enumerate(lines):
        if line in exclude_lines:
            continue

        fixed_words = set()
        asked_for = set()

        # If all URI spelling errors will be ignored, erase any URI before
        # extracting words. Otherwise, apply ignores after extracting words.
        # This ensures that if a URI ignore word occurs both inside a URI and
        # outside, it will still be a spelling error.
        if "*" in uri_ignore_words:
            line = uri_regex.sub(' ', line)
        check_words = extract_words(line, word_regex, ignore_word_regex)
        if "*" not in uri_ignore_words:
            apply_uri_ignore_words(check_words, line, word_regex,
                                   ignore_word_regex, uri_regex,
                                   uri_ignore_words)

        for word in check_words:
            lword = word.lower()
            if lword in misspellings:
                context_shown = False
                fix = misspellings[lword].fix
                fixword = fix_case(word, misspellings[lword].data)

                if options.interactive and lword not in asked_for:
                    if context is not None:
                        context_shown = True
                        print_context(lines, i, context)
                    fix, fixword = ask_for_word_fix(
                        lines[i], word, misspellings[lword],
                        options.interactive)
                    asked_for.add(lword)

                if summary and fix:
                    summary.update(lword)

                if word in fixed_words:  # can skip because of re.sub below
                    continue

                if options.write_changes and fix:
                    changed = True
                    lines[i] = re.sub(r'\b%s\b' % word, fixword, lines[i])
                    fixed_words.add(word)
                    continue

                # otherwise warning was explicitly set by interactive mode
                if (options.interactive & 2 and not fix and not
                        misspellings[lword].reason):
                    continue

                cfilename = "%s%s%s" % (colors.FILE, filename, colors.DISABLE)
                cline = "%s%d%s" % (colors.FILE, i + 1, colors.DISABLE)
                cwrongword = "%s%s%s" % (colors.WWORD, word, colors.DISABLE)
                crightword = "%s%s%s" % (colors.FWORD, fixword, colors.DISABLE)

                if misspellings[lword].reason:
                    if options.quiet_level & QuietLevels.DISABLED_FIXES:
                        continue

                    creason = "  | %s%s%s" % (colors.FILE,
                                              misspellings[lword].reason,
                                              colors.DISABLE)
                else:
                    if options.quiet_level & QuietLevels.NON_AUTOMATIC_FIXES:
                        continue

                    creason = ''

                # If we get to this point (uncorrected error) we should change
                # our bad_count and thus return value
                bad_count += 1

                if (not context_shown) and (context is not None):
                    print_context(lines, i, context)
                if filename != '-':
                    print("%(FILENAME)s:%(LINE)s: %(WRONGWORD)s "
                          "==> %(RIGHTWORD)s%(REASON)s"
                          % {'FILENAME': cfilename, 'LINE': cline,
                             'WRONGWORD': cwrongword,
                             'RIGHTWORD': crightword, 'REASON': creason})
                else:
                    print("%(LINE)s: %(STRLINE)s\n\t%(WRONGWORD)s "
                          "==> %(RIGHTWORD)s%(REASON)s"
                          % {'LINE': cline, 'STRLINE': line.strip(),
                             'WRONGWORD': cwrongword,
                             'RIGHTWORD': crightword, 'REASON': creason})

    if changed:
        if filename == '-':
            print("---")
            for line in lines:
                print(line, end='')
        else:
            if not options.quiet_level & QuietLevels.FIXES:
                print("%sFIXED:%s %s"
                      % (colors.FWORD, colors.DISABLE, filename),
                      file=sys.stderr)
            with codecs.open(filename, 'w', encoding=encoding) as f:
                f.writelines(lines)
    return bad_count


def _script_main():
    """Wrap to main() for setuptools."""
    return main(*sys.argv[1:])


def main(*args):
    """Contains flow control"""
    options, parser = parse_options(args)

    if options.regex and options.write_changes:
        print("ERROR: --write-changes cannot be used together with "
              "--regex")
        parser.print_help()
        return EX_USAGE
    word_regex = options.regex or word_regex_def
    try:
        word_regex = re.compile(word_regex)
    except re.error as err:
        print("ERROR: invalid --regex \"%s\" (%s)" %
              (word_regex, err), file=sys.stderr)
        parser.print_help()
        return EX_USAGE

    if options.ignore_regex:
        try:
            ignore_word_regex = re.compile(options.ignore_regex)
        except re.error as err:
            print("ERROR: invalid --ignore-regex \"%s\" (%s)" %
                  (options.ignore_regex, err), file=sys.stderr)
            parser.print_help()
            return EX_USAGE
    else:
        ignore_word_regex = None

    ignore_words_files = options.ignore_words or []
    ignore_words = parse_ignore_words_option(options.ignore_words_list)
    for ignore_words_file in ignore_words_files:
        if not os.path.isfile(ignore_words_file):
            print("ERROR: cannot find ignore-words file: %s" %
                  ignore_words_file, file=sys.stderr)
            parser.print_help()
            return EX_USAGE
        build_ignore_words(ignore_words_file, ignore_words)

    uri_regex = options.uri_regex or uri_regex_def
    try:
        uri_regex = re.compile(uri_regex)
    except re.error as err:
        print("ERROR: invalid --uri-regex \"%s\" (%s)" %
              (uri_regex, err), file=sys.stderr)
        parser.print_help()
        return EX_USAGE
    uri_ignore_words = parse_ignore_words_option(options.uri_ignore_words_list)

    if options.dictionary:
        dictionaries = options.dictionary
    else:
        dictionaries = ['-']
    use_dictionaries = list()
    for dictionary in dictionaries:
        if dictionary == "-":
            # figure out which builtin dictionaries to use
            use = sorted(set(options.builtin.split(',')))
            for u in use:
                for builtin in _builtin_dictionaries:
                    if builtin[0] == u:
                        use_dictionaries.append(
                            os.path.join(_data_root, 'dictionary%s.txt'
                                         % (builtin[2],)))
                        break
                else:
                    print("ERROR: Unknown builtin dictionary: %s" % (u,),
                          file=sys.stderr)
                    parser.print_help()
                    return EX_USAGE
        else:
            if not os.path.isfile(dictionary):
                print("ERROR: cannot find dictionary file: %s" % dictionary,
                      file=sys.stderr)
                parser.print_help()
                return EX_USAGE
            use_dictionaries.append(dictionary)
    misspellings = dict()
    for dictionary in use_dictionaries:
        build_dict(dictionary, misspellings, ignore_words)
    colors = TermColors()
    if not options.colors or sys.platform == 'win32':
        colors.disable()

    if options.summary:
        summary = Summary()
    else:
        summary = None

    context = None
    if options.context is not None:
        if (options.before_context is not None) or \
                (options.after_context is not None):
            print("ERROR: --context/-C cannot be used together with "
                  "--context-before/-B or --context-after/-A")
            parser.print_help()
            return EX_USAGE
        context_both = max(0, options.context)
        context = (context_both, context_both)
    elif (options.before_context is not None) or \
            (options.after_context is not None):
        context_before = 0
        context_after = 0
        if options.before_context is not None:
            context_before = max(0, options.before_context)
        if options.after_context is not None:
            context_after = max(0, options.after_context)
        context = (context_before, context_after)

    exclude_lines = set()
    if options.exclude_file:
        build_exclude_hashes(options.exclude_file, exclude_lines)

    file_opener = FileOpener(options.hard_encoding_detection,
                             options.quiet_level)
    glob_match = GlobMatch(options.skip)

    bad_count = 0
    for filename in options.files:
        # ignore hidden files
        if is_hidden(filename, options.check_hidden):
            continue

        if os.path.isdir(filename):
            for root, dirs, files in os.walk(filename):
                if glob_match.match(root):  # skip (absolute) directories
                    del dirs[:]
                    continue
                if is_hidden(root, options.check_hidden):  # dir itself hidden
                    continue
                for file_ in files:
                    # ignore hidden files in directories
                    if is_hidden(file_, options.check_hidden):
                        continue
                    if glob_match.match(file_):  # skip files
                        continue
                    fname = os.path.join(root, file_)
                    if glob_match.match(fname):  # skip paths
                        continue
                    bad_count += parse_file(
                        fname, colors, summary, misspellings, exclude_lines,
                        file_opener, word_regex, ignore_word_regex, uri_regex,
                        uri_ignore_words, context, options)

                # skip (relative) directories
                dirs[:] = [dir_ for dir_ in dirs if not glob_match.match(dir_)]

        elif not glob_match.match(filename):  # skip files
            bad_count += parse_file(
                filename, colors, summary, misspellings, exclude_lines,
                file_opener, word_regex, ignore_word_regex, uri_regex,
                uri_ignore_words, context, options)

    if summary:
        print("\n-------8<-------\nSUMMARY:")
        print(summary)
    if options.count:
        print(bad_count, file=sys.stderr)
    return EX_DATAERR if bad_count else EX_OK
