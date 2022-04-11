# ----------------------------------------------------------------------------
# pyglet
# Copyright (c) 2006-2008 Alex Holkner
# Copyright (c) 2008-2021 pyglet contributors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#  * Neither the name of pyglet nor the names of its
#    contributors may be used to endorse or promote products
#    derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ----------------------------------------------------------------------------

"""Extensible attributed text format for representing pyglet formatted
documents.
"""

import re
import ast

import pyglet

_pattern = re.compile(r"""
    (?P<escape_hex>\{\#x(?P<escape_hex_val>[0-9a-fA-F]+)\})
  | (?P<escape_dec>\{\#(?P<escape_dec_val>[0-9]+)\})
  | (?P<escape_lbrace>\{\{)
  | (?P<escape_rbrace>\}\})
  | (?P<attr>\{
        (?P<attr_name>[^ \{\}]+)\s+
        (?P<attr_val>[^\}]+)\})
  | (?P<nl_hard1>\n(?=[ \t]))
  | (?P<nl_hard2>\{\}\n)
  | (?P<nl_soft>\n(?=\S))
  | (?P<nl_para>\n\n+)
  | (?P<text>[^\{\}\n]+)
    """, re.VERBOSE | re.DOTALL)


class AttributedTextDecoder(pyglet.text.DocumentDecoder):

    def __init__(self):
        self.doc = pyglet.text.document.FormattedDocument()
        self.length = 0
        self.attributes = {}

    def decode(self, text, location=None):
        next_trailing_space = True
        trailing_newline = True

        for m in _pattern.finditer(text):
            group = m.lastgroup
            trailing_space = True
            if group == 'text':
                t = m.group('text')
                self.append(t)
                trailing_space = t.endswith(' ')
                trailing_newline = False
            elif group == 'nl_soft':
                if not next_trailing_space:
                    self.append(' ')
                trailing_newline = False
            elif group in ('nl_hard1', 'nl_hard2'):
                self.append('\n')
                trailing_newline = True
            elif group == 'nl_para':
                self.append(m.group('nl_para')[1:])  # ignore the first \n
                trailing_newline = True
            elif group == 'attr':
                value = ast.literal_eval(m.group('attr_val'))
                name = m.group('attr_name')
                if name[0] == '.':
                    if trailing_newline:
                        self.attributes[name[1:]] = value
                    else:
                        self.doc.set_paragraph_style(self.length, self.length, {name[1:]: value})
                else:
                    self.attributes[name] = value
            elif group == 'escape_dec':
                self.append(chr(int(m.group('escape_dec_val'))))
            elif group == 'escape_hex':
                self.append(chr(int(m.group('escape_hex_val'), 16)))
            elif group == 'escape_lbrace':
                self.append('{')
            elif group == 'escape_rbrace':
                self.append('}')
            next_trailing_space = trailing_space

        return self.doc

    def append(self, text):
        self.doc.insert_text(self.length, text, self.attributes)
        self.length += len(text)
        self.attributes.clear()
