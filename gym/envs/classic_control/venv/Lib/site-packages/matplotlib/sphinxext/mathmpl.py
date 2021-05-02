import hashlib
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive, directives
import sphinx

import matplotlib as mpl
from matplotlib import _api, mathtext


# Define LaTeX math node:
class latex_math(nodes.General, nodes.Element):
    pass


def fontset_choice(arg):
    return directives.choice(arg, mathtext.MathTextParser._font_type_mapping)


def math_role(role, rawtext, text, lineno, inliner,
              options={}, content=[]):
    i = rawtext.find('`')
    latex = rawtext[i+1:-1]
    node = latex_math(rawtext)
    node['latex'] = latex
    node['fontset'] = options.get('fontset', 'cm')
    return [node], []
math_role.options = {'fontset': fontset_choice}


class MathDirective(Directive):
    has_content = True
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {'fontset': fontset_choice}

    def run(self):
        latex = ''.join(self.content)
        node = latex_math(self.block_text)
        node['latex'] = latex
        node['fontset'] = self.options.get('fontset', 'cm')
        return [node]


# This uses mathtext to render the expression
def latex2png(latex, filename, fontset='cm'):
    latex = "$%s$" % latex
    with mpl.rc_context({'mathtext.fontset': fontset}):
        try:
            depth = mathtext.math_to_image(
                latex, filename, dpi=100, format="png")
        except Exception:
            _api.warn_external(f"Could not render math expression {latex}")
            depth = 0
    return depth


# LaTeX to HTML translation stuff:
def latex2html(node, source):
    inline = isinstance(node.parent, nodes.TextElement)
    latex = node['latex']
    fontset = node['fontset']
    name = 'math-{}'.format(
        hashlib.md5((latex + fontset).encode()).hexdigest()[-10:])

    destdir = Path(setup.app.builder.outdir, '_images', 'mathmpl')
    destdir.mkdir(parents=True, exist_ok=True)
    dest = destdir / f'{name}.png'

    depth = latex2png(latex, dest, fontset)

    if inline:
        cls = ''
    else:
        cls = 'class="center" '
    if inline and depth != 0:
        style = 'style="position: relative; bottom: -%dpx"' % (depth + 1)
    else:
        style = ''

    return (f'<img src="{setup.app.builder.imgpath}/mathmpl/{name}.png"'
            f' {cls}{style}/>')


def setup(app):
    setup.app = app

    # Add visit/depart methods to HTML-Translator:
    def visit_latex_math_html(self, node):
        source = self.document.attributes['source']
        self.body.append(latex2html(node, source))

    def depart_latex_math_html(self, node):
        pass

    # Add visit/depart methods to LaTeX-Translator:
    def visit_latex_math_latex(self, node):
        inline = isinstance(node.parent, nodes.TextElement)
        if inline:
            self.body.append('$%s$' % node['latex'])
        else:
            self.body.extend(['\\begin{equation}',
                              node['latex'],
                              '\\end{equation}'])

    def depart_latex_math_latex(self, node):
        pass

    app.add_node(latex_math,
                 html=(visit_latex_math_html, depart_latex_math_html),
                 latex=(visit_latex_math_latex, depart_latex_math_latex))
    app.add_role('mathmpl', math_role)
    app.add_directive('mathmpl', MathDirective)
    if sphinx.version_info < (1, 8):
        app.add_role('math', math_role)
        app.add_directive('math', MathDirective)

    metadata = {'parallel_read_safe': True, 'parallel_write_safe': True}
    return metadata
