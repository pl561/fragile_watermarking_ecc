from __future__ import print_function
import os
HOME = os.environ["HOME"]

import sys
import pandas as pd
import numpy as np

def tolatex_confmat(conf_mat):
    latex_confmat = pd.DataFrame(conf_mat)
    pdtable = latex_confmat.to_latex(index=False, header=False)
    lines = pdtable.splitlines()[2:-2]
    lines = [" "*4 + line for line in lines]
    begin = "\\begin{equation*}\n  C =\n  \\begin{pmatrix*}"
    end = "  \\end{pmatrix*}  \n  \\label{eq:confmat_}\n\\end{equation*}\n"
    lines.insert(0, begin)
    lines.append(end)
    lines = "\n".join(lines)
    return lines

def indent(content, tab=2):
    lines = content.splitlines()
    indented = "\n".join([" "*2 + line for line in lines])
    return indented

def env(envname, content, tab=2, params=None):
    latex_list = [
        "\\begin{",
        envname,
        "}",
        "\n",
        indent(content),
        "\n",
        "\\end{",
        envname,
        "}\n"
        ]
    if params is not None:
        latex_list.insert(3, params)
    latex_string = "".join(latex_list)
    return latex_string


def includegraphics(epath, parent="",
                    width="3.8cm", caption=None, label=None):
    # to do : a lot of improvement can be done
    p = epath.replace_parents(parent)
    latex_list = [
        "\\includegraphics",
        "[",
        "]",
        "{",
        str(p),
        "}\n"
    ]
    if width is not None:
        latex_list.insert(2, "width={}".format(width))

    latex_string = "".join(latex_list)
    return latex_string


def image_figure(epaths, parent="../images",
                 width="3.8cm", caption="", params=None):
    assert isinstance(epaths, list)
    ig = [includegraphics(ep, parent=parent, width=width) for ep in epaths]
    ig.append("\\caption{")
    ig.append(caption)
    ig.append("}")
    ig.append("\n")
    ig.append("\\label{fig:")
    ig.append(str(epaths[0].stem()))
    ig.append("}\n")
    ig.append("%% Fig.~\\ref{fig:")
    ig.append(epaths[0].stem().string())
    ig.append("}\n")

    # bad solution !
    ig = map(str, ig)

    joined = "".join(ig)

    ls = env("figure", joined, params=params)
    return ls



def main():
    import pathtools as pt
    eps = [pt.EPath("image_123_22.png"), pt.EPath("image_12781_7821.png")]
    sys.stdout.write(image_figure(eps, width="4cm", caption="Test caption"))


if __name__ == "__main__":
    sys.exit(main())