::: {.document}
::: {.documentwrapper}
::: {.bodywrapper}
::: {.body role="main"}
::: {#module-pymgdg .section}
[]{#multigrid-discontinuous-galerkin-reaction-diffusion-solver}

Multigrid Discontinuous Galerkin Reaction Diffusion solver[¶](#module-pymgdg "Permalink to this headline"){.headerlink}
=======================================================================================================================

::: {#pymgdg-py .section}
pymgdg.py[¶](#pymgdg-py "Permalink to this headline"){.headerlink}
------------------------------------------------------------------

Solution of a symmetric interior penalty discontinuous Galerkin (SIPG)
discretized, singularly perturbed reaction-diffusion equation in 1D,
using linear finite elements.

*class* `pymgdg.`{.sig-prename .descclassname}`CoarseCorrection`{.sig-name .descname}[(]{.sig-paren}*dc*[)]{.sig-paren}[¶](#pymgdg.CoarseCorrection "Permalink to this definition"){.headerlink}

:   Coarse correction object.

    Parameters

    :   **dc** -- Discrete operator

    `assemble`{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[¶](#pymgdg.CoarseCorrection.assemble "Permalink to this definition"){.headerlink}

    :   Assemble coarse correction matrix

<!-- -->

*class* `pymgdg.`{.sig-prename .descclassname}`DiscreteOperator`{.sig-name .descname}[(]{.sig-paren}*problem*, *n*, *periodic\_=False*[)]{.sig-paren}[¶](#pymgdg.DiscreteOperator "Permalink to this definition"){.headerlink}

:   Class implementing the discrete operator corresponding to a problem.

    Parameters

    :   -   **problem** -- Object containing the integrals corresponding
            to cell, boundary and faces.

        -   **n** (*int*) -- Total cells in the mesh.

        -   **periodic** (*bool*) -- True: Periodic boundary conditions,
            False: Dirichlet boundary conditions.

    `assemble`{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[¶](#pymgdg.DiscreteOperator.assemble "Permalink to this definition"){.headerlink}

    :   Assembles the symbolic matrix.

    `assemble_cellBJ`{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[¶](#pymgdg.DiscreteOperator.assemble_cellBJ "Permalink to this definition"){.headerlink}

    :   Assembles the \_cell\_ Block Jacobi matrix.

    `assemble_pointBJ`{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[¶](#pymgdg.DiscreteOperator.assemble_pointBJ "Permalink to this definition"){.headerlink}

    :   Assembles the \_point\_ Block Jacobi matrix.

    `nassemble`{.sig-name .descname}[(]{.sig-paren}*par*[)]{.sig-paren}[¶](#pymgdg.DiscreteOperator.nassemble "Permalink to this definition"){.headerlink}

    :   Assembles the numerical matrix.

        Parameters

        :   **par** (*{}*) -- Values for all the symbols in the symbolic
            matrix.

<!-- -->

*class* `pymgdg.`{.sig-prename .descclassname}`LagrangeBasis`{.sig-name .descname}[(]{.sig-paren}*p*[)]{.sig-paren}[¶](#pymgdg.LagrangeBasis "Permalink to this definition"){.headerlink}

:   Lagrange-type basis class, containing the same lagrange shape and
    test functions if order 'p'.

    Parameters

    :   **p** (*int*) -- Order of the Lagrange polynomial used as a
        shape and test functions.

<!-- -->

*class* `pymgdg.`{.sig-prename .descclassname}`ReactionDiffusion`{.sig-name .descname}[(]{.sig-paren}*bs*[)]{.sig-paren}[¶](#pymgdg.ReactionDiffusion "Permalink to this definition"){.headerlink}

:   Class containing specific data about the problem to be solved, in
    this case corresponding to the bilinear form of a DG Reaction
    Diffusion differential equation.

    Parameters

    :   -   **bs** (*Object containing the basis functions to be used
            and the order of the polynomial* ) --

        -   **desired.** (*degree*) --

    ::: {.math .notranslate .nohighlight}
    \\\[\\newcommand\\scalemath\[2\]{\\scalebox{\#1}{\\mbox{\\ensuremath{\\displaystyle
    \#2}}}} \\newcommand{\\mesh}{\\mathbb{T}}
    \\newcommand{\\cell}{\\kappa} \\newcommand{\\meshfaces}{\\mathbb{F}}
    \\newcommand{\\face}{f}
    \\newcommand{\\ipbf}\[2\]{a\_h\\left(\#1,\#2\\right)}
    \\newcommand{\\ddx}\[1\]{\\frac{d \#1}{dx}}
    \\newcommand{\\eps}{\\varepsilon}
    \\newcommand{\\jump}\[1\]{\\left\[\\!\\left\[\#1\\right\]\\!\\right\]}
    \\newcommand{\\av}\[1\]{\\left\\{\\!\\!\\left\\{\#1\\right\\}\\!\\!\\right\\}}
    \\newcommand{\\avv}\[1\]{\\left\\{\\!\\!\\!\\left\\{\#1\\right\\}\\!\\!\\!\\right\\}}
    \\newcommand\\w\[1\]{\\makebox\[2.5em\]{\$\#1\$}}
    \\newcommand{\\e}\[1\]{e\^{\#1}} \\newcommand{\\I}{i}
    \\newcommand{\\phih}{\\boldsymbol{\\varphi}}
    \\newcommand{\\phiH}{\\boldsymbol{\\phi}} \\newcommand{\\kl}{k}
    \\newcommand{\\kh}{{\\widetilde{k}}}
    \\newcommand{\\dd}{\\delta\_0}\\\]
    :::

    Notes

    In order to define the discrete bilinear form, we need to introduce
    the jump and average operators [\\(\\jump{u}:= u\^+ - u\^-\\)]{.math
    .notranslate .nohighlight} and [\\(\\av{u}:= \\frac{u\^- +
    u\^+}{2}\\)]{.math .notranslate .nohighlight}. The SIPG bilinear
    form is defined as

    ::: {.math .notranslate .nohighlight}
    \\\[\\begin{split}\\begin{align} \\begin{aligned} \\ipbf{u}{v} :=&
    \\int\_\\mesh \\ddx{u} \\ddx{v} dx + \\frac{1}{\\eps} \\int\_\\mesh
    u v dx \\\\ &+ \\int\_\\meshfaces \\left( \\jump{u}
    \\avv{\\ddx{v}} + \\avv{\\ddx{u}} \\jump{v} \\right) ds +
    \\int\_\\meshfaces \\delta \\jump{u} \\jump{v} ds, \\end{aligned}
    \\end{align}\\end{split}\\\]
    :::

    where the boundary conditions have been imposed weakly (i.e. Nitsche
    boundary conditions) and [\\(\\delta \\in \\mathbb{R}\\)]{.math
    .notranslate .nohighlight} is a parameter penalizing the
    discontinuities at the nodes. In order for the discrete bilinear
    form to be coercive, we must choose [\\(\\delta =
    \\delta\_0/h\\)]{.math .notranslate .nohighlight}, where
    [\\(h\\)]{.math .notranslate .nohighlight} is the diameter of the
    cells and [\\(\\delta\_0 \\in \[1,\\infty)\\)]{.math .notranslate
    .nohighlight}. Coercivity and continuity are proven for the
    Laplacian under the assumption that [\\(\\delta\_0\\)]{.math
    .notranslate .nohighlight} is sufficiently large, these estimates
    are still valid under the addition of a reaction term, since such a
    term is positive definite.

    `boundary`{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[¶](#pymgdg.ReactionDiffusion.boundary "Permalink to this definition"){.headerlink}

    :   Boundary integration

    `cell`{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[¶](#pymgdg.ReactionDiffusion.cell "Permalink to this definition"){.headerlink}

    :   Cell integration

    `face`{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[¶](#pymgdg.ReactionDiffusion.face "Permalink to this definition"){.headerlink}

    :   Face integration
:::

[]{#module-lobatto .target}

::: {#lobatto-py .section}
lobatto.py[¶](#lobatto-py "Permalink to this headline"){.headerlink}
--------------------------------------------------------------------

Lobatto quadrature.

`lobatto.`{.sig-prename .descclassname}`lobatto_compute`{.sig-name .descname}[(]{.sig-paren}*order*[)]{.sig-paren}[¶](#lobatto.lobatto_compute "Permalink to this definition"){.headerlink}

:   Compute the lobatto quadrature.

    Parameters

    :   **order** (*int*) -- Order of the quadrature
:::

::: {.toctree-wrapper .compound}
:::
:::

::: {#indices-and-tables .section}
Indices and tables[¶](#indices-and-tables "Permalink to this headline"){.headerlink}
====================================================================================

-   [[Index]{.std .std-ref}](genindex.html){.reference .internal}

-   [[Module Index]{.std .std-ref}](py-modindex.html){.reference
    .internal}

-   [[Search Page]{.std .std-ref}](search.html){.reference .internal}
:::
:::
:::
:::

::: {.sphinxsidebar role="navigation" aria-label="main navigation"}
::: {.sphinxsidebarwrapper}
[Multigrid Discontinuous Galerkin Reaction Diffusion solver](#) {#multigrid-discontinuous-galerkin-reaction-diffusion-solver-1 .logo}
===============================================================

### Navigation

::: {.relations}
### Related Topics

-   [Documentation overview](#)
:::

::: {#searchbox style="display: none" role="search"}
### Quick search {#searchlabel}

::: {.searchformwrapper}
:::
:::
:::
:::

::: {.clearer}
:::
:::

::: {.footer}
©2019, Pablo Lucero. \| Powered by [Sphinx
3.0.0+/911c2ef](http://sphinx-doc.org/) & [Alabaster
0.7.12](https://github.com/bitprophet/alabaster) \| [Page
source](_sources/index.rst.txt)
:::
