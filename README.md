# Multigrid Discontinuous Galerkin Reaction Diffusion solver

## pymgdg.py

Solution of a symmetric interior penalty discontinuous Galerkin (SIPG) discretized,
singularly perturbed reaction-diffusion equation in 1D, using linear finite elements.


#### class pymgdg.CoarseCorrection(dc)
Coarse correction object.


* **Parameters**

    **dc** – Discrete operator



#### assemble()
Assemble coarse correction matrix


#### class pymgdg.DiscreteOperator(problem, n, periodic_=False)
Class implementing the discrete operator corresponding to a problem.


* **Parameters**

    * **problem** – Object containing the integrals corresponding to cell, boundary and faces.

    * **n** (*int*) – Total cells in the mesh.

    * **periodic** (*bool*) – True: Periodic boundary conditions, False: Dirichlet boundary conditions.



#### assemble()
Assembles the symbolic matrix.


#### assemble_cellBJ()
Assembles the _cell_ Block Jacobi matrix.


#### assemble_pointBJ()
Assembles the _point_ Block Jacobi matrix.


#### nassemble(par)
Assembles the numerical matrix.


* **Parameters**

    **par** (*{}*) – Values for all the symbols in the symbolic matrix.



#### class pymgdg.LagrangeBasis(p)
Lagrange-type basis class, containing the same lagrange shape and test functions if
order ‘p’.


* **Parameters**

    **p** (*int*) – Order of the Lagrange polynomial used as a shape and test functions.



#### class pymgdg.ReactionDiffusion(bs)
Class containing specific data about the problem to be solved, in this case
corresponding to the bilinear form of a DG Reaction Diffusion differential equation.


* **Parameters**

    * **bs** (*Object containing the basis functions to be used and the order of the polynomial *) – 

    * **desired.** (*degree*) – 


$$
\newcommand\scalemath[2]{\scalebox{#1}{\mbox{\ensuremath{\displaystyle #2}}}}
\newcommand{\mesh}{\mathbb{T}}
\newcommand{\cell}{\kappa}
\newcommand{\meshfaces}{\mathbb{F}}
\newcommand{\face}{f}
\newcommand{\ipbf}[2]{a_h\left(#1,#2\right)}
\newcommand{\ddx}[1]{\frac{d #1}{dx}}
\newcommand{\eps}{\varepsilon}
\newcommand{\jump}[1]{\left[\!\left[#1\right]\!\right]}
\newcommand{\av}[1]{\left\{\!\!\left\{#1\right\}\!\!\right\}}
\newcommand{\avv}[1]{\left\{\!\!\!\left\{#1\right\}\!\!\!\right\}}
\newcommand\w[1]{\makebox[2.5em]{$#1$}}
\newcommand{\e}[1]{e^{#1}}
\newcommand{\I}{i}
\newcommand{\phih}{\boldsymbol{\varphi}}
\newcommand{\phiH}{\boldsymbol{\phi}}
\newcommand{\kl}{k}
\newcommand{\kh}{{\widetilde{k}}}
\newcommand{\dd}{\delta_0}

$$

### Notes

In order to define the discrete bilinear form, we need to introduce the jump and
average operators $\jump{u}:= u^+ - u^-$ and $\av{u}:= \frac{u^- +
u^+}{2}$.  The SIPG bilinear form is defined as

$$
\begin{align}
\begin{aligned}
\ipbf{u}{v} :=& \int_\mesh \ddx{u} \ddx{v} dx + \frac{1}{\eps} \int_\mesh u v
dx \\
&+ \int_\meshfaces \left( \jump{u} \avv{\ddx{v}} + \avv{\ddx{u}}
\jump{v} \right) ds + \int_\meshfaces \delta \jump{u} \jump{v} ds,
\end{aligned}
\end{align}

$$

where the boundary conditions have been imposed weakly (i.e. Nitsche boundary
conditions) and $\delta \in \mathbb{R}$ is a parameter penalizing the
discontinuities at the nodes. In order for the discrete bilinear form to be coercive,
we must choose $\delta = \delta_0/h$, where $h$ is the diameter of the
cells and $\delta_0 \in [1,\infty)$. Coercivity and continuity are proven for
the Laplacian under the assumption that $\delta_0$ is sufficiently large, these
estimates are still valid under the addition of a reaction term, since such a term is
positive definite.


#### boundary()
Boundary integration


#### cell()
Cell integration


#### face()
Face integration

## lobatto.py

Lobatto quadrature.


#### lobatto.lobatto_compute(order)
Compute the lobatto quadrature.


* **Parameters**

    **order** (*int*) – Order of the quadrature


# Indices and tables

* Index

* Module Index

* Search Page
