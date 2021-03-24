r"""
pymgdg.py
====================================
Solution of a symmetric interior penalty discontinuous Galerkin (SIPG) discretized,
singularly perturbed reaction-diffusion equation in 1D using lagrange polynomial elements.
"""
#import gmres
import copy as cp
import math
import numpy as np
import sympy as sy
import scipy as sc
import sys
from scipy.optimize import minimize,fmin
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as mp
np.set_printoptions(precision=8,threshold=sys.maxsize,linewidth=np.inf,suppress=True)
import lobatto as lo

class LagrangeBasis:
    r"""
    Lagrange-type basis class, containing the same lagrange shape and test functions if
    order 'p'.

    Parameters
    ----------
    p : int
        Order of the Lagrange polynomial used as a shape and test functions. 
    """
    def __init__(self,order_):
        self.p = order_
        self.x,self.h = sy.symbols('x h')
        [self.xtab,self.weights] = lo.lobatto_compute(self.p + 1)
        self.Fx = sy.ones(1,self.p + 1)
        for i in range(self.p + 1):
            for j in range(self.p + 1):
                if (i != j):
                    self.Fx[i] *= (self.x - self.xtab[j] * self.h) / \
                                  (self.xtab[i] * self.h - self.xtab[j] * self.h)

class ReactionDiffusion:
    r"""
    Class containing specific data about the problem to be solved, in this case
    corresponding to the bilinear form of a DG Reaction Diffusion differential equation.

    Parameters 
    ---------- 
    bs : Object containing the basis functions to be used and the order of the polynomial \
    degree desired.

    .. math::
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

    Notes
    -----

    In order to define the discrete bilinear form, we need to introduce the jump and
    average operators :math:`\jump{u}:= u^+ - u^-` and :math:`\av{u}:= \frac{u^- +
    u^+}{2}`.  The SIPG bilinear form is defined as

    .. math::
        \begin{align} 
        \begin{aligned}
        \ipbf{u}{v} :=& \int_\mesh \ddx{u} \ddx{v} dx + \frac{1}{\eps} \int_\mesh u v
        dx \\ 
        &+ \int_\meshfaces \left( \jump{u} \avv{\ddx{v}} + \avv{\ddx{u}}
        \jump{v} \right) ds + \int_\meshfaces \delta \jump{u} \jump{v} ds,
        \end{aligned} 
        \end{align} 

    where the boundary conditions have been imposed weakly (i.e. Nitsche boundary
    conditions) and :math:`\delta \in \mathbb{R}` is a parameter penalizing the
    discontinuities at the nodes. In order for the discrete bilinear form to be coercive,
    we must choose :math:`\delta = \delta_0/h`, where :math:`h` is the diameter of the
    cells and :math:`\delta_0 \in [1,\infty)`. Coercivity and continuity are proven for
    the Laplacian under the assumption that :math:`\delta_0` is sufficiently large, these
    estimates are still valid under the addition of a reaction term, since such a term is
    positive definite.
    """
    def __init__(self,basis_):
        self.bs = basis_
        self.e,self.d = sy.symbols('e d')
        
    def cell(self):
        r"""
        Cell integration
        """
        CM = sy.Matrix(sy.zeros((self.bs.p + 1),(self.bs.p + 1)))
        CM00 = sy.Matrix(sy.zeros((self.bs.p + 1),(self.bs.p + 1)))
        for i in range((self.bs.p + 1)):
            for j in range((self.bs.p + 1)):
                fa=self.bs.Fx[i]
                fb=self.bs.Fx[j]
            # + 1/self.e * sy.integrate(fa*fb,(self.bs.x,0,self.bs.h)) \
                CM[i,j] = sy.integrate(fa.diff(self.bs.x)*fb.diff(self.bs.x),(self.bs.x,0,self.bs.h)) \
                    - (0                    - fa.subs(self.bs.x,0)        )     * (0                    + fb.diff(self.bs.x).subs(self.bs.x,0)) / 2 \
                    - (0                    + fa.diff(self.bs.x).subs(self.bs.x,0)) / 2 * (0                    - fb.subs(self.bs.x,0)        )     \
                    - (fa.subs(self.bs.x,self.bs.h)         - 0                   )     * (fb.diff(self.bs.x).subs(self.bs.x,self.bs.h) + 0                   ) / 2 \
                    - (fa.diff(self.bs.x).subs(self.bs.x,self.bs.h) + 0                   ) / 2 * (fb.subs(self.bs.x,self.bs.h)         - 0                   )     \
                    + self.d / self.bs.h * (0            - fa.subs(self.bs.x,0)) * (0            - fb.subs(self.bs.x,0))                                 \
                    + self.d / self.bs.h * (fa.subs(self.bs.x,self.bs.h) - 0           ) * (fb.subs(self.bs.x,self.bs.h) - 0           )
                CM00[i,j] = sy.integrate(fa.diff(self.bs.x)*fb.diff(self.bs.x),(self.bs.x,0,self.bs.h)) \
                    - 2*(0                    - fa.subs(self.bs.x,0)        )     * (0                    + fb.diff(self.bs.x).subs(self.bs.x,0)) / 2 \
                    - 2*(0                    + fa.diff(self.bs.x).subs(self.bs.x,0)) / 2 * (0                    - fb.subs(self.bs.x,0)        )     \
                    - 2*(fa.subs(self.bs.x,self.bs.h)         - 0                   )     * (fb.diff(self.bs.x).subs(self.bs.x,self.bs.h) + 0                   ) / 2 \
                    - 2*(fa.diff(self.bs.x).subs(self.bs.x,self.bs.h) + 0                   ) / 2 * (fb.subs(self.bs.x,self.bs.h)         - 0                   )     \
                    + 2*self.d / self.bs.h * (0            - fa.subs(self.bs.x,0)) * (0            - fb.subs(self.bs.x,0))                                 \
                    + 2*self.d / self.bs.h * (fa.subs(self.bs.x,self.bs.h) - 0           ) * (fb.subs(self.bs.x,self.bs.h) - 0           )
        return [CM,CM00]
    
    def boundary(self):
        r"""
        Boundary integration
        """
        CM0 = sy.Matrix(sy.zeros((self.bs.p + 1),(self.bs.p + 1)))
        CM1 = sy.Matrix(sy.zeros((self.bs.p + 1),(self.bs.p + 1)))
        for i in range((self.bs.p + 1)):
            for j in range((self.bs.p + 1)):
                fa=self.bs.Fx[i]
                fb=self.bs.Fx[j]
#                           + 1/self.e * sy.integrate(fa*fb,(self.bs.x,0,self.bs.h)) \
                CM0[i,j] = sy.integrate(fa.diff(self.bs.x)*fb.diff(self.bs.x),(self.bs.x,0,self.bs.h)) \
                           - (0                    - fa.subs(self.bs.x,0)        )     * (0                    + fb.diff(self.bs.x).subs(self.bs.x,0)) / 2 \
                           - (0                    + fa.diff(self.bs.x).subs(self.bs.x,0)) / 2 * (0                    - fb.subs(self.bs.x,0)        )     \
                           - (fa.subs(self.bs.x,self.bs.h)         - 0                   )     * (fb.diff(self.bs.x).subs(self.bs.x,self.bs.h) + 0                   ) / 2 \
                           - (fa.diff(self.bs.x).subs(self.bs.x,self.bs.h) + 0                   ) / 2 * (fb.subs(self.bs.x,self.bs.h)         - 0                   )     \
                           + self.d / self.bs.h * (0            - fa.subs(self.bs.x,0)) * (0            - fb.subs(self.bs.x,0))                                 \
                           + self.d / self.bs.h * (fa.subs(self.bs.x,self.bs.h) - 0           ) * (fb.subs(self.bs.x,self.bs.h) - 0           )
                CM1[i,j] = CM0[i,j]
                if ((i==0)or(j==0)):
                    CM0[i,j] += - (0                    - fa.subs(self.bs.x,0)        )     * (0                    + fb.diff(self.bs.x).subs(self.bs.x,0)) / 2 \
                                - (0                    + fa.diff(self.bs.x).subs(self.bs.x,0)) / 2 * (0                    - fb.subs(self.bs.x,0)        )     \
                                + self.d / self.bs.h * (0            - fa.subs(self.bs.x,0)) * (0            - fb.subs(self.bs.x,0))                                 
                if ((i==self.bs.p)or(j==self.bs.p)):
                    CM1[i,j] += - (fa.subs(self.bs.x,self.bs.h)         - 0                   )     * (fb.diff(self.bs.x).subs(self.bs.x,self.bs.h) + 0                   ) / 2 \
                                - (fa.diff(self.bs.x).subs(self.bs.x,self.bs.h) + 0                   ) / 2 * (fb.subs(self.bs.x,self.bs.h)         - 0                   )     \
                                + self.d / self.bs.h * (fa.subs(self.bs.x,self.bs.h) - 0           ) * (fb.subs(self.bs.x,self.bs.h) - 0           )

        return [CM0,CM1]
    
    def face(self):
        r"""
        Face integration
        """
        FM = sy.Matrix(sy.zeros((self.bs.p + 1),(self.bs.p + 1)))
        for i in range((self.bs.p + 1)):
            for j in range((self.bs.p + 1)):
                fa=self.bs.Fx[i]
                fb=self.bs.Fx[j]
                FM[i,j] = - (fa.subs(self.bs.x,self.bs.h)         - 0)     * (0 + fb.diff(self.bs.x).subs(self.bs.x,0)) / 2 \
                          - (fa.diff(self.bs.x).subs(self.bs.x,self.bs.h) + 0) / 2 * (0 - fb.subs(self.bs.x,0)        )     \
                          + self.d / self.bs.h * (fa.subs(self.bs.x,self.bs.h) - 0) * (0 - fb.subs(self.bs.x,0))
        return FM

class DiscreteOperator:
    r"""
    Class implementing the discrete operator corresponding to a problem.

    Parameters
    ----------

    problem : 
        Object containing the integrals corresponding to cell, boundary and faces.

    n : int
        Total cells in the mesh.

    periodic : bool
        True: Periodic boundary conditions, False: Dirichlet boundary conditions.
    """
    def __init__(self,problem_,n_,periodic_=False):
        self.pb = problem_
        self.bs = problem_.bs
        [self.CM,self.CM00] = problem_.cell()
        [self.CM0,self.CM1] = problem_.boundary()
        self.FM = problem_.face()
        self.n = n_
        self.periodic = periodic_
        self.A = sy.Matrix(sy.zeros(1,1))
        self.An = np.zeros((1,1))
        self.Dc = np.zeros((1,1))
        self.Das = np.zeros((1,1))
        self.Dras = np.zeros((1,1))
        self.Dp = np.zeros((1,1))
        
    def assemble(self):
        r"""
        Assembles the symbolic matrix.
        """
        self.A = sy.Matrix(sy.zeros((self.bs.p + 1)*self.n,(self.bs.p + 1)*self.n))
        if (self.n==1):
            self.A = self.CM00
        else:
            for b in range(self.n):
                for i in range((self.bs.p + 1)):
                    for j in range((self.bs.p + 1)):
                        if (b==0):
                            if (self.periodic):
                                self.A[(self.bs.p + 1)*b+i,(self.bs.p + 1)*b+j] = self.CM[i,j]
                                self.A[(self.bs.p + 1)*b+i,(self.bs.p + 1)*(self.n-1)+j] = self.FM.transpose()[i,j]
                            else:
                                self.A[(self.bs.p + 1)*b+i,(self.bs.p + 1)*b+j] = self.CM0[i,j]
                            self.A[(self.bs.p + 1)*b+i,(self.bs.p + 1)*b+j+(self.bs.p + 1)] = self.FM[i,j]
                        elif (b==self.n - 1):
                            if (self.periodic):
                                self.A[(self.bs.p + 1)*b+i,(self.bs.p + 1)*b+j] = self.CM[i,j]
                                self.A[(self.bs.p + 1)*b+i,(self.bs.p + 1)*0+j] = self.FM[i,j]
                            else:
                                self.A[(self.bs.p + 1)*b+i,(self.bs.p + 1)*b+j] = self.CM1[i,j]
                            self.A[(self.bs.p + 1)*b+i,(self.bs.p + 1)*b+j-(self.bs.p + 1)] = self.FM.transpose()[i,j]
                        else:
                            self.A[(self.bs.p + 1)*b+i,(self.bs.p + 1)*b+j] = self.CM[i,j]
                            self.A[(self.bs.p + 1)*b+i,(self.bs.p + 1)*b+j+(self.bs.p + 1)] = self.FM[i,j]
                            self.A[(self.bs.p + 1)*b+i,(self.bs.p + 1)*b+j-(self.bs.p + 1)] = self.FM.transpose()[i,j]

    def nassemble(self,par):
        r"""
        Assembles the numerical matrix.

        Parameters
        ----------
        par : {}
            Values for all the symbols in the symbolic matrix.
        """
        self.An = np.zeros(((self.bs.p + 1)*self.n,(self.bs.p + 1)*self.n))
        CM = np.matrix(self.CM.subs(par)).astype(float)
        CM00 = np.matrix(self.CM00.subs(par)).astype(float)
        CM0 = np.matrix(self.CM0.subs(par)).astype(float)
        CM1 = np.matrix(self.CM1.subs(par)).astype(float)
        FM = np.matrix(self.FM.subs(par)).astype(float)
        if (self.n==1):
            self.An = CM00
        else:
            for b in range(self.n):
                for i in range((self.bs.p + 1)):
                    for j in range((self.bs.p + 1)):
                        if (b==0):
                            if (self.periodic):
                                self.An[(self.bs.p + 1)*b+i,(self.bs.p + 1)*b+j] = CM[i,j]
                                self.An[(self.bs.p + 1)*b+i,(self.bs.p + 1)*(self.n-1)+j] = FM.transpose()[i,j]
                            else:
                                self.An[(self.bs.p + 1)*b+i,(self.bs.p + 1)*b+j] = CM0[i,j]
                            self.An[(self.bs.p + 1)*b+i,(self.bs.p + 1)*b+j+(self.bs.p + 1)] = FM[i,j]
                        elif (b==self.n - 1):
                            if (self.periodic):
                                self.An[(self.bs.p + 1)*b+i,(self.bs.p + 1)*b+j] = CM[i,j]
                                self.An[(self.bs.p + 1)*b+i,(self.bs.p + 1)*0+j] = FM[i,j]
                            else:
                                self.An[(self.bs.p + 1)*b+i,(self.bs.p + 1)*b+j] = CM1[i,j]
                            self.An[(self.bs.p + 1)*b+i,(self.bs.p + 1)*b+j-(self.bs.p + 1)] = FM.transpose()[i,j]
                        else:
                            self.An[(self.bs.p + 1)*b+i,(self.bs.p + 1)*b+j] = CM[i,j]
                            self.An[(self.bs.p + 1)*b+i,(self.bs.p + 1)*b+j+(self.bs.p + 1)] = FM[i,j]
                            self.An[(self.bs.p + 1)*b+i,(self.bs.p + 1)*b+j-(self.bs.p + 1)] = FM.transpose()[i,j]

    def assemble_cellBJ(self):
        r"""
        Assembles the _cell_ Block Jacobi matrix.
        """
        self.Dc = np.zeros(((self.bs.p + 1)*self.n,(self.bs.p + 1)*self.n))
        for b in range(self.n):
            self.Dc[(self.bs.p + 1)*b:(self.bs.p + 1)*(b+1),\
                    (self.bs.p + 1)*b:(self.bs.p + 1)*(b+1)] = np.linalg.inv(self.An[(self.bs.p + 1)*b:(self.bs.p + 1)*(b+1),\
                                                                                     (self.bs.p + 1)*b:(self.bs.p + 1)*(b+1)])
    def assemble_as(self):
        r"""
        Assembles the _overlapping_ Block Jacobi matrix.
        """
        self.Das = np.zeros(((self.bs.p + 1)*self.n,(self.bs.p + 1)*self.n))
        
        Ainv = np.linalg.inv(self.An[(self.bs.p + 1)*1:(self.bs.p + 1)*3,
                                     (self.bs.p + 1)*1:(self.bs.p + 1)*3])

        self.Das[(self.bs.p + 1)*0:(self.bs.p + 1)*1,
                  (self.bs.p + 1)*0:(self.bs.p + 1)*1]+=Ainv[(self.bs.p + 1)*1:(self.bs.p + 1)*2,
                                                             (self.bs.p + 1)*1:(self.bs.p + 1)*2]

        self.Das[(self.bs.p + 1)*0:(self.bs.p + 1)*1,
                  (self.bs.p + 1)*(self.n-1):(self.bs.p + 1)*((self.n-1)+2)]+=Ainv[(self.bs.p + 1)*1:(self.bs.p + 1)*2,
                                                                                   (self.bs.p + 1)*0:(self.bs.p + 1)*1]
        
        for b in range(self.n-1):
            self.Das[(self.bs.p + 1)*b:(self.bs.p + 1)*(b+2),
                      (self.bs.p + 1)*b:(self.bs.p + 1)*(b+2)]+=Ainv[(self.bs.p + 1)*0:(self.bs.p + 1)*2,
                                                                     (self.bs.p + 1)*0:(self.bs.p + 1)*2]

        self.Das[(self.bs.p + 1)*(self.n-1):(self.bs.p + 1)*((self.n-1)+2),
                  (self.bs.p + 1)*(self.n-1):(self.bs.p + 1)*((self.n-1)+2)]+=Ainv[(self.bs.p + 1)*0:(self.bs.p + 1)*1,
                                                                                   (self.bs.p + 1)*0:(self.bs.p + 1)*1]

        self.Das[(self.bs.p + 1)*(self.n-1):(self.bs.p + 1)*((self.n-1)+2),
                  (self.bs.p + 1)*0:(self.bs.p + 1)*1]+=Ainv[(self.bs.p + 1)*0:(self.bs.p + 1)*1,
                                                             (self.bs.p + 1)*1:(self.bs.p + 1)*2]

    def assemble_ras(self):
        r"""
        Assembles the _overlapping_ Block Jacobi matrix.
        """
        self.Dras = np.zeros(((self.bs.p + 1)*self.n,(self.bs.p + 1)*self.n))
        
        Ainv = np.linalg.inv(self.An[(self.bs.p + 1)*1:(self.bs.p + 1)*4,
                                     (self.bs.p + 1)*1:(self.bs.p + 1)*4])
        
        self.Dras[(self.bs.p + 1)*0:(self.bs.p + 1)*1,
                  (self.bs.p + 1)*0:(self.bs.p + 1)*2]=Ainv[(self.bs.p + 1)*1:(self.bs.p + 1)*2,
                                                            (self.bs.p + 1)*1:(self.bs.p + 1)*3]

        self.Dras[(self.bs.p + 1)*0:(self.bs.p + 1)*1,
                  (self.bs.p + 1)*(self.n-1):(self.bs.p + 1)*((self.n-1)+2)]=Ainv[(self.bs.p + 1)*1:(self.bs.p + 1)*2,
                                                                                  (self.bs.p + 1)*0:(self.bs.p + 1)*1]

        for b in range(1,self.n-1):
            self.Dras[(self.bs.p + 1)*b:(self.bs.p + 1)*(b+1),
                      (self.bs.p + 1)*(b-1):(self.bs.p + 1)*(b+2)]=Ainv[(self.bs.p + 1)*1:(self.bs.p + 1)*2,
                                                                        (self.bs.p + 1)*0:(self.bs.p + 1)*3]

        self.Dras[(self.bs.p + 1)*(self.n-1):(self.bs.p + 1)*((self.n-1)+1),
                  (self.bs.p + 1)*((self.n-1)-1):(self.bs.p + 1)*((self.n-1)+1)]=Ainv[(self.bs.p + 1)*1:(self.bs.p + 1)*2,
                                                                                    (self.bs.p + 1)*0:(self.bs.p + 1)*2]

        self.Dras[(self.bs.p + 1)*(self.n-1):(self.bs.p + 1)*((self.n-1)+1),
                  (self.bs.p + 1)*0:(self.bs.p + 1)*1]=Ainv[(self.bs.p + 1)*1:(self.bs.p + 1)*2,
                                                            (self.bs.p + 1)*2:(self.bs.p + 1)*3]
        
    def assemble_pointBJ(self):
        r"""
        Assembles the _point_ Block Jacobi matrix.
        """
        s = int((self.bs.p + 1)/2)
        self.Dp = np.zeros(((self.bs.p + 1)*self.n,(self.bs.p + 1)*self.n))
        self.Dp[0:s,0:s] = np.linalg.inv(self.An[0:s,0:s])
        self.Dp[(self.bs.p + 1)*self.n-s:(self.bs.p + 1)*self.n,\
                (self.bs.p + 1)*self.n-s:(self.bs.p + 1)*self.n] = np.linalg.inv(self.An[(self.bs.p + 1)*self.n-s:(self.bs.p + 1)*self.n,\
                                                                                         (self.bs.p + 1)*self.n-s:(self.bs.p + 1)*self.n])
        for b in range(1,self.n):
            self.Dp[(self.bs.p + 1)*b-s:(self.bs.p + 1)*(b+1)-s,\
                    (self.bs.p + 1)*b-s:(self.bs.p + 1)*(b+1)-s] = np.linalg.inv(self.An[(self.bs.p + 1)*b-s:(self.bs.p + 1)*(b+1)-s,\
                                                                                         (self.bs.p + 1)*b-s:(self.bs.p + 1)*(b+1)-s])

class CoarseCorrection:
    r"""
    Coarse correction object.

    Parameters
    ----------
    dc: 
        Discrete operator
    """
    def __init__(self,discreteOperator_,c_):
        self.dc = discreteOperator_
        self.c = c_
        self.An = self.dc.An
        self.bs = self.dc.bs
        self.n = self.dc.n
        self.RT = np.zeros(((self.bs.p + 1)*self.n,(self.bs.p + 1)*int(self.n/2)))
        self.R = self.RT.transpose()
        self.RTs = sy.zeros(2*self.n,self.n)
        self.Rs = self.RTs.transpose()
        self.A0 = np.zeros((1,1))
        self.A0inv = np.zeros((1,1))
        
    def assemble(self):
        r"""
        Assemble coarse correction matrix
        """
        self.RT = sy.zeros(2*self.n,self.n)

        BT = sy.Matrix([
            [        sy.S(1),        sy.S(0)],
            [sy.S(1)/sy.S(2),sy.S(1)/sy.S(2)],
            [sy.S(1)/sy.S(2),sy.S(1)/sy.S(2)],
            [        sy.S(0),        sy.S(1)]])
        
        for b in range(int(self.n/2)):
            self.RTs[4*b:4*b+4,2*b:2*b+2] = BT

        B = sy.transpose(sy.Matrix([
            [        sy.S(2)             ,        sy.S(0)             ],
            [sy.S(1)-sy.S(2)*self.dc.pb.d,        sy.S(1)             ],
            [        sy.S(1)             ,sy.S(1)-sy.S(2)*self.dc.pb.d],
            [        sy.S(0)             ,        sy.S(2)             ]]))
        for b in range(int(self.n/2)):
            self.Rs[2*b:2*b+2,4*b:4*b+4] = B

    def nassemble(self):
        r"""
        Assemble coarse correction matrix
        """
        self.RT = np.zeros(((self.bs.p + 1)*self.n,(self.bs.p + 1)*int(self.n/2)))
        c = self.c
        for k in range(0,int(((self.bs.p + 1)*self.n)/2),self.bs.p + 1):
            avg = False 
            j = k
            for i in range(2*k,2*k+(self.bs.p + 1),1):
                if avg:
                    self.RT[i,j] = c
                    self.RT[i,j+1] = 1-c
                    j += 1
                    avg = False
                else:
                    self.RT[i,j] = 1.
                    avg = True
            if avg:
                avg = False
            else:
                avg = True
                j -= 1
            for i in range(2*k+(self.bs.p + 1),2*k+2*(self.bs.p + 1),1):
                if avg:
                    self.RT[i,j] = 1-c
                    self.RT[i,j+1] = c
                    j += 1
                    avg = False
                else:
                    self.RT[i,j] = 1.
                    avg = True
        self.R = np.transpose(self.RT)
        self.RT = self.RT
        self.A0 = self.R.dot(self.An.dot(self.RT))
        self.A0inv = np.linalg.pinv(self.A0)

def plot(dc):
    r"""
    Plotting function for a simple problem.

    Parameters
    ----------
    dc: 
        Discrete operator
    """
    p=dc.bs.p
    n=dc.n
    h=1./dc.n
    dc.nassemble({dc.bs.h:h,dc.pb.d:float(p*(p+1)),dc.pb.e:np.infty})
    g = np.zeros(((p + 1)*n,1))
    f = np.zeros(((p + 1)*n,1))
    [xtab,weights] = lo.lobatto_compute(p+1)
    for b in range(n):
        for i in range(p + 1):
            g[(p + 1)*b + i] = (b + xtab[i]) * h
    
    [xtab2,weights2] = lo.lobatto_compute(p+3)
    for b in range(n):
        for i in range(p + 1):
            for j in range(len(xtab2)):
                f[(p + 1)*b + i] += float(weights2[j]*h*\
                                          (dc.bs.Fx[i]*4.*
                                           sy.pi*sy.pi*sy.sin(2.*sy.pi*(dc.bs.x+b*h)))\
                                          .subs({dc.bs.x:xtab2[j]*h}).subs({dc.bs.h:h}))
    
    mp.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75')
    mp.grid(which='minor', axis='x', linewidth=0.25, linestyle='-', color='0.75')
    mp.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75')
    mp.grid(which='minor', axis='y', linewidth=0.25, linestyle='-', color='0.75')
    mp.xlabel("x")
    mp.ylabel("u(x)")
    mp.plot(g,np.linalg.inv(dc.An).dot(f))
    mp.show()

# dc = DiscreteOperator(problem_=ReactionDiffusion(basis_=LagrangeBasis(order_=1)),
#                       n_=64,
#                       periodic_=False)


def MG(g,s,d,rlx):
    n = int(g.shape[0]/2)
    dc = DiscreteOperator(problem_=ReactionDiffusion(basis_=LagrangeBasis(order_=1)),
                          n_=n,
                          periodic_=False)
    dc.assemble()
    dc.nassemble({dc.bs.h:1./dc.n,dc.pb.d:d})
    dc.assemble_cellBJ()
    Dinv = dc.Dc
    cc = CoarseCorrection(discreteOperator_=dc,c_=0.5)
    cc.assemble()
    #cc.nassemble()
    A = np.matrix(dc.A.subs({dc.bs.h:1./dc.n,dc.pb.d:d}),dtype=np.float64)
    RT = np.matrix(cc.RTs.subs({dc.bs.h:1./dc.n,dc.pb.d:d}),dtype=np.float64)
    R = np.transpose(RT)
    RR = np.matrix(cc.Rs.subs({dc.bs.h:1./dc.n,dc.pb.d:d}),dtype=np.float64)
    # RT = cc.RT
    # R = cc.R
    x = 0*g

    for i in range(s):
        x = x + rlx*Dinv.dot(g - A.dot(x))

    if (n > 2):
        x = x + RT.dot(MG(R.dot(g - A.dot(x)),s,d,rlx))
    else:
        dc0 = DiscreteOperator(problem_=ReactionDiffusion(basis_=LagrangeBasis(order_=1)),
                               n_=int(n/2),
                               periodic_=False)
        dc0.assemble()
        A0 = np.matrix((dc0.A).subs({dc0.bs.h:1./dc0.n,dc0.pb.d:d}),dtype=np.float64)
        # np.matrix(R.dot(A).dot(RT)/2.) # Two-level coarse space
        # sy.pprint(sy.simplify(cc.Rs*dc.A*cc.RTs-dc0.A)) # Test of R for inherited Galerkin A0
        A0inv = np.linalg.inv(A0)
        x = x + RT.dot(A0inv.dot(R.dot(g - A.dot(x))))
        
    for i in range(s):
        x = x + rlx*Dinv.dot(g - A.dot(x))

        
    return x


def func(d):
    r"""
    Fmin to obtain the optimal relaxation parameter that delivers the lowest spectral radius.
    
    Parameters
    ----------
    dd: 
    DG method penalty parameter.
    """
    i = 4
    n = 2**i
    # dc = DiscreteOperator(problem_=ReactionDiffusion(basis_=LagrangeBasis(order_=1)),
    #                       n_=n,
    #                       periodic_=False)
    # dc.assemble()
    # A = np.matrix(dc.A.subs({dc.bs.h:1./dc.n,dc.pb.d:d}),dtype=np.float64)
    # def func2(rlx):
    #     E = np.eye(2*n)-MG(A,1,d,rlx[0])
    #     return sorted(abs(np.real(np.linalg.eigvals(E))),reverse=True)[0]
    # xmin,ffmin,dum1,dum2,dum3 = fmin(func2,np.array([1.]),ftol=0.000001,xtol=0.000001,full_output=True,disp=False)
    # print(xmin[0],end=" ")
    # print(ffmin)

    dc = DiscreteOperator(problem_=ReactionDiffusion(basis_=LagrangeBasis(order_=1)),
                          n_=n,
                          periodic_=False)
    dc.assemble()
    def func2(x):
        A = np.matrix(dc.A.subs({dc.bs.h:1./dc.n,dc.pb.d:d}),dtype=np.float64)
        E = np.eye(2*n)-MG(A,1,x[0],x[1])
        return sorted(abs(np.real(np.linalg.eigvals(E))),reverse=True)[0]
    print(minimize(func2,np.array([2.,0.7]),tol=0.01,method='Nelder-Mead'))

#func(2.)
    
n = 16
d = 2.
rlx = np.sqrt(2.)/2. #2*d**2 / (2*d**2+d-1)

dc = DiscreteOperator(problem_=ReactionDiffusion(basis_=LagrangeBasis(order_=1)),
                      n_=n,
                      periodic_=False)
dc.assemble()
A = np.matrix(dc.A.subs({dc.bs.h:1./dc.n,dc.pb.d:d}),dtype=np.float64)

# E = np.eye(2*n)-MG(A,1,d,rlx)
# print(sorted(abs(np.real(np.linalg.eigvals(E))),reverse=True)[0])

rhs = np.ones(((dc.bs.p + 1)*dc.n,1))/float((dc.bs.p + 1)*dc.n)
x = np.zeros(((dc.bs.p + 1)*dc.n,1))
norm0 = np.linalg.norm(np.ones((dc.bs.p + 1)*dc.n)/float((dc.bs.p + 1)*dc.n))
normr = 1.
it = 0
while (normr/norm0 > 1.E-8):
    r = rhs - A.dot(x)
    x = x + MG(r,1,d,rlx)
    normr = np.linalg.norm(r)
    print(it,normr)
    it = it + 1
    if (it > 100):break

    
# rlx = 1.
# rlx = 2*d/(2+3*d)
# rlx = 2*d**2 / (2*d**2+d-1)
# for i in range(1,10):
#     d = 1.+0.1*i
#     print(d,end=" ")
#     func(d)


# dc = DiscreteOperator(problem_=ReactionDiffusion(basis_=LagrangeBasis(order_=1)),
#                       n_=4,
#                       periodic_=False)
# dc.assemble()
# cc = CoarseCorrection(discreteOperator_=dc,c_=0.5)
# cc.assemble()
# A4 = cp.deepcopy(dc.A)
# sy.pprint(A4[0:16,0:16])
# dc = DiscreteOperator(problem_=ReactionDiffusion(basis_=LagrangeBasis(order_=1)),
#                       n_=8,
#                       periodic_=False)
# dc.assemble()
# cc = CoarseCorrection(discreteOperator_=dc,c_=0.5)
# cc.assemble()
# A40 = sy.simplify(cc.Rs*dc.A*cc.RTs)
# sy.pprint(A40[0:16,0:16])
# sy.pprint(sy.simplify(cc.Rs*cc.RTs*sy.ones(cc.Rs.shape[0],1)))
    

# d = 3.
# rlx = 0.98
# i = 4
# n = 2**i
# dc = DiscreteOperator(problem_=ReactionDiffusion(basis_=LagrangeBasis(order_=1)),
#                       n_=n,
#                       periodic_=False)
# dc.assemble()
# A = np.matrix(dc.A.subs({dc.bs.h:1./dc.n,dc.pb.d:d}),dtype=np.float64)
# E = np.eye(2*n)-MG(A,1,d,rlx)
# print(sorted(abs(np.real(np.linalg.eigvals(E))),reverse=True)[0])

# for i in range(1,7):
#     n = 2**i
#     dc = DiscreteOperator(problem_=ReactionDiffusion(basis_=LagrangeBasis(order_=1)),
#                           n_=n,
#                           periodic_=False)
#     dc.assemble()
#     A = np.matrix(dc.A.subs({dc.bs.h:1./dc.n,dc.pb.d:d}),dtype=np.float64)
#     E = np.eye(2*n)-MG(A,1)
#     print(sorted(abs(np.real(np.linalg.eigvals(E))),reverse=True)[0])
        
# for i in range(0,51):
#     func(1.001+((6.-1.001)*(2.**(float(i)/50.) - 1.)))

# c = 0.5
# d = 2
# a = 2*d**2/(2*d**2+d-1)
# dc.nassemble({dc.bs.h:1./dc.n,dc.pb.d:d,dc.pb.e:np.infty})
# dc.assemble_cellBJ()
# cc = CoarseCorrection(discreteOperator_=dc,c_=c)
# cc.nassemble()
# Id = np.eye((dc.bs.p + 1)*dc.n)
# A0inv = cc.A0inv
# R = cc.R
# RT = cc.RT
# An = dc.An
# Dinv = dc.Dc
# el = 0
# if (dc.periodic): el = 1
# MM = a*Dinv + RT.dot(A0inv).dot(R).dot(Id - a*An.dot(Dinv))
# eigvals,eigvecs = np.linalg.eig(Id-MM.dot(An))
# eigvals = np.real(eigvals)
# eigsort = np.array(sorted(eigvals,reverse=True))
# mp.ylim([-0.7,0.7])
# mp.xlabel("#Eigenvalue")
# mp.ylabel("Eigenvalue")
# mp.plot(eigsort,'g.')

# c = 0.5
# d = 1.5
# a = 2*d**2/(2*d**2+d-1)
# dc.nassemble({dc.bs.h:1./dc.n,dc.pb.d:d,dc.pb.e:np.infty})
# dc.assemble_cellBJ()
# cc = CoarseCorrection(discreteOperator_=dc,c_=c)
# cc.nassemble()
# Id = np.eye((dc.bs.p + 1)*dc.n)
# A0inv = cc.A0inv
# R = cc.R
# RT = cc.RT
# An = dc.An
# Dinv = dc.Dc
# el = 0
# if (dc.periodic): el = 1
# MM = a*Dinv + RT.dot(A0inv).dot(R).dot(Id - a*An.dot(Dinv))
# eigvals,eigvecs = np.linalg.eig(Id-MM.dot(An))
# eigvals = np.real(eigvals)
# eigsort = np.array(sorted(eigvals,reverse=True))
# mp.ylim([-0.7,0.7])
# mp.xlabel("#Eigenvalue")
# mp.ylabel("Eigenvalue")
# mp.plot(eigsort,'b.')

# c = 0.5
# d = 1.5
# a = 2*d**2/(2*d**2+d-1)
# dc.nassemble({dc.bs.h:1./dc.n,dc.pb.d:d,dc.pb.e:np.infty})
# dc.assemble_cellBJ()
# cc = CoarseCorrection(discreteOperator_=dc,c_=c)
# cc.nassemble()
# Id = np.eye((dc.bs.p + 1)*dc.n)
# A0inv = cc.A0inv
# R = cc.R
# RT = cc.RT
# An = dc.An
# Dinv = dc.Dc
# el = 0
# if (dc.periodic): el = 1
# MM = a*Dinv + RT.dot(A0inv).dot(R).dot(Id - a*An.dot(Dinv))
# eigvals,eigvecs = np.linalg.eig(Id-MM.dot(An))
# eigvals = np.real(eigvals)
# eigsort = np.array(sorted(eigvals,reverse=True))
# mp.ylim([-0.7,0.7])
# mp.xlabel("#Eigenvalue")
# mp.ylabel("Eigenvalue")
# mp.plot(eigsort,'b.')

# c = 0.56460427612264228825165455619252
# a = 0.90815413446701560826304190680793
# d = 1.51697830014707997232890254807433
# dc.nassemble({dc.bs.h:1./dc.n,dc.pb.d:d,dc.pb.e:np.infty})
# dc.assemble_cellBJ()
# cc = CoarseCorrection(discreteOperator_=dc,c_=c)
# cc.nassemble()
# Id = np.eye((dc.bs.p + 1)*dc.n)
# A0inv = cc.A0inv
# R = cc.R
# RT = cc.RT
# An = dc.An
# Dinv = dc.Dc
# el = 0
# if (dc.periodic): el = 1
# MM = a*Dinv + RT.dot(A0inv).dot(R).dot(Id - a*An.dot(Dinv))
# eigvals,eigvecs = np.linalg.eig(Id-MM.dot(An))
# eigvals = np.real(eigvals)
# eigsort = np.array(sorted(eigvals,reverse=True))
# mp.plot(eigsort,'r.')
# mp.savefig("1Dspec.pdf")

# def mv(g):
#     x = a*Dinv.dot(g)
#     y = x + RT.dot(A0inv).dot(R).dot(g - An.dot(x))
#     return y

# rhs = np.ones((dc.bs.p + 1)*dc.n)/float((dc.bs.p + 1)*dc.n)

# def callback(g):
#     print(g)
#    print('test')
#    print(np.linalg.norm(rhs - An.dot(g)))

# M = sc.sparse.linalg.LinearOperator(((dc.bs.p + 1)*dc.n,(dc.bs.p + 1)*dc.n),matvec=mv)

# MM = a*Dinv + RT.dot(A0inv).dot(R).dot(Id - a*An.dot(Dinv))

# eigvals,eigvecs = np.linalg.eig(MM.dot(An))
# eigvals = np.real(eigvals)
# eigsort = np.array([eigvals[0]])
# for num in eigvals:
#     counted = False
#     for aux in eigsort:
#         if (abs(aux) < 1.E-10):
#             if (abs(num-aux) < 1.E-8):
#                 counted = True
#                 break
#         elif (abs((num-aux)/aux) < 1.E-8):
#             counted = True
#             break
#     if (not counted):
#         eigsort = np.append(eigsort,num)
#eigsort = np.array(sorted(eigsort,reverse=True))
# eigsort = np.array(sorted(eigvals,reverse=True))
# mp.xlabel("x")
# mp.ylabel("u(x)")
# mp.plot(eigsort)
# mp.show()


# x = np.zeros((dc.bs.p + 1)*dc.n)
# norm0 = np.linalg.norm(np.ones((dc.bs.p + 1)*dc.n)/float((dc.bs.p + 1)*dc.n))
# normr = 1.
# while (normr/norm0 > 1.E-8):
#     r = rhs - An.dot(x)
#     x = x + MM.dot(r)
#     normr = np.linalg.norm(r)
#     print(normr)

# x,e = sc.sparse.linalg.gmres(An,
#                              eigvecs[:,100],
#                              x0=None,
#                              tol=1e-10,
#                              restart=1000,
#                              maxiter=None,
#                              M=MM,
#                              callback=callback,
#                              restrt=None,
#                              atol=1e-20)

# x,e = sc.sparse.linalg.cgs(dc.An,
#                            rhs,
#                            None,
#                            1.E-8,
#                            None,
#                            MM,
#                            callback)

# x,e = sc.sparse.linalg.minres(An,
#                               rhs,
#                               x0=None,
#                               shift=0.0,
#                               tol=1e-5,
#                               maxiter=1E6,
#                               M=MM,
#                               callback=callback,
#                               show=False,
#                               check=False)
