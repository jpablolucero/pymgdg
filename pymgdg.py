r"""
pymgdg.py
====================================
Solution of a symmetric interior penalty discontinuous Galerkin (SIPG) discretized,
singularly perturbed reaction-diffusion equation in 1D using lagrange polynomial elements.
"""

import math
import numpy as np
import sympy as sy
import scipy as sc
import sys
from scipy.optimize import fmin
import matplotlib.pyplot as mp
np.set_printoptions(precision=3,threshold=sys.maxsize,linewidth=np.inf,suppress=True)
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
        for i in range((self.bs.p + 1)):
            for j in range((self.bs.p + 1)):
                fa=self.bs.Fx[i]
                fb=self.bs.Fx[j]
                CM[i,j] = sy.integrate(fa.diff(self.bs.x)*fb.diff(self.bs.x),(self.bs.x,0,self.bs.h)) \
                          + 1/self.e * sy.integrate(fa*fb,(self.bs.x,0,self.bs.h)) \
                          - (0                    - fa.subs(self.bs.x,0)        )     * (0                    + fb.diff(self.bs.x).subs(self.bs.x,0)) / 2 \
                          - (0                    + fa.diff(self.bs.x).subs(self.bs.x,0)) / 2 * (0                    - fb.subs(self.bs.x,0)        )     \
                          - (fa.subs(self.bs.x,self.bs.h)         - 0                   )     * (fb.diff(self.bs.x).subs(self.bs.x,self.bs.h) + 0                   ) / 2 \
                          - (fa.diff(self.bs.x).subs(self.bs.x,self.bs.h) + 0                   ) / 2 * (fb.subs(self.bs.x,self.bs.h)         - 0                   )     \
                          + self.d / self.bs.h * (0            - fa.subs(self.bs.x,0)) * (0            - fb.subs(self.bs.x,0))                                 \
                          + self.d / self.bs.h * (fa.subs(self.bs.x,self.bs.h) - 0           ) * (fb.subs(self.bs.x,self.bs.h) - 0           )
        return CM
    
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
                CM0[i,j] = sy.integrate(fa.diff(self.bs.x)*fb.diff(self.bs.x),(self.bs.x,0,self.bs.h)) \
                           + 1/self.e * sy.integrate(fa*fb,(self.bs.x,0,self.bs.h)) \
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
        self.CM = problem_.cell()
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
        CM0 = np.matrix(self.CM0.subs(par)).astype(float)
        CM1 = np.matrix(self.CM1.subs(par)).astype(float)
        FM = np.matrix(self.FM.subs(par)).astype(float)
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
    def __init__(self,discreteOperator_):
        self.dc = discreteOperator_
        self.An = dc.An
        self.bs = dc.bs
        self.n = dc.n
        self.RT = np.zeros(((self.bs.p + 1)*self.n,(self.bs.p + 1)*int(self.n/2)))
        self.R = self.RT.transpose()
        self.A0 = np.zeros((1,1))
        self.A0inv = np.zeros((1,1))
        
    def assemble(self):
        r"""
        Assemble coarse correction matrix
        """
        self.RT = np.zeros(((self.bs.p + 1)*self.n,(self.bs.p + 1)*int(self.n/2)))
        c = 1./2.
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
        self.RT = self.RT / 2.
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

dc = DiscreteOperator(problem_=ReactionDiffusion(basis_=LagrangeBasis(order_=1)),
                      n_=8,
                      periodic_=False)

def func(dd):
    r"""
    Fmin to obtain the optimal relaxation parameter that delivers the lowest spectral radius.
    
    Parameters
    ----------
    dd: 
    DG method penalty parameter.
    """
    dc.nassemble({dc.bs.h:1./float(dc.n),dc.pb.d:dd,dc.pb.e:np.infty})#(6.**(-1))/(dc.n)**2})
    dc.assemble_cellBJ()
    cc = CoarseCorrection(discreteOperator_=dc)
    cc.assemble()
    Id = np.eye((dc.bs.p + 1)*dc.n)
    A0inv = cc.A0inv
    R = cc.R
    RT = cc.RT
    An = dc.An
    Dinv = dc.Dc
    el = 0
    if (dc.periodic): el = 1
    def func2(rlx):
        M=(Id - RT.dot(A0inv.dot(R.dot(An)))).dot((Id-rlx*Dinv.dot(An)))
        return sorted(abs(np.real(np.linalg.eigvals(M))),reverse=True)[el]
        
    xmin,ffmin,dum1,dum2,dum3 = fmin(func2,np.array([0.1]),ftol=0.000001,xtol=0.000001,full_output=True,disp=False)

    # MM = (Id - RT.dot(A0inv.dot(R.dot(An)))).dot((Id-xmin[0]*Dinv.dot(An)))
    # conveig = 0.
    # for _ in range(10):
    #     v = np.random.rand((dc.bs.p + 1)*dc.n)
    #     norm = 1.;
    #     while (norm > 1E-4):
    #         v = MM.dot(v)
    #         norm = np.linalg.norm(v)
    #     eig = abs(v.dot(MM.dot(v))/v.dot(v))
    #     if (eig > conveig) : conveig = eig

    print("{"+str(xmin[0])+","+str(ffmin)+"},")#+str(dd)+","+str(conveig))
    
for i in range(0,51):
    func(1.001+((5.-1.001)*(2.**(float(i)/50.) - 1.)))

