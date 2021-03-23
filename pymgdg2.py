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
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as mp
np.set_printoptions(precision=4,threshold=sys.maxsize,linewidth=np.inf,suppress=True)
import lobatto as lo

from pyevtk.hl import unstructuredGridToVTK
from pyevtk.vtk import VtkQuad
import numpy as np

class TensorBasis:
    def __init__(self,order_):
        self.p = order_
        self.x,self.y,self.h = sy.symbols('x y h')
        self.Fx = sy.ones(1,4)
        self.Fx[0] = (self.h - self.x)*(self.h - self.y) / (self.h * self.h)
        self.Fx[1] = (         self.x)*(self.h - self.y) / (self.h * self.h) 
        self.Fx[2] = (self.h - self.x)*(         self.y) / (self.h * self.h)
        self.Fx[3] = (         self.x)*(         self.y) / (self.h * self.h)

class Diffusion2:
    def __init__(self,basis_):
        self.bs = basis_
        self.e,self.d = sy.symbols('e d')
        
    def cell(self):
        r"""
        Cell integration
        """
        CM = sy.Matrix(sy.zeros(4,4))
        for i in range(4):
            for j in range(4):
                fa=self.bs.Fx[i]
                fb=self.bs.Fx[j]
                CM[i,j] = sy.integrate(\
                    sy.integrate(\
                                 fa.diff(self.bs.x) * fb.diff(self.bs.x) + fa.diff(self.bs.y) * fb.diff(self.bs.y),\
                        (self.bs.x,0,self.bs.h)),
                    (self.bs.y,0,self.bs.h)) - \
                    (sy.integrate( \
                                   (0 - fa.subs(self.bs.y,0)) * (0 + fb.diff(self.bs.y).subs(self.bs.y,0))/2 + \
                                   (0 - fb.subs(self.bs.y,0)) * (0 + fa.diff(self.bs.y).subs(self.bs.y,0))/2 \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     sy.integrate( \
                                   (fa.subs(self.bs.x,self.bs.h) - 0) * (0 + fb.diff(self.bs.x).subs(self.bs.x,self.bs.h))/2 + \
                                   (fb.subs(self.bs.x,self.bs.h) - 0) * (0 + fa.diff(self.bs.x).subs(self.bs.x,self.bs.h))/2 \
                                   ,(self.bs.y,0,self.bs.h)) + \
                     sy.integrate( \
                                   (fa.subs(self.bs.y,self.bs.h) - 0) * (0 + fb.diff(self.bs.y).subs(self.bs.y,self.bs.h))/2 + \
                                   (fb.subs(self.bs.y,self.bs.h) - 0) * (0 + fa.diff(self.bs.y).subs(self.bs.y,self.bs.h))/2 \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     sy.integrate( \
                                   (0 - fa.subs(self.bs.x,0)) * (0 + fb.diff(self.bs.x).subs(self.bs.x,0))/2 + \
                                   (0 - fb.subs(self.bs.x,0)) * (0 + fa.diff(self.bs.x).subs(self.bs.x,0))/2 \
                                   ,(self.bs.y,0,self.bs.h)) \
                    ) + \
                    self.d / self.bs.h * \
                    (sy.integrate( \
                                   (0 - fa.subs(self.bs.y,0)) * (0 - fb.subs(self.bs.y,0)) \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     sy.integrate( \
                                   (fa.subs(self.bs.x,self.bs.h) - 0) * (fb.subs(self.bs.x,self.bs.h) - 0) \
                                   ,(self.bs.y,0,self.bs.h)) + \
                     sy.integrate( \
                                   (fa.subs(self.bs.y,self.bs.h) - 0) * (fb.subs(self.bs.y,self.bs.h) - 0) \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     sy.integrate( \
                                   (0 - fa.subs(self.bs.x,0)) * (0 - fb.subs(self.bs.x,0)) \
                                   ,(self.bs.y,0,self.bs.h)) \
                    )

        CM0 = sy.Matrix(sy.zeros(4,4))
        for i in range(4):
            for j in range(4):
                fa=self.bs.Fx[i]
                fb=self.bs.Fx[j]
                CM0[i,j] = sy.integrate(\
                    sy.integrate(\
                                 fa.diff(self.bs.x) * fb.diff(self.bs.x) + fa.diff(self.bs.y) * fb.diff(self.bs.y),\
                        (self.bs.x,0,self.bs.h)),
                    (self.bs.y,0,self.bs.h)) - \
                    (2 * sy.integrate( \
                                   (0 - fa.subs(self.bs.y,0)) * (0 + fb.diff(self.bs.y).subs(self.bs.y,0))/2 + \
                                   (0 - fb.subs(self.bs.y,0)) * (0 + fa.diff(self.bs.y).subs(self.bs.y,0))/2 \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     2 * sy.integrate( \
                                   (fa.subs(self.bs.x,self.bs.h) - 0) * (0 + fb.diff(self.bs.x).subs(self.bs.x,self.bs.h))/2 + \
                                   (fb.subs(self.bs.x,self.bs.h) - 0) * (0 + fa.diff(self.bs.x).subs(self.bs.x,self.bs.h))/2 \
                                   ,(self.bs.y,0,self.bs.h)) + \
                     2 * sy.integrate( \
                                   (fa.subs(self.bs.y,self.bs.h) - 0) * (0 + fb.diff(self.bs.y).subs(self.bs.y,self.bs.h))/2 + \
                                   (fb.subs(self.bs.y,self.bs.h) - 0) * (0 + fa.diff(self.bs.y).subs(self.bs.y,self.bs.h))/2 \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     2 * sy.integrate( \
                                   (0 - fa.subs(self.bs.x,0)) * (0 + fb.diff(self.bs.x).subs(self.bs.x,0))/2 + \
                                   (0 - fb.subs(self.bs.x,0)) * (0 + fa.diff(self.bs.x).subs(self.bs.x,0))/2 \
                                   ,(self.bs.y,0,self.bs.h)) \
                    ) + \
                    self.d / self.bs.h * \
                    (2 * sy.integrate( \
                                   (0 - fa.subs(self.bs.y,0)) * (0 - fb.subs(self.bs.y,0)) \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     2 * sy.integrate( \
                                   (fa.subs(self.bs.x,self.bs.h) - 0) * (fb.subs(self.bs.x,self.bs.h) - 0) \
                                   ,(self.bs.y,0,self.bs.h)) + \
                     2 * sy.integrate( \
                                   (fa.subs(self.bs.y,self.bs.h) - 0) * (fb.subs(self.bs.y,self.bs.h) - 0) \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     2 * sy.integrate( \
                                   (0 - fa.subs(self.bs.x,0)) * (0 - fb.subs(self.bs.x,0)) \
                                   ,(self.bs.y,0,self.bs.h)) \
                    )
                
        CMxy0 = sy.Matrix(sy.zeros(4,4))
        for i in range(4):
            for j in range(4):
                fa=self.bs.Fx[i]
                fb=self.bs.Fx[j]
                CMxy0[i,j] = sy.integrate(\
                    sy.integrate(\
                                 fa.diff(self.bs.x) * fb.diff(self.bs.x) + fa.diff(self.bs.y) * fb.diff(self.bs.y),\
                        (self.bs.x,0,self.bs.h)),
                    (self.bs.y,0,self.bs.h)) - \
                    (2 * sy.integrate( \
                                   (0 - fa.subs(self.bs.y,0)) * (0 + fb.diff(self.bs.y).subs(self.bs.y,0))/2 + \
                                   (0 - fb.subs(self.bs.y,0)) * (0 + fa.diff(self.bs.y).subs(self.bs.y,0))/2 \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     sy.integrate( \
                                   (fa.subs(self.bs.x,self.bs.h) - 0) * (0 + fb.diff(self.bs.x).subs(self.bs.x,self.bs.h))/2 + \
                                   (fb.subs(self.bs.x,self.bs.h) - 0) * (0 + fa.diff(self.bs.x).subs(self.bs.x,self.bs.h))/2 \
                                   ,(self.bs.y,0,self.bs.h)) + \
                     sy.integrate( \
                                   (fa.subs(self.bs.y,self.bs.h) - 0) * (0 + fb.diff(self.bs.y).subs(self.bs.y,self.bs.h))/2 + \
                                   (fb.subs(self.bs.y,self.bs.h) - 0) * (0 + fa.diff(self.bs.y).subs(self.bs.y,self.bs.h))/2 \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     2 * sy.integrate( \
                                   (0 - fa.subs(self.bs.x,0)) * (0 + fb.diff(self.bs.x).subs(self.bs.x,0))/2 + \
                                   (0 - fb.subs(self.bs.x,0)) * (0 + fa.diff(self.bs.x).subs(self.bs.x,0))/2 \
                                   ,(self.bs.y,0,self.bs.h)) \
                    ) + \
                    self.d / self.bs.h * \
                    (2 * sy.integrate( \
                                   (0 - fa.subs(self.bs.y,0)) * (0 - fb.subs(self.bs.y,0)) \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     sy.integrate( \
                                   (fa.subs(self.bs.x,self.bs.h) - 0) * (fb.subs(self.bs.x,self.bs.h) - 0) \
                                   ,(self.bs.y,0,self.bs.h)) + \
                     sy.integrate( \
                                   (fa.subs(self.bs.y,self.bs.h) - 0) * (fb.subs(self.bs.y,self.bs.h) - 0) \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     2 * sy.integrate( \
                                   (0 - fa.subs(self.bs.x,0)) * (0 - fb.subs(self.bs.x,0)) \
                                   ,(self.bs.y,0,self.bs.h)) \
                    )

        CMxy1 = sy.Matrix(sy.zeros(4,4))
        for i in range(4):
            for j in range(4):
                fa=self.bs.Fx[i]
                fb=self.bs.Fx[j]
                CMxy1[i,j] = sy.integrate(\
                    sy.integrate(\
                                 fa.diff(self.bs.x) * fb.diff(self.bs.x) + fa.diff(self.bs.y) * fb.diff(self.bs.y),\
                        (self.bs.x,0,self.bs.h)),
                    (self.bs.y,0,self.bs.h)) - \
                    (2 * sy.integrate( \
                                   (0 - fa.subs(self.bs.y,0)) * (0 + fb.diff(self.bs.y).subs(self.bs.y,0))/2 + \
                                   (0 - fb.subs(self.bs.y,0)) * (0 + fa.diff(self.bs.y).subs(self.bs.y,0))/2 \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     2 * sy.integrate( \
                                   (fa.subs(self.bs.x,self.bs.h) - 0) * (0 + fb.diff(self.bs.x).subs(self.bs.x,self.bs.h))/2 + \
                                   (fb.subs(self.bs.x,self.bs.h) - 0) * (0 + fa.diff(self.bs.x).subs(self.bs.x,self.bs.h))/2 \
                                   ,(self.bs.y,0,self.bs.h)) + \
                     sy.integrate( \
                                   (fa.subs(self.bs.y,self.bs.h) - 0) * (0 + fb.diff(self.bs.y).subs(self.bs.y,self.bs.h))/2 + \
                                   (fb.subs(self.bs.y,self.bs.h) - 0) * (0 + fa.diff(self.bs.y).subs(self.bs.y,self.bs.h))/2 \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     sy.integrate( \
                                   (0 - fa.subs(self.bs.x,0)) * (0 + fb.diff(self.bs.x).subs(self.bs.x,0))/2 + \
                                   (0 - fb.subs(self.bs.x,0)) * (0 + fa.diff(self.bs.x).subs(self.bs.x,0))/2 \
                                   ,(self.bs.y,0,self.bs.h)) \
                    ) + \
                    self.d / self.bs.h * \
                    (2 * sy.integrate( \
                                   (0 - fa.subs(self.bs.y,0)) * (0 - fb.subs(self.bs.y,0)) \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     2 * sy.integrate( \
                                   (fa.subs(self.bs.x,self.bs.h) - 0) * (fb.subs(self.bs.x,self.bs.h) - 0) \
                                   ,(self.bs.y,0,self.bs.h)) + \
                     sy.integrate( \
                                   (fa.subs(self.bs.y,self.bs.h) - 0) * (fb.subs(self.bs.y,self.bs.h) - 0) \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     sy.integrate( \
                                   (0 - fa.subs(self.bs.x,0)) * (0 - fb.subs(self.bs.x,0)) \
                                   ,(self.bs.y,0,self.bs.h)) \
                    )

        CMxy2 = sy.Matrix(sy.zeros(4,4))
        for i in range(4):
            for j in range(4):
                fa=self.bs.Fx[i]
                fb=self.bs.Fx[j]
                CMxy2[i,j] = sy.integrate(\
                    sy.integrate(\
                                 fa.diff(self.bs.x) * fb.diff(self.bs.x) + fa.diff(self.bs.y) * fb.diff(self.bs.y),\
                        (self.bs.x,0,self.bs.h)),
                    (self.bs.y,0,self.bs.h)) - \
                    (sy.integrate( \
                                   (0 - fa.subs(self.bs.y,0)) * (0 + fb.diff(self.bs.y).subs(self.bs.y,0))/2 + \
                                   (0 - fb.subs(self.bs.y,0)) * (0 + fa.diff(self.bs.y).subs(self.bs.y,0))/2 \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     2 * sy.integrate( \
                                   (fa.subs(self.bs.x,self.bs.h) - 0) * (0 + fb.diff(self.bs.x).subs(self.bs.x,self.bs.h))/2 + \
                                   (fb.subs(self.bs.x,self.bs.h) - 0) * (0 + fa.diff(self.bs.x).subs(self.bs.x,self.bs.h))/2 \
                                   ,(self.bs.y,0,self.bs.h)) + \
                     2 * sy.integrate( \
                                   (fa.subs(self.bs.y,self.bs.h) - 0) * (0 + fb.diff(self.bs.y).subs(self.bs.y,self.bs.h))/2 + \
                                   (fb.subs(self.bs.y,self.bs.h) - 0) * (0 + fa.diff(self.bs.y).subs(self.bs.y,self.bs.h))/2 \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     sy.integrate( \
                                   (0 - fa.subs(self.bs.x,0)) * (0 + fb.diff(self.bs.x).subs(self.bs.x,0))/2 + \
                                   (0 - fb.subs(self.bs.x,0)) * (0 + fa.diff(self.bs.x).subs(self.bs.x,0))/2 \
                                   ,(self.bs.y,0,self.bs.h)) \
                    ) + \
                    self.d / self.bs.h * \
                    (sy.integrate( \
                                   (0 - fa.subs(self.bs.y,0)) * (0 - fb.subs(self.bs.y,0)) \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     2 * sy.integrate( \
                                   (fa.subs(self.bs.x,self.bs.h) - 0) * (fb.subs(self.bs.x,self.bs.h) - 0) \
                                   ,(self.bs.y,0,self.bs.h)) + \
                     2 * sy.integrate( \
                                   (fa.subs(self.bs.y,self.bs.h) - 0) * (fb.subs(self.bs.y,self.bs.h) - 0) \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     sy.integrate( \
                                   (0 - fa.subs(self.bs.x,0)) * (0 - fb.subs(self.bs.x,0)) \
                                   ,(self.bs.y,0,self.bs.h)) \
                    )

        CMxy3 = sy.Matrix(sy.zeros(4,4))
        for i in range(4):
            for j in range(4):
                fa=self.bs.Fx[i]
                fb=self.bs.Fx[j]
                CMxy3[i,j] = sy.integrate(\
                    sy.integrate(\
                                 fa.diff(self.bs.x) * fb.diff(self.bs.x) + fa.diff(self.bs.y) * fb.diff(self.bs.y),\
                        (self.bs.x,0,self.bs.h)),
                    (self.bs.y,0,self.bs.h)) - \
                    (sy.integrate( \
                                   (0 - fa.subs(self.bs.y,0)) * (0 + fb.diff(self.bs.y).subs(self.bs.y,0))/2 + \
                                   (0 - fb.subs(self.bs.y,0)) * (0 + fa.diff(self.bs.y).subs(self.bs.y,0))/2 \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     sy.integrate( \
                                   (fa.subs(self.bs.x,self.bs.h) - 0) * (0 + fb.diff(self.bs.x).subs(self.bs.x,self.bs.h))/2 + \
                                   (fb.subs(self.bs.x,self.bs.h) - 0) * (0 + fa.diff(self.bs.x).subs(self.bs.x,self.bs.h))/2 \
                                   ,(self.bs.y,0,self.bs.h)) + \
                     2 * sy.integrate( \
                                   (fa.subs(self.bs.y,self.bs.h) - 0) * (0 + fb.diff(self.bs.y).subs(self.bs.y,self.bs.h))/2 + \
                                   (fb.subs(self.bs.y,self.bs.h) - 0) * (0 + fa.diff(self.bs.y).subs(self.bs.y,self.bs.h))/2 \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     2 * sy.integrate( \
                                   (0 - fa.subs(self.bs.x,0)) * (0 + fb.diff(self.bs.x).subs(self.bs.x,0))/2 + \
                                   (0 - fb.subs(self.bs.x,0)) * (0 + fa.diff(self.bs.x).subs(self.bs.x,0))/2 \
                                   ,(self.bs.y,0,self.bs.h)) \
                    ) + \
                    self.d / self.bs.h * \
                    (sy.integrate( \
                                   (0 - fa.subs(self.bs.y,0)) * (0 - fb.subs(self.bs.y,0)) \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     sy.integrate( \
                                   (fa.subs(self.bs.x,self.bs.h) - 0) * (fb.subs(self.bs.x,self.bs.h) - 0) \
                                   ,(self.bs.y,0,self.bs.h)) + \
                     2 * sy.integrate( \
                                   (fa.subs(self.bs.y,self.bs.h) - 0) * (fb.subs(self.bs.y,self.bs.h) - 0) \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     2 * sy.integrate( \
                                   (0 - fa.subs(self.bs.x,0)) * (0 - fb.subs(self.bs.x,0)) \
                                   ,(self.bs.y,0,self.bs.h)) \
                    )

        CMx0 = sy.Matrix(sy.zeros(4,4))
        for i in range(4):
            for j in range(4):
                fa=self.bs.Fx[i]
                fb=self.bs.Fx[j]
                CMx0[i,j] = sy.integrate(\
                    sy.integrate(\
                                 fa.diff(self.bs.x) * fb.diff(self.bs.x) + fa.diff(self.bs.y) * fb.diff(self.bs.y),\
                        (self.bs.x,0,self.bs.h)),
                    (self.bs.y,0,self.bs.h)) - \
                    (sy.integrate( \
                                   (0 - fa.subs(self.bs.y,0)) * (0 + fb.diff(self.bs.y).subs(self.bs.y,0))/2 + \
                                   (0 - fb.subs(self.bs.y,0)) * (0 + fa.diff(self.bs.y).subs(self.bs.y,0))/2 \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     2 * sy.integrate( \
                                   (fa.subs(self.bs.x,self.bs.h) - 0) * (0 + fb.diff(self.bs.x).subs(self.bs.x,self.bs.h))/2 + \
                                   (fb.subs(self.bs.x,self.bs.h) - 0) * (0 + fa.diff(self.bs.x).subs(self.bs.x,self.bs.h))/2 \
                                   ,(self.bs.y,0,self.bs.h)) + \
                     sy.integrate( \
                                   (fa.subs(self.bs.y,self.bs.h) - 0) * (0 + fb.diff(self.bs.y).subs(self.bs.y,self.bs.h))/2 + \
                                   (fb.subs(self.bs.y,self.bs.h) - 0) * (0 + fa.diff(self.bs.y).subs(self.bs.y,self.bs.h))/2 \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     sy.integrate( \
                                   (0 - fa.subs(self.bs.x,0)) * (0 + fb.diff(self.bs.x).subs(self.bs.x,0))/2 + \
                                   (0 - fb.subs(self.bs.x,0)) * (0 + fa.diff(self.bs.x).subs(self.bs.x,0))/2 \
                                   ,(self.bs.y,0,self.bs.h)) \
                    ) + \
                    self.d / self.bs.h * \
                    (sy.integrate( \
                                   (0 - fa.subs(self.bs.y,0)) * (0 - fb.subs(self.bs.y,0)) \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     2 * sy.integrate( \
                                   (fa.subs(self.bs.x,self.bs.h) - 0) * (fb.subs(self.bs.x,self.bs.h) - 0) \
                                   ,(self.bs.y,0,self.bs.h)) + \
                     sy.integrate( \
                                   (fa.subs(self.bs.y,self.bs.h) - 0) * (fb.subs(self.bs.y,self.bs.h) - 0) \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     sy.integrate( \
                                   (0 - fa.subs(self.bs.x,0)) * (0 - fb.subs(self.bs.x,0)) \
                                   ,(self.bs.y,0,self.bs.h)) \
                    )

        CMx1 = sy.Matrix(sy.zeros(4,4))
        for i in range(4):
            for j in range(4):
                fa=self.bs.Fx[i]
                fb=self.bs.Fx[j]
                CMx1[i,j] = sy.integrate(\
                    sy.integrate(\
                                 fa.diff(self.bs.x) * fb.diff(self.bs.x) + fa.diff(self.bs.y) * fb.diff(self.bs.y),\
                        (self.bs.x,0,self.bs.h)),
                    (self.bs.y,0,self.bs.h)) - \
                    (sy.integrate( \
                                   (0 - fa.subs(self.bs.y,0)) * (0 + fb.diff(self.bs.y).subs(self.bs.y,0))/2 + \
                                   (0 - fb.subs(self.bs.y,0)) * (0 + fa.diff(self.bs.y).subs(self.bs.y,0))/2 \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     sy.integrate( \
                                   (fa.subs(self.bs.x,self.bs.h) - 0) * (0 + fb.diff(self.bs.x).subs(self.bs.x,self.bs.h))/2 + \
                                   (fb.subs(self.bs.x,self.bs.h) - 0) * (0 + fa.diff(self.bs.x).subs(self.bs.x,self.bs.h))/2 \
                                   ,(self.bs.y,0,self.bs.h)) + \
                     sy.integrate( \
                                   (fa.subs(self.bs.y,self.bs.h) - 0) * (0 + fb.diff(self.bs.y).subs(self.bs.y,self.bs.h))/2 + \
                                   (fb.subs(self.bs.y,self.bs.h) - 0) * (0 + fa.diff(self.bs.y).subs(self.bs.y,self.bs.h))/2 \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     2 * sy.integrate( \
                                   (0 - fa.subs(self.bs.x,0)) * (0 + fb.diff(self.bs.x).subs(self.bs.x,0))/2 + \
                                   (0 - fb.subs(self.bs.x,0)) * (0 + fa.diff(self.bs.x).subs(self.bs.x,0))/2 \
                                   ,(self.bs.y,0,self.bs.h)) \
                    ) + \
                    self.d / self.bs.h * \
                    (sy.integrate( \
                                   (0 - fa.subs(self.bs.y,0)) * (0 - fb.subs(self.bs.y,0)) \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     sy.integrate( \
                                   (fa.subs(self.bs.x,self.bs.h) - 0) * (fb.subs(self.bs.x,self.bs.h) - 0) \
                                   ,(self.bs.y,0,self.bs.h)) + \
                     sy.integrate( \
                                   (fa.subs(self.bs.y,self.bs.h) - 0) * (fb.subs(self.bs.y,self.bs.h) - 0) \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     2 * sy.integrate( \
                                   (0 - fa.subs(self.bs.x,0)) * (0 - fb.subs(self.bs.x,0)) \
                                   ,(self.bs.y,0,self.bs.h)) \
                    )

        CMy0 = sy.Matrix(sy.zeros(4,4))
        for i in range(4):
            for j in range(4):
                fa=self.bs.Fx[i]
                fb=self.bs.Fx[j]
                CMy0[i,j] = sy.integrate(\
                    sy.integrate(\
                                 fa.diff(self.bs.x) * fb.diff(self.bs.x) + fa.diff(self.bs.y) * fb.diff(self.bs.y),\
                        (self.bs.x,0,self.bs.h)),
                    (self.bs.y,0,self.bs.h)) - \
                    (sy.integrate( \
                                   (0 - fa.subs(self.bs.y,0)) * (0 + fb.diff(self.bs.y).subs(self.bs.y,0))/2 + \
                                   (0 - fb.subs(self.bs.y,0)) * (0 + fa.diff(self.bs.y).subs(self.bs.y,0))/2 \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     sy.integrate( \
                                   (fa.subs(self.bs.x,self.bs.h) - 0) * (0 + fb.diff(self.bs.x).subs(self.bs.x,self.bs.h))/2 + \
                                   (fb.subs(self.bs.x,self.bs.h) - 0) * (0 + fa.diff(self.bs.x).subs(self.bs.x,self.bs.h))/2 \
                                   ,(self.bs.y,0,self.bs.h)) + \
                     2 * sy.integrate( \
                                   (fa.subs(self.bs.y,self.bs.h) - 0) * (0 + fb.diff(self.bs.y).subs(self.bs.y,self.bs.h))/2 + \
                                   (fb.subs(self.bs.y,self.bs.h) - 0) * (0 + fa.diff(self.bs.y).subs(self.bs.y,self.bs.h))/2 \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     sy.integrate( \
                                   (0 - fa.subs(self.bs.x,0)) * (0 + fb.diff(self.bs.x).subs(self.bs.x,0))/2 + \
                                   (0 - fb.subs(self.bs.x,0)) * (0 + fa.diff(self.bs.x).subs(self.bs.x,0))/2 \
                                   ,(self.bs.y,0,self.bs.h)) \
                    ) + \
                    self.d / self.bs.h * \
                    (sy.integrate( \
                                   (0 - fa.subs(self.bs.y,0)) * (0 - fb.subs(self.bs.y,0)) \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     sy.integrate( \
                                   (fa.subs(self.bs.x,self.bs.h) - 0) * (fb.subs(self.bs.x,self.bs.h) - 0) \
                                   ,(self.bs.y,0,self.bs.h)) + \
                     2 * sy.integrate( \
                                   (fa.subs(self.bs.y,self.bs.h) - 0) * (fb.subs(self.bs.y,self.bs.h) - 0) \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     sy.integrate( \
                                   (0 - fa.subs(self.bs.x,0)) * (0 - fb.subs(self.bs.x,0)) \
                                   ,(self.bs.y,0,self.bs.h)) \
                    )

        CMy1 = sy.Matrix(sy.zeros(4,4))
        for i in range(4):
            for j in range(4):
                fa=self.bs.Fx[i]
                fb=self.bs.Fx[j]
                CMy1[i,j] = sy.integrate(\
                    sy.integrate(\
                                 fa.diff(self.bs.x) * fb.diff(self.bs.x) + fa.diff(self.bs.y) * fb.diff(self.bs.y),\
                        (self.bs.x,0,self.bs.h)),
                    (self.bs.y,0,self.bs.h)) - \
                    (2 * sy.integrate( \
                                   (0 - fa.subs(self.bs.y,0)) * (0 + fb.diff(self.bs.y).subs(self.bs.y,0))/2 + \
                                   (0 - fb.subs(self.bs.y,0)) * (0 + fa.diff(self.bs.y).subs(self.bs.y,0))/2 \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     sy.integrate( \
                                   (fa.subs(self.bs.x,self.bs.h) - 0) * (0 + fb.diff(self.bs.x).subs(self.bs.x,self.bs.h))/2 + \
                                   (fb.subs(self.bs.x,self.bs.h) - 0) * (0 + fa.diff(self.bs.x).subs(self.bs.x,self.bs.h))/2 \
                                   ,(self.bs.y,0,self.bs.h)) + \
                     sy.integrate( \
                                   (fa.subs(self.bs.y,self.bs.h) - 0) * (0 + fb.diff(self.bs.y).subs(self.bs.y,self.bs.h))/2 + \
                                   (fb.subs(self.bs.y,self.bs.h) - 0) * (0 + fa.diff(self.bs.y).subs(self.bs.y,self.bs.h))/2 \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     sy.integrate( \
                                   (0 - fa.subs(self.bs.x,0)) * (0 + fb.diff(self.bs.x).subs(self.bs.x,0))/2 + \
                                   (0 - fb.subs(self.bs.x,0)) * (0 + fa.diff(self.bs.x).subs(self.bs.x,0))/2 \
                                   ,(self.bs.y,0,self.bs.h)) \
                    ) + \
                    self.d / self.bs.h * \
                    (2 * sy.integrate( \
                                   (0 - fa.subs(self.bs.y,0)) * (0 - fb.subs(self.bs.y,0)) \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     sy.integrate( \
                                   (fa.subs(self.bs.x,self.bs.h) - 0) * (fb.subs(self.bs.x,self.bs.h) - 0) \
                                   ,(self.bs.y,0,self.bs.h)) + \
                     sy.integrate( \
                                   (fa.subs(self.bs.y,self.bs.h) - 0) * (fb.subs(self.bs.y,self.bs.h) - 0) \
                                   ,(self.bs.x,0,self.bs.h)) + \
                     sy.integrate( \
                                   (0 - fa.subs(self.bs.x,0)) * (0 - fb.subs(self.bs.x,0)) \
                                   ,(self.bs.y,0,self.bs.h)) \
                    )
                
                
        return [CM,CM0,CMxy0,CMxy1,CMxy2,CMxy3,CMx0,CMx1,CMy0,CMy1]

    def face(self):
        r"""
        Face integration
        """
        FMx = sy.Matrix(sy.zeros(4,4))
        FMy = sy.Matrix(sy.zeros(4,4))
        for i in range(4):
            for j in range(4):
                fa=self.bs.Fx[i]
                fb=self.bs.Fx[j]
                FMx[i,j] = - \
                    (sy.integrate( \
                                   (fa.subs(self.bs.x,self.bs.h) - 0) * (0 + fb.diff(self.bs.x).subs(self.bs.x,0))/2 + \
                                   (0 - fb.subs(self.bs.x,0)) * (0 + fa.diff(self.bs.x).subs(self.bs.x,self.bs.h))/2 \
                                   ,(self.bs.y,0,self.bs.h))
                    ) + \
                    self.d / self.bs.h * \
                    (sy.integrate( \
                                   (fa.subs(self.bs.x,self.bs.h) - 0) * (0 - fb.subs(self.bs.x,0)) \
                                   ,(self.bs.y,0,self.bs.h))
                    )
                FMy[i,j] = - \
                    (sy.integrate( \
                                   (fa.subs(self.bs.y,self.bs.h) - 0) * (0 + fb.diff(self.bs.y).subs(self.bs.y,0))/2 + \
                                   (0 - fb.subs(self.bs.y,0)) * (0 + fa.diff(self.bs.y).subs(self.bs.y,self.bs.h))/2 \
                                   ,(self.bs.x,0,self.bs.h))
                    ) + \
                self.d / self.bs.h * \
                    (sy.integrate( \
                                   (fa.subs(self.bs.y,self.bs.h) - 0) * (0 - fb.subs(self.bs.y,0)) \
                                   ,(self.bs.x,0,self.bs.h))
                    )


        return [FMx,FMy]

class DiscreteOperator:
    def __init__(self,problem_,n_,periodic_=False):
        self.pb = problem_
        self.bs = problem_.bs
        [self.CM,self.CM0,self.CMxy0,self.CMxy1,self.CMxy2,self.CMxy3,self.CMx0,self.CMx1,self.CMy0,self.CMy1] = problem_.cell()
        [self.FMx,self.FMy] = problem_.face()
        self.n = n_
        self.periodic = periodic_
        self.A = sy.Matrix(sy.zeros(1,1))
        self.An = np.zeros((1,1))
        self.An = sc.sparse.dok_matrix((1,1),dtype=np.float32)
        self.Dc = np.zeros((1,1))

        self.B = np.matrix([[0,2],
                            [1,3]])
        for i in range(self.n.bit_length()-2):
            self.B = np.block([[self.B+0*np.max(self.B)+0,self.B+2*np.max(self.B)+2],\
                               [self.B+1*np.max(self.B)+1,self.B+3*np.max(self.B)+3]])
        
    def assemble(self):
        r"""
        Assembles the symbolic matrix.
        """
        self.A = sy.Matrix(sy.zeros(4*self.n**2,4*self.n**2))
        B = self.B
        if (self.periodic):
            for i in range(self.n):
                for j in range(self.n):
                    self.A[4*B[i,j]:4*B[i,j]+4,4*(B[i,j]):4*(B[i,j])+4] = self.CM
                    self.A[4*B[i,j]:4*B[i,j]+4,4*(B[i,(j+1)%self.n]):4*(B[i,(j+1)%self.n])+4] = self.FMy
                    self.A[4*B[i,j]:4*B[i,j]+4,4*(B[i,(j-1)%self.n]):4*(B[i,(j-1)%self.n])+4] = self.FMy.transpose()
                    self.A[4*B[i,j]:4*B[i,j]+4,4*(B[(i-1)%self.n,j]):4*(B[(i-1)%self.n,j])+4] = self.FMx.transpose()
                    self.A[4*B[i,j]:4*B[i,j]+4,4*(B[(i+1)%self.n,j]):4*(B[(i+1)%self.n,j])+4] = self.FMx
        else:
            if (self.n==1):
                self.A = self.CM0
            else:
                for i in range(self.n):
                    for j in range(self.n):
                        self.A[4*B[i,j]:4*B[i,j]+4,4*(B[i,j]):4*(B[i,j])+4] = self.CM
                        if ((j+1)==(j+1)%self.n): self.A[4*B[i,j]:4*B[i,j]+4,4*(B[i,(j+1)]):4*(B[i,(j+1)])+4] = self.FMy
                        if ((j-1)==(j-1)%self.n): self.A[4*B[i,j]:4*B[i,j]+4,4*(B[i,(j-1)]):4*(B[i,(j-1)])+4] = self.FMy.transpose()
                        if ((i-1)==(i-1)%self.n): self.A[4*B[i,j]:4*B[i,j]+4,4*(B[(i-1),j]):4*(B[(i-1),j])+4] = self.FMx.transpose()
                        if ((i+1)==(i+1)%self.n): self.A[4*B[i,j]:4*B[i,j]+4,4*(B[(i+1),j]):4*(B[(i+1),j])+4] = self.FMx
                for i in range(1,self.n-1):
                    self.A[4*B[i,0]:4*B[i,0]+4,4*(B[i,0]):4*(B[i,0])+4] = self.CMy1
                    self.A[4*B[i,self.n-1]:4*B[i,self.n-1]+4,4*(B[i,self.n-1]):4*(B[i,self.n-1])+4] = self.CMy0
                for j in range(1,self.n-1):
                    self.A[4*B[0,j]:4*B[0,j]+4,4*(B[0,j]):4*(B[0,j])+4] = self.CMx1
                    self.A[4*B[self.n-1,j]:4*B[self.n-1,j]+4,4*(B[self.n-1,j]):4*(B[self.n-1,j])+4] = self.CMx0
                self.A[4*B[0,0]:4*B[0,0]+4,4*(B[0,0]):4*(B[0,0])+4] = self.CMxy0
                self.A[4*B[self.n-1,0]:4*B[self.n-1,0]+4,4*(B[self.n-1,0]):4*(B[self.n-1,0])+4] = self.CMxy1
                self.A[4*B[0,self.n-1]:4*B[0,self.n-1]+4,4*(B[0,self.n-1]):4*(B[0,self.n-1])+4] = self.CMxy3
                self.A[4*B[self.n-1,self.n-1]:4*B[self.n-1,self.n-1]+4,4*(B[self.n-1,self.n-1]):4*(B[self.n-1,self.n-1])+4] = self.CMxy2


    def nassemble(self,par):
        r"""
        Assembles the numerical matrix.

        Parameters
        ----------
        par : {}
            Values for all the symbols in the symbolic matrix.
        """
        self.An = np.zeros((4*self.n**2,4*self.n**2))
        CM = np.matrix(self.CM.subs(par)).astype(float)
        CMxy0 = np.matrix(self.CMxy0.subs(par)).astype(float)
        CMxy1 = np.matrix(self.CMxy1.subs(par)).astype(float)
        CMxy2 = np.matrix(self.CMxy2.subs(par)).astype(float)
        CMxy3 = np.matrix(self.CMxy3.subs(par)).astype(float)
        CMx0 = np.matrix(self.CMx0.subs(par)).astype(float)
        CMx1 = np.matrix(self.CMx1.subs(par)).astype(float)
        CMy0 = np.matrix(self.CMy0.subs(par)).astype(float)
        CMy1 = np.matrix(self.CMy1.subs(par)).astype(float)
        FMx = np.matrix(self.FMx.subs(par)).astype(float)
        FMy = np.matrix(self.FMy.subs(par)).astype(float)
        B = self.B
        if (self.periodic):
            for i in range(self.n):
                for j in range(self.n):
                    self.An[4*B[i,j]:4*B[i,j]+4,4*(B[i,j]):4*(B[i,j])+4] = CM
                    if ((j+1)==(j+1)%self.n): self.An[4*B[i,j]:4*B[i,j]+4,4*(B[i,(j+1)]):4*(B[i,(j+1)])+4] = FMy
                    if ((j-1)==(j-1)%self.n): self.An[4*B[i,j]:4*B[i,j]+4,4*(B[i,(j-1)]):4*(B[i,(j-1)])+4] = FMy.transpose()
                    if ((i-1)==(i-1)%self.n): self.An[4*B[i,j]:4*B[i,j]+4,4*(B[(i-1),j]):4*(B[(i-1),j])+4] = FMx.transpose()
                    if ((i+1)==(i+1)%self.n): self.An[4*B[i,j]:4*B[i,j]+4,4*(B[(i+1),j]):4*(B[(i+1),j])+4] = FMx
        else:
            for i in range(self.n):
                for j in range(self.n):
                    self.An[4*B[i,j]:4*B[i,j]+4,4*(B[i,j]):4*(B[i,j])+4] = CM
                    if ((j+1)==(j+1)%self.n): self.An[4*B[i,j]:4*B[i,j]+4,4*(B[i,(j+1)]):4*(B[i,(j+1)])+4] = FMy
                    if ((j-1)==(j-1)%self.n): self.An[4*B[i,j]:4*B[i,j]+4,4*(B[i,(j-1)]):4*(B[i,(j-1)])+4] = FMy.transpose()
                    if ((i-1)==(i-1)%self.n): self.An[4*B[i,j]:4*B[i,j]+4,4*(B[(i-1),j]):4*(B[(i-1),j])+4] = FMx.transpose()
                    if ((i+1)==(i+1)%self.n): self.An[4*B[i,j]:4*B[i,j]+4,4*(B[(i+1),j]):4*(B[(i+1),j])+4] = FMx
            for i in range(1,self.n-1):
                self.An[4*B[i,0]:4*B[i,0]+4,4*(B[i,0]):4*(B[i,0])+4] = CMy1
                self.An[4*B[i,self.n-1]:4*B[i,self.n-1]+4,4*(B[i,self.n-1]):4*(B[i,self.n-1])+4] = CMy0
            for j in range(1,self.n-1):
                self.An[4*B[0,j]:4*B[0,j]+4,4*(B[0,j]):4*(B[0,j])+4] = CMx1
                self.An[4*B[self.n-1,j]:4*B[self.n-1,j]+4,4*(B[self.n-1,j]):4*(B[self.n-1,j])+4] = CMx0
            self.An[4*B[0,0]:4*B[0,0]+4,4*(B[0,0]):4*(B[0,0])+4] = CMxy0
            self.An[4*B[self.n-1,0]:4*B[self.n-1,0]+4,4*(B[self.n-1,0]):4*(B[self.n-1,0])+4] = CMxy1
            self.An[4*B[0,self.n-1]:4*B[0,self.n-1]+4,4*(B[0,self.n-1]):4*(B[0,self.n-1])+4] = CMxy3
            self.An[4*B[self.n-1,self.n-1]:4*B[self.n-1,self.n-1]+4,4*(B[self.n-1,self.n-1]):4*(B[self.n-1,self.n-1])+4] = CMxy2
               
    def assemble_cellBJ(self,par):
        r"""
        Assembles the _cell_ Block Jacobi matrix.
        """
        self.Dc = np.zeros((4*self.n**2,4*self.n**2))
        if (self.periodic):
            CMinv = np.linalg.inv(np.matrix(self.CM.subs(par)).astype(float))
            for b in range(self.n**2):
                self.Dc[4*b:4*b+4,4*b:4*b+4] = CMinv
        else:
            CMinv = np.linalg.inv(np.matrix(self.CM.subs(par)).astype(float))
            CMxy0inv = np.linalg.inv(np.matrix(self.CMxy0.subs(par)).astype(float))
            CMxy1inv = np.linalg.inv(np.matrix(self.CMxy1.subs(par)).astype(float))
            CMxy2inv = np.linalg.inv(np.matrix(self.CMxy2.subs(par)).astype(float))
            CMxy3inv = np.linalg.inv(np.matrix(self.CMxy3.subs(par)).astype(float))
            CMx0inv = np.linalg.inv(np.matrix(self.CMx0.subs(par)).astype(float))
            CMx1inv = np.linalg.inv(np.matrix(self.CMx1.subs(par)).astype(float))
            CMy0inv = np.linalg.inv(np.matrix(self.CMy0.subs(par)).astype(float))
            CMy1inv = np.linalg.inv(np.matrix(self.CMy1.subs(par)).astype(float))
            B = self.B

            for i in range(self.n):
                for j in range(self.n):
                    self.Dc[4*B[i,j]:4*B[i,j]+4,4*(B[i,j]):4*(B[i,j])+4] = CMinv
            for i in range(1,self.n-1):
                self.Dc[4*B[i,0]:4*B[i,0]+4,4*(B[i,0]):4*(B[i,0])+4] = CMy1inv
                self.Dc[4*B[i,self.n-1]:4*B[i,self.n-1]+4,4*(B[i,self.n-1]):4*(B[i,self.n-1])+4] = CMy0inv
            for j in range(1,self.n-1):
                self.Dc[4*B[0,j]:4*B[0,j]+4,4*(B[0,j]):4*(B[0,j])+4] = CMx1inv
                self.Dc[4*B[self.n-1,j]:4*B[self.n-1,j]+4,4*(B[self.n-1,j]):4*(B[self.n-1,j])+4] = CMx0inv
            self.Dc[4*B[0,0]:4*B[0,0]+4,4*(B[0,0]):4*(B[0,0])+4] = CMxy0inv
            self.Dc[4*B[self.n-1,0]:4*B[self.n-1,0]+4,4*(B[self.n-1,0]):4*(B[self.n-1,0])+4] = CMxy1inv
            self.Dc[4*B[0,self.n-1]:4*B[0,self.n-1]+4,4*(B[0,self.n-1]):4*(B[0,self.n-1])+4] = CMxy3inv
            self.Dc[4*B[self.n-1,self.n-1]:4*B[self.n-1,self.n-1]+4,4*(B[self.n-1,self.n-1]):4*(B[self.n-1,self.n-1])+4] = CMxy2inv

class CoarseCorrection:
    r"""
    Coarse correction object.

    Parameters
    ----------
    dc: 
        Discrete operator
    """
    def __init__(self,discreteOperator_,c_=0.5):
        self.dc = discreteOperator_
        self.c = c_
        self.An = self.dc.An
        self.A = self.dc.A
        self.bs = self.dc.bs
        self.n = self.dc.n
        self.RT = np.zeros((4*self.n**2,self.n**2))
        self.R = np.zeros((self.n**2,4*self.n**2))
        self.RTs = sy.Matrix(sy.zeros(4*self.n**2,self.n**2))
        self.Rs = sy.Matrix(sy.zeros(self.n**2,4*self.n**2))
        self.A0 = np.zeros((1,1))
        self.A0inv = np.zeros((1,1))

    def assemble(self):
        r"""
        Assemble coarse correction matrix
        """
        c = sy.symbols('c')
        d = self.dc.pb.d
        BT0 = sy.Matrix([
            [                    sy.S(1),                    sy.S(0)],
            [(sy.S(1)-sy.S(2)*d)/sy.S(2),            sy.S(1)/sy.S(2)],
            [            sy.S(1)/sy.S(2),(sy.S(1)-sy.S(2)*d)/sy.S(2)],
            [                    sy.S(0),                    sy.S(1)]])
        BT0 = sy.Matrix(np.kron(BT0,BT0)*sy.S(2)/(sy.S(2)-d))
        
        B0 = sy.Matrix([
            [        sy.S(1),        sy.S(0)],
            [sy.S(1)/sy.S(2),sy.S(1)/sy.S(2)],
            [sy.S(1)/sy.S(2),sy.S(1)/sy.S(2)],
            [        sy.S(0),        sy.S(1)]])
        B0 = sy.Matrix(np.kron(B0,B0))
        
        Ind = np.matrix([[0,2],
                         [1,3]])
        Ind = np.block([[Ind+0*np.max(Ind)+0,Ind+2*np.max(Ind)+2],\
                        [Ind+1*np.max(Ind)+1,Ind+3*np.max(Ind)+3]])

        BT = sy.Matrix(sy.zeros(16,4))
        k = 0
        for j in range(4):
            for i in range(4):
                BT[Ind[i,j],:] = BT0[k,:]
                k +=1

        B = sy.Matrix(sy.zeros(16,4))
        k = 0
        for j in range(4):
            for i in range(4):
                B[Ind[i,j],:] = B0[k,:]
                k +=1

        for b in range(int(self.n**2/4)):
            self.RTs[16*b:16*b+16,4*b:4*b+4] = B

        for b in range(int(self.n**2/4)):
            self.Rs[4*b:4*b+4,16*b:16*b+16] = sy.transpose(BT)
                
    def nassemble(self,par):
        r"""
        Assemble coarse correction matrix
        """
        c = sy.symbols('c')
        d = self.dc.pb.d
        BT0 = sy.Matrix([
            [                    sy.S(1),                    sy.S(0)],
            [(sy.S(1)-sy.S(2)*d)/sy.S(2),            sy.S(1)/sy.S(2)],
            [            sy.S(1)/sy.S(2),(sy.S(1)-sy.S(2)*d)/sy.S(2)],
            [                    sy.S(0),                    sy.S(1)]])
        BT0 = np.matrix(sy.Matrix(np.kron(BT0,BT0)*sy.S(2)/(sy.S(2)-d)).subs(par))
        
        B0 = sy.Matrix([
            [        sy.S(1),        sy.S(0)],
            [sy.S(1)/sy.S(2),sy.S(1)/sy.S(2)],
            [sy.S(1)/sy.S(2),sy.S(1)/sy.S(2)],
            [        sy.S(0),        sy.S(1)]])
        B0 = np.matrix(sy.Matrix(np.kron(B0,B0)).subs(par))
        
        Ind = np.matrix([[0,2],
                         [1,3]])
        Ind = np.block([[Ind+0*np.max(Ind)+0,Ind+2*np.max(Ind)+2],\
                        [Ind+1*np.max(Ind)+1,Ind+3*np.max(Ind)+3]])

        BT = np.zeros((16,4))
        k = 0
        for j in range(4):
            for i in range(4):
                BT[Ind[i,j],:] = BT0[k,:]
                k +=1

        B = np.zeros((16,4))
        k = 0
        for j in range(4):
            for i in range(4):
                B[Ind[i,j],:] = B0[k,:]
                k +=1

        for b in range(int(self.n**2/4)):
            self.RT[16*b:16*b+16,4*b:4*b+4] = B

        for b in range(int(self.n**2/4)):
            self.R[4*b:4*b+4,16*b:16*b+16] = np.transpose(BT)

            
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
    
def MG(g,s,d,rlx):
    n = np.size(g)
    dc = DiscreteOperator(problem_=Diffusion2(basis_=TensorBasis(order_=1)),
                          n_=n,
                          periodic_=False)
    #dc.assemble()
    dc.nassemble({dc.bs.h:1./dc.n,dc.pb.d:d})
    dc.assemble_cellBJ({dc.bs.h:1./dc.n,dc.pb.d:d})
    cc = CoarseCorrection(discreteOperator_=dc,c_=0.5)
    cc.nassemble({dc.bs.h:1./dc.n,dc.pb.d:d})
    A = dc.An#np.matrix(dc.A.subs({dc.bs.h:1./dc.n,dc.pb.d:d}),dtype=np.float64)
    RT = cc.RT#np.matrix(cc.RTs.subs({dc.bs.h:1./dc.n,dc.pb.d:d}),dtype=np.float64)
    R = np.transpose(RT)
    Dinv = dc.Dc
    x = 0*g
    
    for i in range(s):
        x = x + rlx*Dinv.dot(g - A.dot(x))

    if (n > 2):
        x = x + RT.dot(MG(R.dot(g - A.dot(x)),s,d,rlx))
    else:
        dc0 = DiscreteOperator(problem_=Diffusion2(basis_=TensorBasis(order_=1)),
                               n_=int(n/2),
                               periodic_=False)
        dc0.assemble()
        A0 = np.matrix(dc0.A.subs({dc0.bs.h:1./dc0.n,dc0.pb.d:d}),dtype=np.float64)
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
    i = 3
    n = 2**i
    dc = DiscreteOperator(problem_=Diffusion2(basis_=TensorBasis(order_=1)),
                          n_=n,
                          periodic_=False)
    dc.assemble()
    A = np.matrix(dc.A.subs({dc.bs.h:1./dc.n,dc.pb.d:d}),dtype=np.float64)
    def func2(rlx):
        E = np.eye(4*n**2)-MG(A,1,d,rlx[0])
        return sorted(abs(np.real(np.linalg.eigvals(E))),reverse=True)[0]
    xmin,ffmin,dum1,dum2,dum3 = fmin(func2,np.array([1.]),ftol=0.000001,xtol=0.000001,full_output=True,disp=False)
    print(xmin[0],end=" ")
    print(ffmin)

n = 4
d = 1.5
rlx = 1.
dc = DiscreteOperator(problem_=Diffusion2(basis_=TensorBasis(order_=1)),
                      n_=n,
                      periodic_=False)
dc.nassemble({dc.bs.h:1./dc.n,dc.pb.d:d})
A = dc.An#np.matrix(dc.A.subs({dc.bs.h:1./dc.n,dc.pb.d:d}),dtype=np.float64)
#E = np.eye(4*n**2)-MG(A,1,d,rlx)
#print(sorted(abs(np.real(np.linalg.eigvals(E))),reverse=True)[0])


rhs = np.ones((4*dc.n**2,1))/float(4*dc.n**2)
x = np.zeros((4*dc.n**2,1))
normr = np.linalg.norm(rhs)
norm0 = normr
it = 0
while (normr/norm0 > 1.E-8):
    r = rhs - A.dot(x)
    normr = np.linalg.norm(r)
    x = x + MG(r,1,d,rlx)
    print(it,normr)
    it = it + 1
    if it > 40: break

# dc = DiscreteOperator(problem_=Diffusion2(basis_=TensorBasis(order_=1)),
#                       n_=4,periodic_=False)
# dc.assemble()
# cc = CoarseCorrection(discreteOperator_=dc,c_=1/2)
# cc.assemble(dc.pb.d)
# d = 1.4
# # print((5*d-2)/(3*d+2))
# a = 2*d/(2+3*d)
# dc.nassemble({dc.bs.h:1./dc.n,dc.pb.d:d})
# dc.assemble_cellBJ({dc.bs.h:1./dc.n,dc.pb.d:d})
# A = np.matrix(dc.A.subs({dc.bs.h:1./dc.n,dc.pb.d:d}),dtype=np.float64)
# RT = np.matrix(cc.RTs.subs({dc.bs.h:1./dc.n,dc.pb.d:d}),dtype=np.float64)
# R = np.matrix(cc.Rs.subs({dc.bs.h:1./dc.n,dc.pb.d:d}),dtype=np.float64)
# A0inv = np.linalg.inv(R.dot(A).dot(RT))
# a = 1.
# R = np.transpose(RT)/4
# Dinv = dc.Dc
# Id = np.eye(4*dc.n**2)
# M=(Id - RT.dot(A0inv).dot(R).dot(A)).dot(Id-a*Dinv.dot(A))
# print(sorted(abs(np.real(np.linalg.eigvals(M))),reverse=True)[0])


# Optimizing smoother and coarse solver in 1D
# dd = 1.5
# rlx = 2*dd**2/(2*dd**2+dd-1)
# cc = 0.5

# Optimizing smoother, coarse solver and interpolation in 1D
# dd = 1.51697830014707997232890254807433
# rlx = 0.90815413446701560826304190680793
# cc = 0.56460427612264228825165455619252

# Optimizing smoother and coarse solver in 2D
# dd = 1.569 
# rlx = 0.890
# cc = 0.5

# # Optimizing smoother only in 1D
# dd = 2
# rlx = 2/3
# cc = 0.5
# dc.nassemble({dc.pb.d:dd})
# dc.assemble_cellBJ({dc.pb.d:dd})
# cc = CoarseCorrection(discreteOperator_=dc,c_=cc)
# cc.nassemble()
# Id = np.eye(((dc.bs.p + 1)*dc.n)**2)
# A0inv = cc.A0inv
# R = cc.R
# RT = cc.RT
# An = dc.An
# Dinv = dc.Dc
# el = 0
# if (dc.periodic): el = 1
# M=rlx*Dinv + RT.dot(A0inv).dot(R).dot(Id - rlx*An.dot(Dinv))
# eigvals,eigvecs = np.linalg.eig(Id-M.dot(An))
# eigvals = np.real(eigvals)
# eigsort = np.array(sorted(eigvals,reverse=True))
# print(max(abs(eigsort)))
# mp.ylim([-0.7,0.7])
# mp.xlabel("#Eigenvalue")
# mp.ylabel("Eigenvalue")
# mp.plot(eigsort,'k,',markersize=0.00001)

# # Optimizing smoother and coarse solver in 1D
# dd = 1.5
# rlx = 2*dd**2/(2*dd**2+dd-1)
# cc = 0.5
# dc.nassemble({dc.pb.d:dd})
# dc.assemble_cellBJ({dc.pb.d:dd})
# cc = CoarseCorrection(discreteOperator_=dc,c_=cc)
# cc.nassemble()
# Id = np.eye(((dc.bs.p + 1)*dc.n)**2)
# A0inv = cc.A0inv
# R = cc.R
# RT = cc.RT
# An = dc.An
# Dinv = dc.Dc
# el = 0
# if (dc.periodic): el = 1
# M=rlx*Dinv + RT.dot(A0inv).dot(R).dot(Id - rlx*An.dot(Dinv))
# eigvals,eigvecs = np.linalg.eig(Id-M.dot(An))
# eigvals = np.real(eigvals)
# eigsort = np.array(sorted(eigvals,reverse=True))
# print(max(abs(eigsort)))
# mp.plot(eigsort,'g,',markersize=0.00001)

# # Optimizing smoother, coarse solver and interpolation in 1D
# dd = 1.51697830014707997232890254807433
# rlx = 0.90815413446701560826304190680793
# cc = 0.56460427612264228825165455619252
# dc.nassemble({dc.pb.d:dd})
# dc.assemble_cellBJ({dc.pb.d:dd})
# cc = CoarseCorrection(discreteOperator_=dc,c_=cc)
# cc.nassemble()
# Id = np.eye(((dc.bs.p + 1)*dc.n)**2)
# A0inv = cc.A0inv
# R = cc.R
# RT = cc.RT
# An = dc.An
# Dinv = dc.Dc
# el = 0
# if (dc.periodic): el = 1
# M=rlx*Dinv + RT.dot(A0inv).dot(R).dot(Id - rlx*An.dot(Dinv))
# eigvals,eigvecs = np.linalg.eig(Id-M.dot(An))
# eigvals = np.real(eigvals)
# eigsort = np.array(sorted(eigvals,reverse=True))
# print(max(abs(eigsort)))
# mp.plot(eigsort,'b,',markersize=0.00001)

# # Optimizing smoother, coarse solver and interpolation in 2D
# dd = 1.684 
# rlx = 0.941
# cc = 0.596 
# dc.nassemble({dc.pb.d:dd})
# dc.assemble_cellBJ({dc.pb.d:dd})
# cc = CoarseCorrection(discreteOperator_=dc,c_=cc)
# cc.nassemble()
# Id = np.eye(((dc.bs.p + 1)*dc.n)**2)
# A0inv = cc.A0inv
# R = cc.R
# RT = cc.RT
# An = dc.An
# Dinv = dc.Dc
# el = 0
# if (dc.periodic): el = 1
# M=rlx*Dinv + RT.dot(A0inv).dot(R).dot(Id - rlx*An.dot(Dinv))
# eigvals,eigvecs = np.linalg.eig(Id-M.dot(An))
# eigvals = np.real(eigvals)
# eigsort = np.array(sorted(eigvals,reverse=True))
# print(max(abs(eigsort)))
# mp.plot(eigsort,'r,',markersize=0.00001)

# mp.savefig("plot.pdf")

# rhs = np.ones(((dc.bs.p + 1)*dc.n)**2)/float(((dc.bs.p + 1)*dc.n)**2)
# x = np.zeros(((dc.bs.p + 1)*dc.n)**2)
# normr = np.linalg.norm(rhs)
# norm0 = normr
# it = 0
# while (normr/norm0 > 1.E-8):
#     r = rhs - An.dot(x)
#     normr = np.linalg.norm(r)
#     x = x + M.dot(r)
#     print(it,normr/norm0)
#     it = it + 1
#     if it > 40: break

# sol = x

# # Define vertices
# x = np.zeros(4*dc.n**2)
# y = np.zeros(4*dc.n**2)
# z = np.zeros(4*dc.n**2)
# s = np.zeros(4*dc.n**2)
# for j in range(dc.n):
#     for i in range(dc.n):
#         x[4*(i+dc.n*j)+0], y[4*(i+dc.n*j)+0], z[4*i+0] = (i+0.0),(j+0.0), 0.0
#         x[4*(i+dc.n*j)+1], y[4*(i+dc.n*j)+1], z[4*i+1] = (i+1.0),(j+0.0), 0.0
#         x[4*(i+dc.n*j)+2], y[4*(i+dc.n*j)+2], z[4*i+2] = (i+1.0),(j+1.0), 0.0
#         x[4*(i+dc.n*j)+3], y[4*(i+dc.n*j)+3], z[4*i+3] = (i+0.0),(j+1.0), 0.0

# for j in range(dc.n):
#     for i in range(dc.n):
#         s[4*(i+dc.n*j)+0] = sol[4*dc.B[i,j]+0]
#         s[4*(i+dc.n*j)+1] = sol[4*dc.B[i,j]+1]
#         s[4*(i+dc.n*j)+3] = sol[4*dc.B[i,j]+2]
#         s[4*(i+dc.n*j)+2] = sol[4*dc.B[i,j]+3]
       
# x=x/float(dc.n)
# y=y/float(dc.n)

# # Define connectivity or vertices that belongs to each element
# conn = np.zeros(4*dc.n**2)
# for i in range(4*dc.n**2):
#     conn[i] = i

# # Define offset of last vertex of each element
# offset = np.zeros(dc.n**2)
# for i in range(dc.n**2):
#     offset[i] = 4*(i+1)

# # Define cell types
# ctype = np.zeros(dc.n**2)
# for i in range(dc.n**2):
#     ctype[i] = VtkQuad.tid

# unstructuredGridToVTK("unstructured", x, y, z, connectivity = conn, offsets = offset,
#                       cell_types = ctype, cellData = None,
#                       pointData = {"data":s})

#print("{"+str(dd)+","+str(rlx)+"},"+str(sorted(abs(np.real(np.linalg.eigvals(M))),reverse=True)[el]))
#print(np.array(sorted(abs(np.real(np.linalg.eigvals(M))),reverse=True)))
# y = np.array(sorted(abs(np.real(np.linalg.eigvals(M))),reverse=True)[1:])
# g = np.linspace(0,1,255)
# mp.plot(g,y)
# mp.show()

# def func(dd):
#     dc.nassemble({dc.pb.d:dd})
#     dc.assemble_cellBJ({dc.pb.d:dd})
#     cc = CoarseCorrection(discreteOperator_=dc,c_=0.564604)
#     cc.nassemble()
#     Id = np.eye(4*dc.n**2)
#     A0inv = cc.A0inv
#     R = cc.R
#     RT = cc.RT
#     An = dc.An
#     Dinv = dc.Dc
#     el = 0
#     if (dc.periodic): el = 1
#     def func2(rlx):
#         # M=sc.sparse.bsr_matrix( (Id - RT.dot(A0inv).dot(R).dot(An)).dot(Id-rlx*Dinv.dot(An)) , blocksize=(4,4))
#         M=(Id - RT.dot(A0inv).dot(R).dot(An)).dot(Id-rlx*Dinv.dot(An))
#         # return abs(np.real(sc.sparse.linalg.eigs(M, 2, which='LM')[0][el]))
#         return sorted(abs(np.real(np.linalg.eigvals(M))),reverse=True)[el]

#     xmin,ffmin,dum1,dum2,dum3 = fmin(func2,np.array([1]),ftol=1.E-8,xtol=1.E-8,full_output=True,disp=False)
#     # print("{"+str(dd)+","+str(xmin[0])+"},"+str(ffmin))
#     print("{"+str(xmin[0])+","+str(ffmin)+"},")#+str(dd))
#     # print(np.array(sorted(abs(np.real(np.linalg.eigvals(M))),reverse=True)))

# # n = 20
# # for i in range(1,n+1):
# #     func(1.+float(i)/float(n))
    
# for i in range(0,51):
#     func(1.001+((5.-1.001)*(2.**(float(i)/50.) - 1.)))


# dd = 2.
# dc.nassemble({dc.pb.d:dd})
# print(np.linalg.inv(dc.An).dot(np.ones((4*dc.n**2))/(4*dc.n**2)))

# dd = 2.
# dc.nassemble({dc.pb.d:dd})
# M = dc.An
# for i in range(int(np.sqrt(M.size))):
#     for j in range(int(np.sqrt(M.size))):
#         if abs(M[i,j]) > 1.E-10:
#             print(format(float(M[i,j]),' .3e'), end = " ")
#         else:
#             print("           ", end = "")
#     print("")

# dd = 2
# dc.nassemble({dc.pb.d:dd})
# M = dc.An
# for i in range(4*dc.n**2):
#     for j in range(4*dc.n**2):
#         if abs(M[i,j]) > 1.E-10:
#             print(format(float(M[i,j]),' .3e'), end = " ")
#         else:
#             print("           ", end = "")
#     print("")
            
