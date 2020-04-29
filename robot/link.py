###
#   Based on Robotics Toolbox for Matlab created by Peter Corke and forked
#       from the repository created by Aditya Dua to continue the port of
#       the toolbox for python
#
#   Author: Rafael Torres
#
#   31 March 2020
###

import numpy as np
from math import sin, cos
import robopy.robot.transforms as tr
#from util import *

class Link:
    """
    Link object class.
    """

    def __init__(self, 
                 alpha=0, A=0, theta=None, D=None, sigma=None, offset=0, mdh=None, qlim=None, 
                 m=0, r=None, I=None):
        """
        Initializes the link object.
        
        Kinematic params:
            :param alpha:  Link twist
            :param A:      Link length
            :param theta:  Joint angle
            :param D:      Link offset
            :param sigma:  [0, "Revolute" or 'R'] if revolute, [1, "Prismatic" or 'P'] if prismatic
            :param offset: Joint variable offset
            :param mdh:    0 if standard D&H, else 1
            :param qlim:   Joint variable limits [min max]

        Dynamic params:
            :param m: Link mass
            :param r: Link COG with respect to link coordinate frame, in format 1x3 [x, y, z]
            :param I: Link inertia matrix, symmetric 3x3, about link COG
        """

        # Defaults
        self.alpha = 0
        self.A = 0
        self.theta = 0
        self.d = 0
        self.sigma = None # Type will be retrieved from parameters parsed
        self.offset = 0
        self.mdh = 0
        self.qlim = []
        self.m = 0
        self.r = np.zeros(3)
        self.I = np.zeros( (3, 3) ) 

        ### Type checking ###
        if sigma != None:
            if isinstance(sigma, str):
                sigma = sigma.lower()
                
                if sigma == "revolute" or sigma == 'r':
                    self.sigma = 0
                elif sigma == "prismatic" or sigma == 'p':
                    self.sigma = 1
                else:
                    raise ValueError
            elif isinstance(sigma, int):
                if sigma not in [0, 1]:
                    raise ValueError

                self.sigma = sigma
            else:
                raise AttributeError

        if(self.sigma == 0):
            assert ( theta==None ), "'Theta' cannot be specified for a Revolute link!"
        elif(self.sigma == 1):
            assert ( D==None ), "'d' cannot be specified for a Prismatic link!"

        assert( theta!=None or D!=None ), "Cannot specify 'd' and 'theta'"

        ### D&H parameters check
        if isinstance(alpha, (int, float)) and isinstance(A, (int, float)):
            self.alpha = alpha
            self.a = A
        else:
            raise AttributeError

        if theta != None:
            if isinstance(theta, (int, float)):
                self.theta = theta
                if sigma == None:   # Constant value of theta means it must be prismatic
                    sigma = 1
            else:
                raise AttributeError
        
        if D != None:
            if isinstance(D, (int, float)):
                self.d = D
                if sigma == None:   # Constant value of D means it must be revolute
                    sigma = 0
            else:
                raise AttributeError

        if isinstance(offset, (int, float)):
            self.offset = offset
        else:
            raise AttributeError

        if qlim:
            if isinstance(qlim, list) and isinstance(qlim[0], (int, float)) and len(qlim) == 2:
                self.qlim = qlim
            else:
                raise AttributeError

        ### Mass checks
        if isinstance(m, (int, float)):
            self.m = m
        else:
            raise AttributeError

        ### COG position check
        if r:
            if isinstance(r, list) and isinstance(r[0], (int, float)) and len(r) == 3:
                self.r = np.array(r)
            else:
                raise AttributeError

        ### Inertia check
        if I:
            if isinstance(I, list) and isinstance(I[0], list) and isinstance(I[0][0], (int, float)) and len(I) == 3 and len(I[0]) == 3:
                self.I = np.array(I)
            else:
                raise AttributeError

        ### D&H Convention checking ###
        if mdh == None:
            self.mdh = 0    # Standard: Classis D&H
        else:
            if isinstance(mdh, str):
                mdh = mdh.lower()
                
                if mdh == "classic" or mdh == 'c':
                    self.mdh = 0
                elif mdh == "modified" or mdh == 'm':
                    self.mdh = 1
                else:
                    raise ValueError
            elif isinstance(mdh, int):
                if mdh not in [0, 1]:
                    raise ValueError

                self.mdh = mdh
            else:
                raise AttributeError

    def isrevolute(self):
        return self.sigma == 0

    def isprismatic(self):
        return self.sigma == 1

    def Tm(self, q):
        """
        Calculates a transformation matrix for the current link
            - For standard DH parameters, this is from the previous frame to the current
            - For modified DH parameters, this is from the current frame to the previous

        Note:
            - The link offset parameter is added to 'q' before computation of the matrix

        :param q: Value of the D&H non fixed parameter, 'theta' for revolute links and 'd' for prismatic links

        :returns 4x4 numpy.ndarray representing the Transformation Matrix 
        """

        sa = sin(self.alpha)
        ca = cos(self.alpha)
        a = self.a

        q = q + self.offset

        if self.isrevolute():
            st = sin(q)
            ct = cos(q)
            d = self.d
        else:
            st = sin(self.theta)
            ct = cos(self.theta)
            d = q

        if self.mdh == 0:

            return np.array([[ct, -st*ca,  st*sa, a*ct],
                             [st,  ct*ca, -ct*sa, a*st],
                             [ 0,     sa,     ca,  d  ],
                             [ 0,      0,      0,  1  ]])

        else:

            return np.array([[   ct,   -st,   0,   a  ],
                             [st*ca, ct*ca, -sa, -sa*d],
                             [st*sa, ct*sa,  ca,  ca*d],
                             [    0,     0,   0,   1  ]])

class Revolute(Link):
    """
    Revolute Link object class.
    """

    def __init__(self, 
                 alpha=0, A=0, theta=None, D=None, offset=0, mdh=None, qlim=None, 
                 m=0, r=None, I=None):
        """
        Initializes revolute link.
        Check Link class to understand params
        """
        super().__init__(alpha=alpha, A=A, theta=theta, D=D, sigma="Revolute", offset=offset, mdh=mdh, qlim=qlim,
                         m=m, r=r, I=I)
        pass

class Prismatic(Link):
    """
    Prismatic Link object class.
    """

    def __init__(self, 
                 alpha=0, A=0, theta=None, D=None, offset=0, mdh=None, qlim=None, 
                 m=0, r=None, I=None):
        """
        Initializes Prismatic link.
        Check Link class to understand params
        """
        super().__init__(alpha=alpha, A=A, theta=theta, D=D, sigma="Prismatic", offset=offset, mdh=mdh, qlim=qlim,
                         m=m, r=r, I=I)
        pass

class SerialLink:
    """
    SerialLink object class.
    """

    def __init__(self, arg=None, gravity=None, base=None, tool=None, name=""):
        """
        Creates a SerialLink object.
        :param links:   List of Link objects that will constitute the SerialLink object.
        :param gravity: Direction of gravity [gx, gy, gz]
        :param base:    Pose of robot's base (4x4)
        :param tool:    Robot's tool transform, with respect to last link coordinate frame
        """

        # Defaults
        self.links = []

        self.gravity = np.array([0, 0, 9.81])
        self.base = np.identity(4)
        self.tool = np.identity(4)
        self.name = "noname"

        self.mdh = -1

        if isinstance(arg, list):
            if isinstance(arg[0], Link):
                self.links = arg
            else:    
                raise AttributeError
        else:
            raise AttributeError    # Other initialization methods not implemented yet

        # Check links for D&H convention
        for link in self.links:
            if( (link.mdh != self.mdh) and (self.mdh != -1) ):
                raise ValueError("Links have mixed D&H conventions!")
            else:
                self.mdh = link.mdh

        if name:
            self.name = name

        # Gravity checks
        if gravity != None:
            assert (     isinstance(gravity, list) 
                     and len(gravity)==3 
                     and isinstance(gravity[0], (int, float)) ), "Gravity must be a 1x3 list of int or floats"
            self.gravity = np.array(gravity)

        # Base transformation checks
        if base != None:
            assert (     isinstance(base, list) 
                     and len(base)==4 
                     and isinstance(base[0], list)
                     and len(base[0])==4
                     and isinstance(base[0][0], (int, float))), "Base must be a 4x4 list of int or floats"
            self.base = np.array(base)

        # Tool transformation checks
        if tool != None:
            assert (     isinstance(tool, list) 
                     and len(tool)==4 
                     and isinstance(tool[0], list)
                     and len(tool[0])==4
                     and isinstance(tool[0][0], (int, float))), "Tool must be a 4x4 list of int or floats"
            self.tool = np.array(tool)

    def append_link(self, link):
        if isinstance(link, Link):
            if( self.mdh != -1 ):
                assert (link.mdh == self.mdh), "Links have mixed D&H conventions!" 
            else:
                self.mdh = link.mdh
            
            self.links.append(link)
        else:
            raise AttributeError

    def __add__(self, link):
        self.append_link(link)

        return self

    def __len__(self):
        return len(self.links)

    def display(self):
        
        ### SerialLink info ###
        print('\n' + self.name + ": " + str(len(self)) + " axis", end="")
        if(self.mdh==0):
            print(", standard D&H\n")
        elif(self.mdh==1):
            print(", modified D&H\n")
        else:
            print('\n')

        ### Links info ###
        space_to_j = 3
        space_to_params = 10

        params=["theta", 'd', 'a', "alpha", "offset"]

        header_str = '+' + ('-'*space_to_j) + len(params)*('+' + ('-'*space_to_params)) + '+' # +---+----------+ (...)

        # Strings formatting:
        #   ^x -> Center in a block of 'x' size
        #   .y -> Limit to a block of 'x' size
        j_format =     ("{:^" + str(space_to_j) + '.' + str(space_to_j) + '}')           # "{:^3.3}"   
        param_format = ("{:^" + str(space_to_params) + '.' + str(space_to_params) + '}') # "{:^10.10}"
        
        # Header display
        print(header_str)

        print('|' + j_format.format('j'), end="")
        
        for param in params:
            print('|' + param_format.format(param), end="")

        print('|')        
        print(header_str)
        # End of header display

        for j, link in enumerate(self.links):
            
            # Index printing
            print('|' + j_format.format( str(j+1) ), end="")

            # Parameters printing
            for param in params:
                print('|' + param_format.format( str( getattr(link, param) ) ), end="")

            print('|')

        print(header_str + '\n') # But isn't this the footer? ¯\_(u.u)_/¯

        ### Gravity, base and tool info ###
        grav_format = ("{:>4.4}")
        base_format = ("{:>4.4}")
        tool_format = ("{:>4.4}")

        grav_title = "grav = "
        base_title = "  base ="
        tool_title = "  tool ="
        for row in range( len(self.base) ):
            try:
                print(grav_title + grav_format.format( str(self.gravity[row]) ), end="")
            except IndexError:
                # Gravity only has 3 rows while base and tool transforms have 4, so, print placeholder
                print(grav_title + ' '*len(grav_format.format( str(self.gravity[0]) )), end="")

            print(base_title, end="")
            for value in self.base[row]:
                print(' ' + base_format.format( str(value) ), end="")

            print(tool_title, end="")
            for value in self.tool[row]:
                print(' ' + tool_format.format( str(value) ), end="")

            print()

            if row == 0:
                # After first row titles are no longer necessary, substitute for placeholder
                grav_title = ' '*len(grav_title)
                base_title = ' '*len(base_title)
                tool_title = ' '*len(tool_title)

        print()

    def rne(self, Q, QD=None, QDD=None, PL=None):
        """
        Verifies input arguments and calculates torques using recursive Newton-Euler equations
        indirectly by calling the adequate methods

        :param Q:   Array of joint positions
        :param QD:  Array of joint velocities -> Qderivative = V
        :param QDD: Array of joint accelerations -> Vderivative
        :param PL:  Array describing external forces (payload) on the end effector [Fx Fy Fz Nx Ny Nz]

        :return:    Array of joint torques
        """
        assert len(self), "No links detected!"

        if not isinstance(Q, list):
            raise AttributeError 
        Q = np.array(Q)
        assert Q.shape == (len(self),), "Wrong dimension!"

        if QD != None:
            if not isinstance(QD, list):
                raise AttributeError
            QD = np.array(QD)
            assert QD.shape == (len(self),), "Wrong dimension!"
        else:
            QD = np.zeros(len(self))

        if QDD != None:
            if not isinstance(QDD, list):
                raise AttributeError
            QDD = np.array(QDD)
            assert QDD.shape == (len(self),), "Wrong dimension!"
        else:
            QDD = np.zeros(len(self))

        if PL != None:
            if not isinstance(PL, list):
                raise AttributeError
            PL = np.array(PL)
            assert PL.shape == (6,), "Wrong dimension!"
        else:
            PL = np.zeros(6)

        if self.mdh == 0:
            return self.__rne_dh(Q, QD, QDD, PL)
        elif self.mdh == 1:
            return self.__rne_mdh(Q, QD, QDD, PL)

    def __rne_dh(self, q, qd=None, qdd=None, pl=None):
        """
        Calculates torques for joints using Recursive Newton-Euler equations
        Assumes classic D&H notation

        Invoque this method indirectly calling the 'rne' method
        """
        print("Calculating torques...")

    def __rne_mdh(self, q, qd=None, qdd=None, pl=None):
        """
        Calculates torques for joints using Recursive Newton-Euler equations
        Assumes modified D&H notation

        Invoque this method indirectly calling the 'rne' method

        TODO: Include payload and external forces
        """

        # Set debug to
        #   0 -> No messages
        #   1 -> Display results of outwards and inwards recursion
        debug = 1

        # Initial setup
        z = np.array([0, 0, 1]) # Initial 'Z' points "up", 'X' and 'Y' constitute top plane

        w = np.zeros(3)     # Base has 0 joint angle,
        wd = np.zeros(3)    #          0 joint speed and
        vd = self.gravity   #          fictitious upwards acceleration
        
        tau = np.zeros(len(self))
        F = np.zeros( (len(self), 3) )
        N = np.zeros( (len(self), 3) )

        if debug:
            print("\nOutwards iterations")

        # Outwards iteration
        for j, link in enumerate(self.links):
            Tm = link.Tm(q[j])

            if j == 0:
                # Include base transform
                Tm = self.base * Tm

            R = tr.t2r(Tm)
            R = np.linalg.inv(R) # Inverse
            
            P = (Tm)[0:-1, -1]

            Pc = link.r
            m = link.m
            I = link.I

            #
            # Trailing underscore means new value
            #
            if link.isrevolute():
                w_  =   np.matmul(R, w) + qd[j]*z
                wd_ =   np.matmul(R, wd) + np.cross(np.matmul(R, w), qd[j]*z) + qdd[j]*z
                vd_ =   np.matmul(R, ( np.cross(wd, P) + np.cross(w, np.cross(w, P) ) + vd ) ) 
            elif link.isprismatic():
                w_  =   np.matmul(R, w)
                wd_ =   np.matmul(R, wd)
                vd_ = ( np.matmul(R, ( np.cross(wd, P) + np.cross(w, np.cross(w, P) ) + vd ) ) + 
                        2*np.cross(w_, qd[j]*z) + qdd[j]*z )

            vdc = np.cross(wd_, Pc) + np.cross(w_, np.cross(w_, Pc) ) + vd_
            
            F[j] = m*vdc
            N[j] = np.matmul(I, wd_) + np.cross(w_, np.matmul(I, w_) )

            # Update variables
            w = w_
            wd = wd_
            vd = vd_

            if debug:
                print("\nLink " + str(j+1) + ':')
                print("w = " + str(w))
                print("wd = " + str(wd))
                print("vd = " + str(vd))
                print("vdc = " + str(vdc))
                print("\nF = " + str(F[j]))
                print("N = " + str(N[j]))

        f = pl[:3]
        n = pl[3:]

        # Initial transform matrix
        Tm = self.tool

        if debug:
            print("\nInwards iterations")

        # Inwards iteration
        for j, link in reversed( list( enumerate(self.links) ) ):
            
            R = tr.t2r(Tm)
            P = (Tm)[0:-1, -1]
            
            Pc = link.r 

            #
            # Trailing underscore means new value
            #
            f_ = np.matmul(R, f) + F[j]
            n_ = N[j] + np.matmul(R, n) + np.cross(Pc, F[j]) + np.cross(P, np.matmul(R, f) )

            if link.isrevolute():
                tau[j] = np.matmul(n_, z)
            elif link.isprismatic():
                tau[j] = np.matmul(f_, z)

            # Update variables
            f = f_
            n = n_
            Tm = link.Tm(q[j])

            if debug:
                print("\nLink " + str(j+1) + ':')
                print("f = " + str(f))
                print("n = " + str(n))

        if debug:
            print()

        return tau
