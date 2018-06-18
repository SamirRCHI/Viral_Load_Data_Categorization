# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 09:55:47 2017

Company: University of Rochester Medical Center
Team: Rochester Center for Health Informatics
Supervisor: Dr. Martin Zand (martin_zand@urmc.rochester.edu)
Author: Samir Farooq (samir_farooq@urmc.rochester.edu)

Documentation Available on GitHub
"""
import numpy as np
from copy import deepcopy as dc
import time
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as Rect
from matplotlib.patches import Ellipse
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError
import colorsys

def example_B(center,radius,with_data = 100, with_plot = True):
    B = []
    for i in range(3):
        d = random.random()*2*np.pi
        B.append([center[0]+radius*np.cos(d),center[1]+radius*np.sin(d)])
    if with_data:
        for j in range(with_data):
            rd = random.random()
            while (rd > 0.95) or (rd < 0.1):
                rd = random.random()
            s_rd,d = radius*rd,random.random()*2*np.pi
            B.append([center[0]+s_rd*np.cos(d),center[1]+s_rd*np.sin(d)])
        if with_plot:
            t = np.linspace(0,2*np.pi,1000)
            x,y = center[0] + radius*np.cos(t),center[1] + radius*np.sin(t)
            pB = np.array(B)
            fig = plt.figure(figsize=(5,5))
            ax = fig.add_subplot(111)
            ax.scatter(pB[0:3,0],pB[0:3,1],color='r')
            ax.scatter(pB[3:,0],pB[3:,1],color='k')
            ax.scatter(center[0],center[1],marker=r'+',s=81,color='b')
            ax.plot(x,y,'--',color='b')
            ax.set_xlim(left=center[0]-radius-0.1*radius,
                        right=center[0]+radius+0.1*radius)
            ax.set_ylim(bottom=center[1]-radius-0.1*radius,
                        top=center[1]+radius+0.1*radius)
            plt.show()
    return np.array(B)

def example_threedim(C,r,points=100):
    B = []
    for i in range(4):
        d = random.random()*2*np.pi
        phi = random.random()*2*np.pi
        x = r*np.sin(d)*np.cos(phi)
        y = r*np.sin(d)*np.sin(phi)
        z = r*np.cos(d)
        B.append([C[0]+x,C[1]+y,C[2]+z])
    for j in range(points):
        rd = random.random()
        while (rd > 0.95) or (rd < 0.1):
            rd = random.random()
        th,phi = random.random()*2*np.pi,random.random()*2*np.pi
        x = r*rd*np.sin(th)*np.cos(phi)
        y = r*rd*np.sin(th)*np.cos(phi)
        z = r*rd*np.cos(th)
        B.append([C[0]+x,C[1]+y,C[2]+z])
    return np.array(B)

def solve_center(B):
    a,b = [],[]
    for Bi in B:
        m,constant = [],0.0
        for e in Bi:
            m.append(-2*e)
            constant += -(e**2)
        m.append(1.0)
        b.append(constant),a.append(np.array(m))
    try: 
        R = np.linalg.solve(np.array(a),np.array(b))
    except np.linalg.LinAlgError:
        return ()
    r = np.sqrt(-(R[-1]-sum(R[:-1]**2)))
    return R[:-1],r

def euc(x,y):
    return np.sqrt(sum((x-y)**2))

def minidisk(Data):
    return sed(set(range(len(Data))),set(),Data,len(Data[0]))

def sed(P,R,Data,dimensions):
    if (len(P) == 0) or (len(R) == dimensions+1):
        if len(R) < dimensions+1: D = ()
        else: D = solve_center(Data[list(R)])
    else:
        p = random.sample(P,1)[0]
        D = sed(P - {p}, R, Data, dimensions)
        if (D == ()) or (euc(Data[p],D[0]) > D[1]):
            D = sed(P - {p}, R.union({p}), Data, dimensions);
    return D

def eucM(x,M,keep_track = False):
    if not keep_track: return [euc(x,x2) for x2 in M]
    dm,m,a = 0,0,0
    for x2 in M:
        d = euc(x,x2)
        if d > dm: 
            m = a if keep_track == 'arg' else d
            dm = d
        a += 1
    return m

def affine_coeffs(T,p):
    data_coeffs = []
    for i in range(len(T[0])):
        data_coeffs.append(T.T[i])
    data_coeffs.append([1]*len(T))
    b = list(p) + [1]
    coeffs = np.linalg.solve(data_coeffs,b)
    return coeffs

def determine_eq(points,MAX_rand = 1):
    dim,M,coeffs = len(points[0]),[],np.array([])
    while len(points) < dim:
        new_rand_point = [MAX_rand*random.random() for i in range(dim)]
        points = np.append(points, [new_rand_point], axis = 0)
    for i in range(1,len(points)): M.append(points[i]-points[0])
    for i in range(dim):
        coeffs=np.append(coeffs,((-1)**i)*np.linalg.det(np.delete(M,i,axis=1)))
    return coeffs,sum(coeffs*points[0])

def circumcenter(T):
    if len(T) == 1: return T[0]
    elif len(T) == 2: return T[0] + (T[1] - T[0])/2.0
    dim,M,b = len(T[0]),[],[]
    MAX = 1 if len(T) == dim else np.max(T)
    for t in T: M.append(np.append(t*-2,1.0)),b.append(sum(-(t**2)))
    for i in range(dim-len(T)+1):
        coeffs,intercept = determine_eq(T,MAX)
        M.append(np.append(coeffs,0.0)),b.append(intercept)
    R = np.linalg.solve(M,b)
    r = np.sqrt(-(R[-1]-sum(R[:-1]**2)))
    return R[:-1]

def hyperplanes(old,new,lenT):
    dim,M,b = len(old),[],[]
    if (dim == 2) and (lenT == 1):
        vector = new-old
        coeffs = np.array([vector[1],-vector[0]])
        M.append(np.append(coeffs,0.0)),b.append(sum(coeffs*old))
        return np.array(M),np.array(b)
    MAX,P = max([max(old),max(new)]),np.array([old,new])
    for i in range(dim-lenT):
        coeffs,intercept = determine_eq(P)
        M.append(np.append(coeffs,0.0)),b.append(intercept)
    return np.array(M),np.array(b)
        
def walking_distance(old_center,new_center,points,T):
    M2,b2 = hyperplanes(old_center,new_center,len(T))
    e,V,min_e,C,pos,i = [],new_center-old_center,float('inf'),False,False,0
    for point in points:
        M,b = [],[]
        for t in T:
            M.append(np.append(-2*t,1.0))
            b.append(sum(-(t**2)))
        M = np.append(M,[np.append(-2*point,1.0)],axis=0)
        b = np.append(b,sum(-(point**2)))
        if len(M2) != 0:
            M,b = np.append(M,M2,axis=0),np.append(b,b2)
        Ri = np.linalg.solve(M,b)
        nV = (Ri[:-1]-old_center)/V # Measures overshooting or undershooting
        ei = nV[0] if ((nV[0] > 0) and (nV[0] < 1)) else float('inf')
        if ei < min_e: min_e,C,pos = ei,Ri[:-1],i
        i += 1
    return C,pos

def in_hull(p, hull):
    if len(hull) <= len(p): return False
    hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def equi_color(n,start = 0):
    mx = 1.0
    if start == 'random':
        start = mx*random.random()
    distance = mx/n
    rgbs = []
    for i in range(n):
        hue = start % mx
        start += distance
        rgbs.append(list(colorsys.hsv_to_rgb(hue, mx, mx)))
    return rgbs

def bubble(S,rand_c='rand',plot = True):
    if plot:
        if len(S[0]) > 2: 
            print("Plotting only works with 2-D, will not plot!")
            plot = False
    if plot:
        fig = plt.figure(figsize=(8.5,8.5))
        ax = fig.add_subplot(111)
        t = np.linspace(0,2*np.pi,1000)
        cos,sin = np.cos(t),np.sin(t)
        tot_col = 5
        C = equi_color(tot_col)
    if rand_c == 'rand': rand_c = random.sample(range(len(S)),1)[0]
    c = S[rand_c]
    if plot: print(rand_c)
    m_T = eucM(c,S,'arg')
    T = np.array([S[m_T]])
    S2 = np.delete(S,m_T,axis=0)
    if plot: 
        ax.scatter(S.T[0],S.T[1],marker='+',s=81,color='k')
        r = euc(S[m_T],c)
        ax.plot(c[0]+r*cos,c[1]+r*sin,'--k')
        i = 0
    while not in_hull(c, T):
        if len(T) > len(c):
            affineC = np.argmin(affine_coeffs(T,c))
            S2 = np.append(S2, np.array([T[affineC]]),axis=0)
            T = np.delete(T,affineC,axis=0)
        new_c,p = walking_distance(c,circumcenter(T),S2,T)
        if type(p) is bool:
            break
        if plot:
            clr = 'k' if i == 0 else C[(i-1)%tot_col]
            CC = circumcenter(T)
            ax.plot([c[0],CC[0]],[c[1],CC[1]],'--',lw=0.5,color=clr)
            ax.arrow(c[0],c[1],new_c[0]-c[0],new_c[1]-c[1],color=C[i%tot_col])
            ax.scatter(new_c[0],new_c[1],color=C[i%tot_col],s=81)
            r = euc(S2[p],new_c)
            ax.plot(new_c[0]+r*cos,new_c[1]+r*sin,'--',color=C[i%tot_col])
        T = np.append(T,[S2[p]],axis=0)
        S2 = np.delete(S2,p,axis=0)
        c = new_c
        if plot: i += 1
    if plot: plt.show()
    return c,eucM(c,S,True)

def iter_bubble(S):
    Cs,rs = [],[]
    for i in range(len(S)):
        C,r = bubble(S,i,False)
        Cs.append(C),rs.append(r)
    i = np.argmin(rs)
    return Cs[i],rs[i],i

def warm_start_bubble(S):
    R = np.arange(len(S))
    maxs=[eucM(S[i],S[R!=i],True) for i in R]
    c_pos = np.argmin(maxs)
    return bubble(S,c_pos,False)

def net_direction(P,A,k,C):
    push = np.array([0.0 for i in range(len(P[0]))])
    pull = k*(A - C)
    for i in range(len(P)):
      p = P[i]
      V = C - p
      d = np.sqrt(sum(V**2));
      push += (1/(d**2)) * (1/d)*V;
    return push + pull;

def push_and_pull(Push_Points,Pull_Points,TOL=10.0**-6):
    A,k = sum(Pull_Points)/len(Pull_Points),len(Push_Points)
    C,i,E = [dc(A)],0,[]
    try:
        while (i == 0) or (E[i-1] > TOL):
            i += 1
            C.append(C[i-1]+(1.0/(k*3))*net_direction(Push_Points,A,k,C[i-1]))
            E.append(euc(C[i-1],C[i]))
            if len(E) > 2:
                if E[i-1] > E[i-2]:
                    print(E[i-1],E[i-2])
    except KeyboardInterrupt:
        fig = plt.figure(figsize=(15.5,8.5))
        ax = fig.add_subplot(111)
        ax.plot(range(len(E)),E)
        raise KeyboardInterrupt
    return C[i]

def nPoly_centroid(Data,return_hull=False):
    try:
        hull = ConvexHull(Data)
    except QhullError:
        hull = ConvexHull(Data,False,'QJ')
    V = hull.points[hull.vertices]
    T,W,C = Delaunay(V),[],np.array([0.0 for i in range(len(V.T))])
    for m in range(len(T.simplices)):
        sp = V[T.simplices[m,:],:]
        try:
            convex_hull = ConvexHull(sp)
        except QhullError:
            convex_hull = ConvexHull(sp,False,'QJ')
        W.append(convex_hull.volume)
        mean = np.array([0.0 for i in range(len(V.T))])
        for i in range(len(sp.T)): mean[i] = np.mean(sp.T[i])
        C += convex_hull.volume * mean
    if return_hull:
        return C / sum(W) , hull
    return C / sum(W)

def get_brdpt(m,b,xside,ymin,ymax):
    y = m*xside + b
    if y < ymin:
        y = ymin
        x = (ymin - b)/m
    elif y > ymax:
        y = ymax
        x = (ymax - b)/m
    else:
        x = xside
    return x,y

def get_line(xmin,xmax,ymin,ymax,eq):
    m = -eq[0]/eq[1]
    b = -eq[2]/eq[1]
    x1,y1 = get_brdpt(m,b,xmin,ymin,ymax)
    x2,y2 = get_brdpt(m,b,xmax,ymin,ymax)
    return [x1,x2],[y1,y2]

def plot_Poly(Data):
    fig = plt.figure(figsize=(15.5,8.5))
    ax = fig.add_subplot(111)
    ax.scatter(Data.T[0],Data.T[1])
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    hull = ConvexHull(Data)
    for eq in hull.equations:
        X,Y = get_line(xmin,xmax,ymin,ymax,eq)
        ax.plot(X,Y,'--')
    C = nPoly_centroid(Data)
    ax.scatter(C[0],C[1],marker='+',color='red')
    return C,hull

def solver_plot(Data,method = 'bubble'):
    if method == 'minidisk':
        D = minidisk(Data)
    elif method == 'bubble':
        D = bubble(Data,False)
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.scatter(Data.T[0],Data.T[1],color='k')
    t = np.linspace(0,2*np.pi,1000)
    x,y = D[0][0] + D[1]*np.cos(t), D[0][1] + D[1]*np.sin(t)
    ax.plot(x,y,'--',color='b')
    plt.show()
    print(D)

def point_to_plane_dist(point,plane):
    location = np.dot(point,plane[:-1]) + plane[-1]
    norm = np.sqrt(sum(plane[:-1]**2))
    return -location/norm

def two_norm(x):
    return np.sqrt(np.sum(x**2))

def sgn(x):
    return -1 if x < 0 else 1

def directed_plane_dist(center,point,plane):
    if np.all(point == center):
        return 0.0
    uD = point-center
    d = uD/two_norm(uD)
    divisor = np.dot(plane[:-1],d)
    if divisor == 0:
        return float('inf')
    loc = plane[-1] + np.dot(plane[:-1],center)
    if loc >= 0:
        raise AssertionError("Center is not inside the convex hull!")
    t = -loc/divisor
    if t < 0:
        return float('inf')
    coord = center + d*t
    return euc(center,coord)

def projected_distance(center,point,hull):
    if type(point) is list: np.array(point)
    if type(center) is list: np.array(center)
    distances = []
    for plane in hull.equations:
        distances.append(directed_plane_dist(center,point,plane))
    Min = min(distances)
    if Min == 0:
        raise ZeroDivisionError("Minimum distance is 0? Check hull!")
    return euc(center,point)/min(distances)

def quadratic_formula(a,b,c):
    sqrt = np.sqrt(b**2 - 4*a*c)
    return ((-b+sqrt)/(2*a), (-b-sqrt)/(2*a))

def ellipsoid_distance(center,point,radii):
    R2 = radii**2
    GR = 1
    for r in R2: GR *= r
    R2 = GR / R2
    vector = center - point
    A = np.sum(vector**2 * R2)
    B = 2*np.sum(point*vector*R2) - 2*np.sum(center*vector*R2)
    C = -GR -2*np.sum(point*center*R2)+np.sum(point**2*R2)+np.sum(center**2*R2)
    t = quadratic_formula(A,B,C)
    T = [point + vector*t[0] , point + vector*t[1]]
    E = [euc(point, T[0]), euc(point, T[1])]
    return euc(center,point) / euc(center, T[np.argmin(E)])

def min_max_norm(v,oldmin,oldmax,newmin,newmax):
    return ((v - oldmin)/(oldmax - oldmin))*(newmax - newmin) + newmin

class Centroid:
    def __init__(self,method = 'smallest disk',weights = 'ones',norm=True,
                 predictor = 'radius'):
        self.method_,self.C_,self.weights_,self.norm_=method,[],weights,norm
        self.predictor_ = predictor
        self.LT_ = {} if norm == 'neg' else [[],[]]
        self.radii_,self.hulls_ = {},{}
        self.func_call={'smallestdisk':self.SD,'boundingbox':self.BB,
                        'median':self.MED,'bestrep':self.BR,'poly':self.NP,
                        'average':self.AVG,'push&pull':self.PaP}
        self.predict_call={'nearest':self.nearest,'radius':self.bndradius,
                           'projection':self.projectedhyperplane,
                           'elliptical':self.elliptical,
                           'closestborder':self.closestborder}
    def LT(self,x,DC = False):
        if DC: x = dc(x)
        if self.norm_ == True: return x*self.LT_[0] + self.LT_[1]
        elif self.norm_ == 'neg':
            for i in range(len(x)):
                if x[i]<0:
                    try: x[i]=x[i]*self.LT_[i]['n'][0] + self.LT_[i]['n'][1]
                    except KeyError:
                        x[i] = x[i]*self.LT_[i]['p'][0] + self.LT_[i]['p'][1]
                else:
                    try: x[i] = x[i]*self.LT_[i]['p'][0] + self.LT_[i]['p'][1]
                    except KeyError:
                        x[i]=x[i]*self.LT_[i]['n'][0] + self.LT_[i]['n'][1]
            return x
        else: return x*self.weights_
    def inv_LT(self,x, DC = False):
        if DC: x = dc(x)
        if self.norm_ == True: return (x-self.LT_[1])/self.LT_[0]
        elif self.norm_ == 'neg':
            for i in range(len(x)):
                if x[i] < 0:
                    try: x[i]=(x[i]-self.LT_[i]['n'][1])/self.LT_[i]['n'][0]
                    except KeyError:
                        x[i] = (x[i]-self.LT_[i]['p'][1])/self.LT_[i]['p'][0]
                else:
                    try: x[i] = (x[i]-self.LT_[i]['p'][1])/self.LT_[i]['p'][0]
                    except KeyError:
                        x[i]= (x[i]-self.LT_[i]['n'][1])/self.LT_[i]['n'][0]
            return x
        else: return x/self.weights_
    def get_inv_centers(self):
        iC = {}
        for c,x in self.centers_.items():
            iC[c] = self.inv_LT(x,True)
        return iC
    def normer(self,v,i,n,a):
        if len(v) == 0: return v
        if n == 'n': b,c,d = 0.0,-0.5,0.0
        elif n == 'p': b,c,d = max(v),0.0,0.5
        elif n == 'a': b,c,d,n = max(v),0.0,1.0,'p'
        else: b,c,d = max(v),0.0,1.0
        m,y_int=(d-c)*self.weights_[i]/(b-a),(-a*d+c*b)*self.weights_[i]/(b-a)
        if not n: self.LT_[0].append(m),self.LT_[1].append(y_int)
        else: 
            try: self.LT_[i][n] = [m,y_int]
            except KeyError: self.LT_[i] = {n:[m,y_int]}
        return v*m + y_int
    def onenorm(self,x,i):
        a = min(x)
        if self.norm_ == 'neg':
            if a < 0:
                x[x < 0] = self.normer(x[x < 0],i,'n',a)
                x[x >= 0] = self.normer(x[x >= 0],i,'p',0.0)
            else: x = self.normer(x,i,'a',a)
        else: x = self.normer(x,i,False,a)
        return x
    def euc(self,x1,x2):
        return np.sqrt(sum((x1-x2)**2))
    def eucM(self,x,M,keep_track = False,use_max = True):
        if not keep_track: return [self.euc(x,x2) for x2 in M]
        m,a = 0,0
        dm = 0 if use_max else float('inf')
        for x2 in M:
            d = self.euc(x,x2)
            condition = (d > dm) if use_max else (d < dm)
            if condition: 
                m = a if keep_track == 'arg' else d
                dm = d
            a += 1
        return m
    def fit(self,X,y):
        self.D_,self.centers_,X = {},{},dc(X)
        if self.weights_ == 'ones': self.weights_ = np.ones((1,len(X.T)))[0]
        if self.norm_: 
            for i in range(len(X.T)): X.T[i] = self.onenorm(X.T[i],i)
        if self.norm_ != 'neg': self.LT_ = np.array(self.LT_)
        for x,c in zip(X,y):
            try: self.D_[c].append(x)
            except KeyError:
                self.D_[c],self.centers_[c] = [x],None
                self.C_.append(c)
        for c in self.D_:
            self.D_[c] = np.array(self.D_[c])
        self.C_ = np.sort(self.C_)
        try:
            self.func_call[self.method_]()
        except KeyError:
            print("Warning: Method is not recognized, only LT was performed.")
        return self
    def NP(self,Cs = None):
        if Cs == None: Cs = self.C_
        for c in Cs: 
            self.centers_[c],self.hulls_[c] = nPoly_centroid(self.D_[c],True)
    def PaP(self,Cs = None):
        if Cs == None: Cs = self.C_
        for c in Cs:
            Pull_points,Push_points = self.D_[c],[]
            for c2,P in self.D_.items():
                if c2 == c: continue
                for p in self.D_[c2]: Push_points.append(p)
            Push_points = np.array(Push_points)
            self.centers_[c] = push_and_pull(Push_points,Pull_points)
    def SD(self,Cs = None):
        if Cs == None: Cs = self.C_
        for c in Cs:
            a = self.D_[c]
            b = np.ascontiguousarray(a).view(
                    np.dtype((np.void,a.dtype.itemsize*a.shape[1])))
            ua = np.unique(b).view(a.dtype).reshape(-1,a.shape[1])
            self.centers_[c],self.radii_[c] = warm_start_bubble(ua)
    def BB(self,Cs = None):
        if Cs == None: Cs = self.C_
        for c in Cs:
            self.centers_[c] = []
            for Dt in self.D_[c].T:
                m,M = float('inf'),0
                for d in Dt:
                    if d < m: m = d
                    if d > M: M = d
                self.centers_[c].append(((M-m)/2.0)+m)
    def BR(self,Cs = None):
        if Cs == None: Cs = self.C_
        for c in Cs:
            R = np.arange(len(self.D_[c]))
            maxs=[self.eucM(self.D_[c][i],self.D_[c][R!=i],True) for i in R]
            arg = np.argmin(maxs)
            self.centers_[c],self.radii_[c] = self.D_[c][arg],maxs[arg]
    def AVG(self,Cs = None):
        if Cs == None: Cs = self.C_
        for c in Cs: 
            self.centers_[c] = sum(self.D_[c])/len(self.D_[c])
    def MED(self,Cs = None):
        if Cs == None: Cs = self.C_
        for c in Cs:
            self.centers_[c] = []
            for i in range(len(self.D_[c].T)):
                self.centers_[c].append(np.median(self.D_[c].T[i]))
    def learn_radii(self):
        for c in self.C_:
            self.radii_[c] = self.eucM(self.centers_[c],self.D_[c],True)
    def learn_elipsoidal_radii(self):
        for c in self.C_:
            self.radii_[c] = np.max(np.abs(self.D_[c] - self.centers_[c]),0)
    def learn_hull(self):
        for c in self.C_:
            self.hulls_[c] = ConvexHull(self.D_[c])
    def nearest(self,x,xc,c):
        return self.euc(x,xc)
    def bndradius(self,x,xc,c):
        try:
            return self.euc(x,xc)/self.radii_[c]
        except KeyError:
            R = self.eucM(self.centers_[c],self.D_[c],True)
            self.radii_[c] = R
            return self.euc(x,xc)/R
    def closestborder(self,x,xc,c):
        if c not in self.radii_:
            self.learn_elipsoidal_radii()
        elif type(self.radii_[c]) is float:
            self.learn_elipsoidal_radii()
        return np.max(np.abs(x-xc)/self.radii_[c])
    def elliptical(self,x,xc,c):
        if c not in self.radii_:
            self.learn_elipsoidal_radii()
        elif type(self.radii_[c]) is float:
            self.learn_elipsoidal_radii()
        return ellipsoid_distance(xc,x,self.radii_[c])
    def projectedhyperplane(self,x,xc,c):
        try:
            return projected_distance(xc,x,self.hulls_[c])
        except KeyError:
            hull = ConvexHull(self.D_[c])
            self.hulls_[c] = hull
            return projected_distance(xc,x,hull)
    def classDis(self,x,pred = 'predict'):
        L = [] if pred == 'predict' else {}
        if pred == 'predict':
            for c,xc in self.centers_.items():
                distance = self.predict_call[self.predictor_](x,xc,c)
                L.append((distance,c))
        else:
            for c,xc in self.centers_.items():
                distance = self.predict_call[self.predictor_](x,xc,c)
                L[c] = distance
        return L
    def argmin(self,cD):
        a,m = 0,float('inf')
        for i in range(len(cD)):
            if cD[i][0] < m: a,m = i,cD[i][0]
        return a
    def predict(self,X):
        lbl,X = [],dc(X)
        for x in X:
            cD = self.classDis(self.LT(x),'predict')
            lbl.append(cD[self.argmin(cD)][1])
        if len(lbl) == 1: return lbl[0]
        return np.array(lbl)
    def predict_proba(self,X):
        probas,X = [],dc(X)
        for x in X:
            cD = self.classDis(self.LT(x),'predict_proba')
            S,wS,i_pos,cond = sum(cD.values()),0,0,True
            for v in cD.values(): 
                if v == 0:
                    z = [0.0 for i in range(len(cD))]
                    z[i_pos],cond = 1.0,False
                    probas.append(z)
                    break
                wS += S/v
                i_pos += 1
            if cond: probas.append([(S/cD[c])/wS for c in self.C_])
        return np.array(probas)
    def plot(self,feat_names = None,class_colors = {},title="Centroid Rules",
             c2f=0.05,f2b=0.01,b2b=0.15,b2c=0.05,b_size=0.05,b2s_ratio=0.9,
             b2u=0.005,c_size=18,f_size=10,u_size=10,s_size=12):
        if self.radii_ == {}:
            print("Warning: Radii must be known... calculating radii...")
            self.learn_radii()
        if np.all(feat_names == None):
            feat_names = [str(i) for i in range(self.D_[self.C_[0]][0])]
        fig = plt.figure(figsize=(8.5,12.5))
        ax = fig.add_subplot(111)
        y = 1.0
        inst = 0
        z = 0
        for c in self.C_:
            try:
                color = class_colors[c]
            except KeyError:
                color = 'b'
            ax.text(0,y,c,fontsize=c_size,color=color)
            X = np.linspace(0,1.0,len(self.D_[c][0])+1)
            sep = (X[1]-X[0])*b2s_ratio
            if inst == 0:
                y -= c2f
                for i in range(len(X)-1):
                    x = (X[i] + X[i] + sep)/2.0
                    ax.text(x, y, feat_names[i],ha='center', fontsize = f_size)
                y -= (f2b + b_size)
            else:
                y -= (c2f + b_size)
            for i in range(len(self.D_[c][0])):
                C,r = self.centers_[c][i],self.radii_[c]
                if type(r) is np.ndarray:
                    r = r[i]
                left,right = np.max([0,C-r]),np.min([1,C+r])
                xs = min_max_norm(left,0,1,X[i],X[i]+sep)
                xw = min_max_norm(right,0,1,X[i],X[i]+sep) - xs
                xc = min_max_norm(C,0,1,X[i],X[i]+sep)
                ax.add_patch(Rect((xs,y),xw,b_size,fc=color,ec='none',
                                     zorder=z))
                z+=1
                ax.add_patch(Rect((xc-(sep/100),y),sep/50,b_size,
                                  fc=[0.831,0.686,0.216],ec='none',zorder=z))
                z += 1
                center = str(np.round(C,2))
                ax.text(xc,y-b2u,center,fontsize=u_size,ha='center',va='top')
            for i in range(len(self.D_[c][0])):
                ax.add_patch(Rect((X[i],y),sep,b_size,
                                  fill=False,ec=[0.8,0.8,0.8],
                                  fc = 'none', capstyle='round',zorder=z))
                z += 1
            if inst != (len(self.C_)-1):
                y -= b2c
                inst += 1
        ax.set_ylim(bottom=y)
        ax.axis("off")
        ax.set_title(title,fontsize=c_size)
        plt.show()
        
def Indicator(xy,width,height,left_direction=True,fc='w',
              fill=True,ec='none'):
    x,y = xy
    s = 1 if left_direction else -1
    XY = np.array([[x, y + height/2],
                   [x + s*width/2, y + height],
                   [x + s*width, y + height],
                   [x + s*width/2, y + height/2],
                   [x + s*width, y],
                   [x + s*width/2, y]])
    return Polygon(XY,fc=fc,fill=fill,ec=ec)

def txt(x,y,s,size_val,clr,ax,r,ha='left',va='bottom'):
    t = ax.text(x,y,s,size=size_val,color = clr,ha=ha,va=va)
    transf = ax.transData.inverted()
    bb = t.get_window_extent(renderer=r)
    bb_datacoords = bb.transformed(transf)
    x_axis = bb_datacoords.intervalx
    y_axis = bb_datacoords.intervaly
    coords = [x_axis[0],x_axis[1],y_axis[0],y_axis[1]]
    return t,coords
        
def comparative_plot(X,y,feat_names=None,class_colors={},
                     title="Center Comparison",c2f=0.05,f2b=0.01,b2b=0.15,
                     b2c=0.2,b_size=0.05,b2s_ratio=0.9,b2u=0.005,bardiv=30,
                     c_size=18,f_size=10,u_size=10,s_size=12,methods=None,
                     save='Fig6.pdf'):
    if np.all(feat_names == None):
        feat_names = [str(i) for i in range(X[0])]
    C = Centroid()
    if np.all(methods == None):
        methods = list(C.func_call.keys())
    li = np.argmax([len(methods[i]) for i in range(len(methods))])
    classes = np.unique(y)
    D = {c:np.zeros((len(methods),len(X[0]))) for c in classes}
    for i in range(len(methods)):
        C = Centroid(methods[i])
        C.fit(X,y)
        for c,center in C.centers_.items():
            D[c][i] = center
    fig = plt.figure(figsize=(11.5,14.5))
    ax = fig.add_subplot(111)
    r = fig.canvas.get_renderer()
    y = 1.0
    inst = 0
    for c,matrix in D.items():
        try:
            color = class_colors[c]
        except KeyError:
            color = 'b'
        ax.text(0,y,c,fontsize=c_size,color=color)
        I = np.linspace(0,1.0,len(X[0])+1)
        sep = (I[1]-I[0])*b2s_ratio
        if inst == 0:
            y -= c2f
            for i in range(len(X[0])):
                x = (I[i] + I[i] + sep)/2.0
                ax.text(x, y, feat_names[i],ha='center',fontsize=f_size)
            y -= (f2b + len(methods)*b_size)
        else:
            y -= (c2f + len(methods)*b_size)
        for i in range(len(X[0])):
            ax.add_patch(Rect((I[i],y),sep,b_size*len(methods),fill=False,
                             ec=[0.8,0.8,0.8],fc='none',capstyle='round'))
            ax.text(I[i],y-b2u,'0',fontsize=u_size,ha='center',va='top')
            ax.text(I[i]+sep,y-b2u,'1',fontsize=u_size,ha='center',va='top')
            for j in range(len(methods)):
                xc = min_max_norm(D[c][j][i],0,1,I[i],I[i]+sep)
                if xc < I[i]:
                    ax.add_patch(Indicator((I[i],y+j*b_size),sep/bardiv,b_size,
                                           True,color))
                elif xc > (I[i]+sep):
                    ax.add_patch(Indicator((I[i]+sep,y+j*b_size),sep/bardiv,
                                           b_size,False,color))
                else:
                    ax.add_patch(Rect((xc-sep/(2*bardiv),y+j*b_size),
                                      sep/bardiv,b_size,fc=color,ec='none'))
        for j in range(len(methods)):
            if (inst == 0) and (j == li):
                tx,coords=txt(I[-1],y+j*b_size,methods[j],u_size,'k',ax,r)
            else:
                ax.text(I[-1],y+j*b_size,methods[j],fontsize=u_size)
        if inst != (len(classes)-1):
            y -= b2c
            inst += 1
    ax.set_ylim(bottom=y-f2b)
    ax.set_xlim(left=-0.02,right=coords[1])
    ax.axis("off")
    ax.set_title(title,fontsize=c_size)
    plt.show()
    if save:
        fig.savefig(save,bbox_inches='tight',pad_inches=0)
                