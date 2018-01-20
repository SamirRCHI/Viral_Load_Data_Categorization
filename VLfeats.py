# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:57:51 2017

@author: sfarooq1
"""
import numpy as np
import colorsys
from time import time
import math
import operator as op
#import os
#import graphviz
import pydotplus
import webcolors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import random as random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import gridspec
from copy import deepcopy as dc
#from scipy import interpolate as intrp
from scipy import stats
#from coloring import *
from Networking import adj_axis,apply_white_ticks
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.metrics import confusion_matrix, roc_curve,roc_auc_score
from sklearn.manifold import TSNE
#import tsne
from sklearn.cluster import KMeans as kMeans
#from kMedoids import kMedoids
from sklearn import tree
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.neural_network import MLPClassifier as Backpropogation
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from Centroid import Centroid
import csv
import types

def generate_class_artists(Net,style='rectangle',which='all',fig=None):
    if which == 'all': L = Net.classes.keys()
    elif (type(which) is list) or (type(which) is np.ndarray): L = which
    elif type(which is str) and (which in Net.classes): L = [which]
    else: raise AssertionError("Unrecognized classes inputted")
    Artists,used_labels = [],[]
    for c in L:
        try: clr = Net.class_colors[c]
        except KeyError: continue
        used_labels.append(c)
        if style=='rectangle': 
            Artists.append(patches.Rectangle((0,0),1,1,
                            facecolor=clr,linewidth=0))
        elif style=='scatter':
            ax = fig.add_subplot(111)
            Artists.append(plt.scatter([0],[0],c=clr))
            fig.delaxes(ax)
        elif style=='line': Artists.append(plt.Line2D([0],[0],color=clr,lw=2))
        else:
            ax = fig.add_subplot(111)
            Artists.append(plt.scatter([0],[0],c=clr,marker=style))
            fig.delaxes(ax)
    return Artists,used_labels

def add_legend(Net,fig,style='rectangle',which='all',bbox=(0.,1.02,1., .102),
               loc=3,pad=0.5,size=14):
    Artists,Labels = generate_class_artists(Net,style,which,fig)
    ax = fig.add_subplot(111)
    ax.legend(Artists,Labels,bbox_to_anchor=bbox,loc = loc,\
             ncol=len(Artists),mode='expand',borderaxespad=pad,fontsize=size)
    ax.axis('off')

def before_and_after(VLpath, save = False):
    S_vl,S_d = [],[]
    for path in VLpath:
        VL,D = path.as_list(string=False,with_dates=True)
        VL,D = np.log10(np.array(VL)+1),np.array(D)
        if len(VL) < 3:
            continue
        for i in range(1,len(VL)):
            S_vl.append(VL[i]-VL[i-1]), S_d.append(D[i]-D[i-1])
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    ax.scatter(S_d,S_vl,alpha=0.3)
    Yl,Ylbl = ax.get_yticks(),[]
    yL=r'$\log_{10}(VL_{i+1}+1)-\log_{10}(VL_i+1)$ (copies/ml)'
    for y in Yl:
        if y < 0:
            Ylbl.append(r'$-10^{'+str(int(-y))+'} - 1$')
        else:
            Ylbl.append(r'$10^{'+str(int(y))+'} - 1$')
    adj_axis(ax,{'tick labelsize':16,'ylabel':(yL,16,20),
             'xlabel':(r'$t_{i+1} - t_i}$ (Days)',16,20),'standard':True})
    ax.arrow(-25,0.5,0,5.0,head_width=20,head_length=0.5,fc='k',ec='k')
    ax.arrow(-25,-0.5,0,-5.0,head_width=20,head_length=0.5,fc='k',ec='k')
    plt.show()
    if save: fig.savefig(save)
            
#before_and_after(VLpath2)

def GenerateExamples(second=True):
    x = np.array(range(7))
    H = np.array([6,5.75,4.7,5,5.35,6,5.6])
    L = np.array([2,3,2.5,2.75,3,2.5,1.9])
    Z = np.array([0,0,0,0,0,0,0])
    RS = np.array([6,2,0.75,0.25,0.5,0.35,0.3])
    RE = np.array([0.3,0.68,1.6,3.9,5.5,5.2,5.4])
    CS,CE = np.array([6,5,4,3,2,1,0]),np.array([0,1,2,3,4,5,6])
    SS,SE = np.array([6,6,5,5,4,3,1]),np.array([0,0,1,1,2,3,5])
    CU,CD = np.array([6,3,1,0,1,3,6]),np.array([0,3,5,6,5,3,0])
    S1,S2 = np.array([3,4,0.5,4.5,2,5.6,2]),np.array([6,6,2,1,2,6,6])
    S3 = np.array([6,2,4,1,4,2,4])
    if second:
        Names = ['Sustained High VL','Sustained Low VL',
                 'Durably Suppressed VL','High VL Suppression',
                 'Constant Suppression','Slow Suppression',
                 'Rebounding VL','Style 2','Style 3','Emerging VL']
        return x,[H,L,Z,RS,CS,SS,S1,S2,S3,RE],Names
    else:
        Names = ['Sustained High VL','Sustained Low VL',
                 'Durably Suppressed VL','Rapid Suppression',
                 'Constant Suppression',
                 'Slow Suppression','Re-Suppression','Rapid Emergence',
                 'Constant Emergence','Slow Emergence','Re-Emergence']
        return x,[H,L,Z,RS,CS,SS,CD,RE,CE,SE,CU],Names

def skip_num(smaller,bigger):
    return (bigger-smaller)/(smaller-1) # Eq: smaller + (smaller-1)*x = bigger

def resize(vec,new_size):
    l = len(vec)
    if l == 1: return vec*new_size
    elif l == new_size: return vec
    elif l > new_size: raise NotImplementedError
    new_vec,skp = [None]*new_size,skip_num(l,new_size)
    new_vec[0],new_vec[-1],prev_i = vec[0],vec[-1],0
    for i in range(1,(l/2)+1):
        new_vec[1+prev_i+skp] = vec[i]
        prev_i = prev_i+skp+1
    prev_i = new_size - 1 if l % 2 else -1
    start = (l/2)+1 if l % 2 else (l/2)
    for i in reversed(range(start,(l-1))):
        new_vec[prev_i - skp - 1] = vec[i]
        prev_i = prev_i - skp - 1
    return new_vec

def fill_None(vec):
    return list(np.linspace(vec[0],vec[-1],len(vec)))[0:(len(vec)-1)]
    
def fill_vec(vec):
    None_list,new_vec = [],[]
    for val in vec:
        if (None_list == []) and (new_vec == []): None_list.append(val)
        elif val == None: None_list.append(val)
        else:
            if len(None_list) == 1:
                new_vec.append(None_list.pop())
                None_list.append(val)
            else:
                None_list.append(val)
                new_vec = new_vec + fill_None(None_list)
                None_list = [val]
    new_vec.append(val)
    return new_vec

def area_of_severity(x,y,maximum = 7.0,minimum=0.0):
    area = 0
    for i in range(1,len(x)):
        area += (y[i] + y[i-1])*(x[i]-x[i-1])/2.0
    area_shift = (minimum - 0.0)*(x[-1] - x[0])
    area -= area_shift # This is to account for the adjustable minimum.
    maxarea = float((maximum-minimum)*(x[-1] - x[0]))
    return area/maxarea

def max_difference(y,maximum = 7.0,minimum=0.0):
    D,aD = [],[]
    for i in range(len(y)-1):
        d = (y[-1]-y[i])
        D.append(d)
        aD.append(np.abs(d))
    return float(D[np.argmax(aD)])

def compare_shrinkage():
    x = np.arange(10000)
    y1 = 1.0/(x+1)
    y2 = 1.0/((x)**(1.0/2.0)+1)
    y3 = 1.0/((x)**(1.0/3.0)+1)
    yl = 1.0/(np.log(x+1)+1)
    fig = plt.figure(figsize=(15.5,8.5))
    for i in range(1,5):
        ax = fig.add_subplot(2,2,i)
        ax.plot(x,y1,color='black')
        ax.plot(x,y2,color='blue')
        ax.plot(x,y3,color='green')
        ax.plot(x,yl,color='orange')
        ax.set_xlim(left=-1,right=(10.0**i))
    plt.show()

def pos_neg_diff(y):
    pos,neg,num_pos,num_neg = 0.0,0.0,0.0,0.0
    for i in range(len(y)-1):
        diff = y[i+1] - y[i]
        if diff >= 0:
            pos += diff
            num_pos += 1
        else:
            neg += np.abs(diff)
            num_neg += 1
    avg_pos = 0.0 if num_pos == 0.0 else (pos/num_pos)
    avg_neg = 0.0 if num_neg == 0.0 else (neg/num_neg)
    return avg_pos,avg_neg

def iqr(y):
    return np.percentile(y,75)-np.percentile(y,25)

def LRr2(x,y):
    if len(np.shape(x)) == 1:
        x = x.reshape((-1,1))
    if len(np.shape(y)) == 1:
        y = y.reshape((-1,1))
    L = LinearRegression()
    L.fit(x,y)
    r2 = L.score(x,y)
    return r2

def weighted_recency(x,y):
    if type(x) is np.ndarray:
        xl = x[-1] - x
    elif type(x) is list:
        xl = []
        for u in x:
            xl.append(x[-1] - u)
    else:
        raise AssertionError("Inputted 'x' not supported")
    aw = 0.0
    ay = 0.0
    for i in range(len(x)):
        w = 1.0/((xl[i])**0.5 + 1)
        aw += w
        ay += w*y[i]
    return ay/aw

def recency_deviation(x,y):
    L = weighted_recency(x,y)
    #L = y[-1]
    sum_term = 0.0
    for i in range(len(y)):
        sum_term += (L - y[i])**2.0
    return np.sqrt(sum_term/(len(y)-1))

def median_recency_dev(x,y):
    L = weighted_recency(x,y)
    all_terms = []
    for i in range(len(y)):
        all_terms.append(np.abs(L - y[i]))
    return np.median(all_terms)

def recency_reliance(x,y):
    return 1.0 / (median_recency_dev(x,y) + 1.0)

def adj_max_diff(x,y):
    D,aD,wr = [],[],weighted_recency(x,y)
    for i in range(len(y)):
        d = (wr-y[i])
        D.append(d)
        aD.append(np.abs(d))
    adjust_options = [recency_reliance(x,y)]#,LRr2(x,y)]
    max_diff = float(D[np.argmax(aD)])
    if max_diff > 0:
        max_diff = 0.0
    max_diff *= max(adjust_options)
    return max_diff

def concavity(y):
    summed = 0.0
    first_deriv = []
    for i in range(len(y)-1):
        first_deriv.append(y[i+1]-y[i])
    for i in range(len(first_deriv)-1):
        summed += np.abs(first_deriv[i+1]-first_deriv[i])
    summed /= len(first_deriv)-1
    return np.log(summed+1)

def naive_gap(x,y):
    m = (y[-1]-y[0])/float((x[-1]-x[0]))
    b,g = y[0] - m*x[0],[]
    m2 = -1.0/m
    for i in range(len(x)): 
        b2 = y[i] + (x[i]/m)
        xa,ya = (b2-b)/(m-m2) , m*((b2-b)/(m-m2)) + b
        g.append(np.sqrt((x[i]-xa)**2+(y[i]-ya)**2))
        #g.append(y[i] - (m*x[i] + b))
    return g

def integer_linear_interpolation(X,Y):
    newX,newY = [X[0]],[Y[0]]
    for i in range(1,len(X)):
        if X[i] == (X[i-1] + 1):
            newX.append(X[i])
            newY.append(Y[i])
            continue
        else:
            m = float(Y[i] - Y[i-1])/(X[i] - X[i-1])
            b = Y[i] - m*X[i]
            for x in range(X[i-1]+1,X[i]):
                newX.append(x)
                newY.append(m*x + b)
            newX.append(X[i])
            newY.append(Y[i])
    return newX,newY

def eucP(P1,P2):
    return np.sqrt( (P2[1]-P1[1])**2 + (P2[0]-P1[0])**2 )

def law_of_cosine(X,Y, angle = 'A'):
    a = eucP((X[0],Y[0]),(X[1],Y[1]))
    b = eucP((X[1],Y[1]),(X[2],Y[2]))
    c = eucP((X[0],Y[0]),(X[2],Y[2]))
    if angle == 'A':
        return np.arccos((c**2+b**2-a**2)/(2*c*b))
    elif angle == 'B':
        return np.arccos((a**2+c**2-b**2)/(2*a*c))
    else:
        return np.arccos((a**2+b**2-c**2)/(2*a*b))
    
def angle_based_knee_detection(X,Y,concave_down = True):
    X,Y = integer_linear_interpolation(X,Y)
    dX2,dY2 = [],[]
    for i in range(1,len(X)-1):
        dX2.append(X[i])
        dY2.append(Y[i-1] + Y[i+1] - 2*Y[i])
    increasing_order = np.argsort(dY2)
    if not concave_down:
        increasing_order = increasing_order[::-1]
    prev_angle,best_point = -float('inf'),None
    for i in increasing_order:
        p,c,n = i, i + 1, i + 2
        angle = law_of_cosine([X[p],X[c],X[n]],[Y[p],Y[c],Y[n]])
        #print X[c],dY2[p],angle*(180./np.pi)
        if angle >= prev_angle:
            prev_angle,best_point = angle,c
        else:
            break
    if best_point == None:
        print "No knee detected, returning None"
        return None,None
    return best_point

def two_knot_LR(x,y):
    if len(x) < 3:
        raise AssertionError('There must at least be three data points!')
    if len(np.shape(x)) == 1:
        x = x.reshape((-1,1))
    if len(np.shape(y)) == 1:
        y = y.reshape((-1,1))
    gaps = naive_gap(x,y)
    #knee = angle_based_knee_detection(x,gaps)
    bp = np.argmax(gaps)
    L1,L2 = LinearRegression(),LinearRegression()
    L1.fit(x[0:(bp+1)],y[0:(bp+1)]),L2.fit(x[bp:],y[bp:])
    s1,s2 = L1.score(x[0:(bp+1)],y[0:(bp+1)]),L2.score(x[bp:],y[bp:])
    e = (s1 + s2)/2.0
    L = [L1,L2]
    return L,e,bp

def two_knot_test(Net,real=0):
    if not real:
        x = np.arange(10)
        iy = np.array([10.5,5.5,2.1,1.6,1.4,1.2,1,1.2,1,0.9])
        sy = np.array([10,9.5,9.2,3,2,2.1,1.8,1.6,1.4,1.2])
        ky = np.array([9.1,9.0,8.4,6.4,6,5,3.5,3.6,2.4,2])
        cy = np.array([6,6.1,6.2,5.9,5.6,5.8,5.3,5,5.5,5.1])
        uy = np.array([2,3,7,7.5,8,8.2,8.3,8.5,9,9.1])
        ys = [iy,sy,ky,cy,uy]
        gposx = []
        gposy = []
        kposx = []
        kposy = []
        wposx = []
        wposy = []
        fig = plt.figure(figsize=(15.5,8.5))
        ax = fig.add_subplot(111)
        for y in ys:
            L,S,gpos,abp = two_knot_LR(x,y)
            winner = np.argmax(S)
            kposx.append(x[abp])
            kposy.append(y[abp])
            ax.plot(x,gaps)
            gpos = np.argmax(gaps)
            gposx.append(x[gpos])
            gposy.append(y[gpos])
            if winner == 0:
                wposx.append(x[gpos])
                wposy.append(y[gpos])
            else:
                wposx.append(x[abp])
                wposy.append(y[abp])
        fig = plt.figure(figsize=(15.5,8.5))
        ax = fig.add_subplot(111)
        for y in ys:
            ax.plot(x,y)
        ax.scatter(gposx,gposy,color='blue')
        ax.scatter(kposx,kposy,color='orange')
        ax.scatter(wposx,wposy,color='red',marker='x')
    else:
        last_vals,wr_vals,cc,ad,au,iqr,std,clrs,rd = [],[],[],[],[],[],[],[],[]
        for path in Net:
            if path.Class == None:
                continue
            VL,D = path.as_list(string=False,with_dates=True,clean=True)
            VL,D = tls(VL,'l10'), np.array(D) - D[0]
            L,S,bgap = two_knot_LR(D,VL)
            wr_vals.append(weighted_recency(D,VL))
            last_vals.append(VL[-1])
            cc.append(S)
            iqr.append(np.percentile(VL,75) - np.percentile(VL,25))
            std.append(np.std(VL))
            mean_pos,mean_neg = pos_neg_diff(VL)
            au.append(mean_pos)
            ad.append(mean_neg)
            #rd.append(recency_deviation(D,VL))
            rd.append(median_recency_dev(D,VL))
            clrs.append(Net.class_colors[path.Class])
        fig = plt.figure(figsize=(15.5,8.5))
        ax = fig.add_subplot(111)
        ax.scatter(last_vals,wr_vals,color=clrs)
        plt.show()
        fig = plt.figure(figsize=(15.5,8.5))
        ax = fig.add_subplot(111)
        ax.scatter(iqr,std,color=clrs)
        plt.show()
        fig = plt.figure(figsize=(15.5,8.5))
        ax = fig.add_subplot(111)
        ax.scatter(au,ad,color=clrs)
        plt.show()
        fig = plt.figure(figsize=(15.5,8.5))
        ax = fig.add_subplot(111)
        ax.scatter(std,rd,color=clrs)
        plt.show()
        return cc,clrs

def avg_slope(x,y):
    L1,L2 = two_knot_LR(x,y)
    return (L1.coef_[0][0] + L2.coef_[0][0])/2.0

class Continuous_Transform:
    def __init__(self,x,y):
        if type(x) is not np.ndarray:
            x,y = np.array(x),np.array(y)
        self.x,self.y = x,y
    def lin_interp(self,i):
        x1,x2,y1,y2 = self.x[i],self.x[i+1],self.y[i],self.y[i+1]
        m = (y2-y1)/(x2-x1)
        b = y2 - m*x2
        return m,b
    def predict(self,x):
        if (type(x) is int) or (type(x) is float):
            x = [x]
        i = 0
        y = []
        for xp in x:
            if xp > self.x[-1]:
                y.append(self.y[-1])
                continue
            while xp > self.x[i+1]:
                i += 1
            m,b = self.lin_interp(i)
            y.append(m*xp + b)
        return y

def feat_calc(x,y,maximum=7.0,minimum=0.0):
    A = area_of_severity(x,y,maximum,minimum)
    #D = max_difference(y,maximum,minimum)
    aD = adj_max_diff(x,y)
    rr = recency_reliance(x,y)
    IQR = iqr(y)
    #conc = concavity(y)
    #wr = weighted_recency(x,y)
    #s = np.std(y)
    #last = float(y[-1])
    #au,ad = pos_neg_diff(y)
    #r = median_recency_dev(x,y)
    return [A,rr,aD,IQR]

def get_feat_names():
    return ['Area','wRR','Adj MD','IQR']
    #return ['Area','Max Diff','Std','Last']
    

def get_feat_weights():
    return [1.0,1.0,1.0,1.0]
    # return [1.5,1.0,1.0,1.0]

# For Feature-less clustering comment the above 3 defs and uncomment next 3.
#def feat_calc(x,y,maximum=7.0,minimum=0.0):
#    return fill_vec(resize(y,35))
#
#def get_feat_names():
#    return [str(i+1) for i in range(35)]
#
#def get_feat_weights():
#    return [1.0 for i in range(35)]

def normalizer(v,oldminv,oldmaxv,newminv,newmaxv):
    return ((v - oldminv)/(oldmaxv - oldminv))*(newmaxv - newminv) + newminv

def one_transform(d, negatives = 'auto'):
    if negatives == 'auto':
        negatives = True if (min(d) < 0) else False
    if negatives:
        d[d < 0] = normalizer(d[d < 0],min(d[d < 0]),0,-0.5,0.0)
        d[d >= 0] = normalizer(d[d >= 0],0,max(d[d >= 0]),0.0,0.5)
        return d
    return normalizer(d,min(d),max(d),0.0,1.0)

def num_of_rows_and_cols(number_of_plots,row_greater=True):
    sq = int(np.ceil(number_of_plots ** 0.5))
    if number_of_plots <= (sq ** 2) - sq:
        if row_greater:
            rows,cols = sq,sq-1
        else:
            rows,cols = sq-1,sq
    else:
        rows,cols = sq,sq
    return rows,cols

def split_by_coords(rows,cols,division='center',d=True,x0=0,xf=1.,y0=0,yf=1.):
    coords,ri,ci = [],yf/rows,xf/cols
    if division == 'center':
        ca,ra = ci/2.0,ri/2.0
    else:
        ca = ci if ('right' in division) else 0
        ra = ri if ('top' in division) else 0
    for i in range(rows):
        for j in range(cols):
            coords.append((((j*ci)+ca),((i*ri)+ra)))
    if d:
        dvdlines = []
        for i in range(1,rows):
            dvdlines.append([[x0,xf],[i*ri,i*ri]])
        for j in range(1,cols):
            dvdlines.append([[j*ci,j*ci],[y0,yf]])
        return coords,dvdlines
    return coords

def PlotSingleAxEx(save=False):
    fig = plt.figure(figsize=(15.5,8.5))
    ax = fig.add_subplot(111)
    x,Y,Names = GenerateExamples(True)
    clrs = np.array([[213,94,0],[0,114,178],[0,158,115],[204,121,167],
                     [1,1,1],[1,1,1],[0,0,0],[1,1,1],[1,1,1],[240,228,66]])
    clrs = clrs / 255.0
    mrks = ['^','D','P','*','','','o','','','s']
    Artists = []
    Labels = []
    for i in [0,1,2,3,6,9]:
        ax.plot(x,Y[i],'--',color=clrs[i],lw=2.5)
        if i == 3:
            Artists.append(ax.scatter(x,Y[i],color=clrs[i],
                                      s=121,marker=mrks[i],alpha=0.6))
        else:
            Artists.append(ax.scatter(x,Y[i],color=clrs[i],s=64,
                                      alpha=0.6,marker=mrks[i]))
        Labels.append(Names[i])
    #ax.annotate(Names[0], xy=(5, 6), xytext=(4, 5.75),
    #            arrowprops=dict(facecolor='black', shrink=0.05))
    ax.legend(Artists,Labels,bbox_to_anchor=(0.,1.02,1., .102),loc = 3,\
             ncol=3,mode='expand',borderaxespad=0.5,fontsize=18)
    adj_axis(ax,{'labelbottom':False,'bottom xtick':False,'ytick labelsize':14,
                 'xlabel':('Time',18),'standard':True,
                 'ylabel':(r'$log_{10}$(Viral Load + 1) (copies/ml)',18)})
    plt.show()
    if save:
        fig.savefig(save)
    
def b2r(I):
    if I >= 0:
        if I >= 0.5:
            return([1,2-(2*I),0])
        else:
            return([1,1,1-(2*I)])
    else:
        if I >= -0.5:
            return([1+(2*I),1,1])
        else:
            return([0,2+(2*I),1])
        
def PlotExampleFeatures(save=False,feat_extract=True,second=True,OT=False):
    x,Y,Names = GenerateExamples(second)
    fourth = 'Rebounding' if second else 'Emergence'
    sup_x = 0.77 if second else 1.005
    LGwidth = 0.77 if second else 1
    feat_names = get_feat_names()
    rows,colms = num_of_rows_and_cols(len(feat_names))
    c_coords,key_lines = split_by_coords(rows,colms)
    b_coords = split_by_coords(rows,colms,'bottomleft',False)
    excds,ex_lns = split_by_coords(rows,colms,'bottomleft',xf=6.0,yf=6.0)
    feat_vals = {n:[] for n in feat_names}
    for i in range(len(Y)):
        F = feat_calc(x,Y[i],6.0)
        for j in range(len(feat_names)):
            feat_vals[feat_names[j]].append(F[j])
    for name in feat_names:
        d = np.array(feat_vals[name])
        feat_vals[name] = one_transform(d,OT)
    fig = plt.figure(figsize=(15,10))
    ax,ot,tt,of,x0 = fig.add_subplot(111),0.39,0.78,0,0.015
    LG,LP = [0.67,1,0.67],[1,0.67,1]
    ax.add_patch(patches.Rectangle((x0,ot+of),LGwidth,ot,facecolor=LG,
                                   edgecolor=LG,linewidth=0))
    ax.add_patch(patches.Rectangle((x0,0),1,ot,facecolor=LP,
                                   edgecolor=LP,linewidth=0))
    ax.text(1.005,ot/2,fourth,verticalalignment='center',rotation=270,
            horizontalalignment='right',fontsize=14)
    ax.text(sup_x,(tt-ot-of)/2+ot+of,'Suppression',rotation=270,fontsize=14,
            verticalalignment='center',horizontalalignment='right')
    ax.text(0,(tt-ot-of)/2+ot+of,'Viral Load (copies/ml)',rotation=90,
            verticalalignment='center',horizontalalignment='center',
            fontsize=14)
    ax.text(0.5,0,'Time',verticalalignment='top',fontsize=14)
    ax.set_xlim(left=-0.01,right=1.01)
    ax.set_ylim(bottom=-0.01,top=1.01)
    ax.axis("off")
    gs = gridspec.GridSpec(3,4)
    for i in range(len(Y)):
        if (i == 3) and feat_extract:
            j = i + 2
            #ax = fig.add_subplot(3,4,4)
            ax = fig.add_subplot(gs[3])
            ax.set_title('Feature Color Key',fontsize=14)
            for n in range(len(feat_names)):
                cd,bcrd,pts = c_coords[n],b_coords[n],50
                ax.text(cd[0],cd[1],feat_names[n],verticalalignment='center',
                        horizontalalignment='center',fontsize=14)
                if min(feat_vals[feat_names[n]]) < 0:
                    lnsp = np.linspace(-0.5,0.5,pts)
                else:
                    lnsp = np.linspace(0,1,pts)
                for m in range(pts):
                    ax.add_patch(patches.Rectangle((bcrd[0],
                                 bcrd[1]+m*((1.0/rows)/pts)),(1.0/colms),
                                 ((1.0/rows)/pts),edgecolor=b2r(lnsp[m]),
                                 facecolor=b2r(lnsp[m]),linewidth=0.0))
            adj_axis(ax,{'labelbottom':False,'bottom xtick':False,
                         'labelleft':False,'left ytick':False})
            for xl,yl in key_lines:
                ax.plot(xl,yl,':',color='k',lw=0.5)
            ax.set_xlim(left=-0.05,right=1.05)
            ax.set_ylim(bottom=-0.05,top=1.05)
        elif i >= 3:
            j = i + 2
        else:
            j = i + 1
        #ax = fig.add_subplot(3,4,j)
        if (i >= 6) and second:
            j += 1
        ax = fig.add_subplot(gs[j-1])
        if feat_extract:
            for k in range(len(feat_names)):
                coord,clr = excds[k],b2r(feat_vals[feat_names[k]][i])
                ax.add_patch(patches.Rectangle(coord,6.0/colms,6.0/rows,
                                               facecolor=clr,
                                               edgecolor=clr,linewidth=0))
            for xl,yl in ex_lns:
                ax.plot(xl,yl,':',color='k',lw=0.5)
        ax.plot(x,Y[i],'--',marker='o',color=[0,100.0/255,50.0/255])
        ax.set_ylim(bottom=-0.5,top=6.5)
        ax.set_xlim(left=-0.5,right=6.5)
        adj_axis(ax,{'labelbottom':False,'bottom xtick':False})
        if (j == 1) or (j == 5) or (j == 9):
            ypos,ylbl = [0,2,4,6],[r'0',r'$10^2$',r'$10^4$',r'$10^6$']
            adj_axis(ax,{'yticks':ypos,'yticklabels':(ylbl,14)})
        else:
            adj_axis(ax,{'labelleft':False,'left ytick':False})
        ax.set_title(Names[i],fontsize=14)
    gs.tight_layout(fig,rect = [0.14,0.13,0.895,1])
    plt.show()
    if save != False:
        if save[-4:] != '.pdf':
            fig.savefig(save+'.pdf')
        else:
            fig.savefig(save)
        
#PlotExampleFeatures()

def tls(value, base_or_root = 2.0, func = 'log', add1 = True):
    br = base_or_root
    if br == None:
        return value
    if type(br) is str:
        func = 'log' if br[0] == 'l' else 'root'
        br = float(br[1:])
    if (type(value) is list) or (type(value) is np.ndarray):
        if func == 'log':
            return np.array([math.log(v+10 if add1 else v,br) for v in value])
        elif func == 'root':
            return np.array([v ** (1.0/br) for v in value])
        else:
            raise AssertionError('Function |'+func+'| is not recognized')
    if func == 'log':
        return math.log(value+10 if add1 else value,br)
    elif func == 'root':
        return value ** (1.0 / br)
    else:
        raise AssertionError('Function |'+func+'| is not recognized')

def patient_feats(VLpath,patient_id,days = float('inf'),tform = 'l10',E=False):
    path = VLpath.Nets[patient_id]['Path']
    VL,D = path.as_list(string=False,with_dates=True,clean=True)
    if len(VL) < 3: return False
    VL,D = tls(VL,tform), np.array(D) - D[0]
    if days <= 1: days *= D[-1]
    if E and (D[-1] <= days): return False
    VL,D = VL[D <= days],D[D <= days]
    if len(VL) < 3: return False
    return feat_calc(D,VL,tls(10000000.0,tform),tls(0.0,tform))
    

def get_Data(VLpath,with_order=False,tform='l10',days=float('inf'),E=False):
    names = get_feat_names()
    data,p_ids = {n:[] for n in names},[]
    for path in VLpath:
        feat = patient_feats(VLpath,path.patient_id,days,tform,E)
        if not feat:
            continue
        if feat[0] > 1:
            print path.patient_id
        for i in range(len(feat)):
            data[names[i]].append(feat[i])
        if with_order:
            if with_order == 'color': 
                p_ids.append(VLpath.class_colors[path.Class])
            else: p_ids.append(path.patient_id)
    for n in names:
        data[n] = np.array(data[n])
    if with_order:
        return data,p_ids
    return data

def get_Matrix(VLpath,normalize=True,with_order=False,dat=False,tform = 'l10',
               days = float('inf'),incomplete=False):
    names = get_feat_names()
    weights = get_feat_weights()
    data,p_ids = get_Data(VLpath,True,tform,days,incomplete)
    M = []
    for i in range(len(names)):
        n,w = names[i],weights[i]
        if normalize:
            if (normalize == True) or (normalize == 'auto'):
                M.append(w*one_transform(data[n]))
            else:
                M.append(w*one_transform(data[n],False))
        else:
            M.append(data[n])
    M = np.array(M).T
    if dat:
        if with_order:
            return M,data,p_ids
        else:
            return M,data
    if with_order:
        return M,p_ids
    return M

#M = get_Matrix(VLpath2)

def feat_vs_feat(VLpath,clustered=False,save = False,mk = 'o',legend=True):
    names = get_feat_names()
    if clustered == False: data,colors = get_Data(VLpath),'b'
    elif type(clustered) is bool: data,colors = get_Data(VLpath,'color')
    else: data,colors = clustered[0],clustered[1]
    amt = (len(names)*(len(names)-1))/2.0
    fig = plt.figure(figsize=(14,10))
    plot_num = 1
    r,c = num_of_rows_and_cols(amt,False)
    for i in range(len(names)):
        n1 = names[i]
        for n2 in names[(i+1):]:
            ax = fig.add_subplot(r,c,plot_num)
            plot_num += 1
            if mk == 'o':
                ax.scatter(data[n1],data[n2],facecolors=colors,alpha=0.4)
            else:
                for x,y,m,clr in zip(data[n1],data[n2],mk,colors):
                    ax.scatter(x,y,marker=m,facecolors=clr,alpha=0.4)
            adj_axis(ax,{'xlabel':(n1,14),'ylabel':(n2,14),'tick labelsize':14,
                     'standard':True})
    if amt == 3:
        ax = fig.add_subplot(2,2,4,projection = '3d')
        ax.scatter(data[names[0]],data[names[1]],data[names[2]],
                   facecolors=colors,alpha=0.4)
        ax.set_xlabel(names[0],fontsize=14)
        ax.set_ylabel(names[1],fontsize=14)
        ax.set_zlabel(names[2],fontsize=14)
    if legend: 
        ax = fig.add_subplot(111)
        ax.axis("off")
        add_legend(VLpath,fig,'scatter',pad=0.25)
    plt.show()
    if save: fig.savefig(save)

#feat_vs_feat(VLpath2

def get_colorblind_colors():
    o,g = np.array([230,159,0])/255.0,np.array([0,158,115])/255.0
    b,k = np.array([0,114,178])/255.0,np.array([0,0,0])
    r,p = np.array([213,94,0])/255.0,np.array([204,121,167])/255.0
    return o,g,b,k,r,p

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

class linkage_clustering:
    def __init__(self, Z, thresh, label = None):
        self.p = len(Z)
        self.S = set(range(self.p+1))
        self.Clusters = []
        self.Z = Z
        self.thresh = thresh
        self.Zt = Z[Z[:,2] <= thresh, :]
        self.C,self.B,self.cn = set(range(len(self.Zt))),{},0
        if type(label) is type(None):
            self.label = np.arange(self.p+1)
        else:
            self.label = np.array(label)
        i = len(self.Zt) - 1
        while len(self.C) > 0:
            cluster = self.add_subcomponents(i, set())
            while (i not in self.C) and i > 0:
                i -= 1
            self.Clusters.append(cluster)
            self.cn += 1
        for s in self.S:
            self.Clusters.append({s})
            self.B[self.label[s]] = self.cn
            self.cn += 1
    def add_subcomponents(self,i,cluster):
        first = int(self.Zt[i, 0])
        second = int(self.Zt[i, 1])
        for index in [first,second]:
            if index <= self.p:
                cluster.add(index)
                self.B[self.label[index]] = self.cn
                self.S.remove(index)
            else:
                cluster = self.add_subcomponents(index-(self.p+1),cluster)
        self.C.remove(i)
        return cluster
    def generate_cluster_colors(self,colorblind_friendly=True,set_own=False):
        self.colors = {}
        if colorblind_friendly:
            if len(self.Clusters) > 8:
                err = 'Only up to 8 classes are supported with '+\
                'colorblind_friendly. Either set your own colors or set '+\
                'colorblind_friendly to False.'
                raise AssertionError(err)
            clrs = np.array([[0,114,178],[213,94,0],[204,121,167],[0,0,0],
                            [0,158,115],[230,159,0],[86,180,233],[240,228,66]])
            clrs = clrs/255.0
            if len(self.Clusters) == 5:
                clrs = clrs[0:len(self.Clusters)]
                clrs = clrs[[2,3,0,4,1]]
            clrs = clrs[0:len(self.Clusters)]
        elif set_own:
            clrs = set_own
        else:
            clrs = equi_color(len(self.Clusters))
        for i in range(len(clrs)):
            self.colors[i] = clrs[i]
        self.def_bracket_color = np.array([0.65,0.65,0.65])
    def plot_segment(self,x,bot,top,color):
        if bot > self.thresh:
            self.ax.plot([x,x],[bot,top],color=self.def_bracket_color)
        elif top <= self.thresh:
            self.ax.plot([x,x],[bot,top],color=color)
        else:
            self.ax.plot([x,x],[bot,self.thresh],color=color)
            self.ax.plot([x,x],[self.thresh,top],color=self.def_bracket_color)
    def check_position(self,index):
        if index <= self.p:
            self.pos += 10
            cluster = self.B[self.label[index]]
            color = self.colors[cluster]
            self.brackets[index] = (self.pos,0.0,color)
            self.lo.append(index)
            if self.pos < self.clus_pos[cluster][0]:
                self.clus_pos[cluster][0] = self.pos
            if self.pos > self.clus_pos[cluster][1]:
                self.clus_pos[cluster][1] = self.pos
        else:
            self.update_bracket(index-self.p-1)
    def update_bracket(self,zi):
        left,right = self.Z[zi,0:2]
        left,right = int(left),int(right)
        self.check_position(left)
        self.check_position(right)
        x1,d1,c1 = self.brackets[left]
        x2,d2,c2 = self.brackets[right]
        top = self.Z[zi,2]
        self.plot_segment(x1,d1,top,c1)
        self.plot_segment(x2,d2,top,c2)
        new_color = c1 if top <= self.thresh else self.def_bracket_color
        self.brackets[zi+self.p+1] = ((x1+x2)/2.0,top,new_color)
        self.ax.plot([x1,x2],[top,top],color=new_color)
    def plot_dendrogram(self, ax = None):
        self.lo = []
        self.clus_pos = {i:[float('inf'),0] for i in range(len(self.Clusters))}
        try:
            self.colors
        except AttributeError:
            cbf = True if len(self.Clusters) <= 8 else False
            self.generate_cluster_colors(cbf)
        self.pos,self.brackets = -5.0,{}
        if ax == None:
            fig = plt.figure()
            self.ax = fig.add_subplot(111)
        else:
            self.ax = ax
        self.update_bracket(self.p-1)
        lx = 5+10*self.p
        self.ax.plot([5,lx],[self.thresh,self.thresh],'--',
                     color=self.def_bracket_color)
        self.ax.set_ylim(bottom=0)
        self.ax.set_xlim(left=5,right=lx)
        adj_axis(self.ax,{'xticks':range(5,lx+1,10),
        'xticklabels':self.label[self.lo],'bottom xtick':False})

def auto_cluster(Net,title='Dendrogram',cluster_threshold = 3.8,
                 update_class = False,show_feat_plot = False,transform='l10',
                 include_heatmap = False,OT=True,lw = 0.5,fs=(15.5,8.5),
                 save=False):
    f_n,weights = get_feat_names(),get_feat_weights()
    VL_feature_matrix,dat,p_ids = get_Matrix(Net,OT,True,True,transform)
    Z = linkage(VL_feature_matrix,'ward')
    LC = linkage_clustering(Z,cluster_threshold,p_ids)
    LC.generate_cluster_colors()
    if update_class:
        for i in range(len(p_ids)):
            p = p_ids[i]
            Net.Nets[p]['Path'].update_class(LC.B[p])
        Net.set_class_colors(LC.colors)
        if update_class == 'only':
            return Net
    matplotlib.rcParams['lines.linewidth'] = lw
    fig = plt.figure(figsize=fs) 
    if include_heatmap:
        gs = gridspec.GridSpec(2,1,height_ratios=[1,3.5],hspace=0)
        ax = fig.add_subplot(gs[0])
    else:
        ax = fig.add_subplot(111)
    plt.title(title,fontsize=20)
    LC.plot_dendrogram(ax)
    adj_axis(ax,{'spines':False,'all ticks':'off','ytick labelsize':14,
                 'ylabel':('Euclidean',18,20),'labelbottom':False})
    if not include_heatmap:
        plt.xlabel('Patient',fontsize=18)
    if include_heatmap:
        l = len(f_n)
        ax = fig.add_subplot(gs[1])
        ax.set_xlim(left=0,right=len(VL_feature_matrix))
        ax.set_ylim(bottom=0,top=len(VL_feature_matrix.T))
        col,index,start,end = 0,0,float('inf'),-float('inf')
        for column in VL_feature_matrix.T:
            column /= weights[index]
            index += 1
            for f in range(len(column)):
                i = LC.lo[f]
                if column[i] > end: end = column[i]
                if column[i] < start: start = column[i]
                ax.add_patch(patches.Rectangle((f,col),1,1,
                             facecolor=b2r(column[i]),linewidth=0))
            col += 1
        adj_axis(ax,{'yticks':[j+0.5 for j in range(len(VL_feature_matrix.T))],
                     'yticklabels':f_n,'tick labelsize':14,'spines':False,
                     'ylabel':('Features',18,10),'all ticks':False})
        cluster_pos,mag = [],[]
        for cluster,L in LC.clus_pos.items():
            mn,mx = (L[0]-5.0)/10.0,(L[1]-5.0)/10.0
            cluster_pos.append(mn + ((mx-mn)/2.0))
            mag.append(int(mx-mn+1))
            ax.plot([mn,mx,mx,mn,mn],[0.01,0.01,l,l,0.01],color='k')
        adj_axis(ax,{'xticks':cluster_pos,'xticklabels':mag,
                     'xlabel':('Cluster Size',18)})
        # displaying heatmap colormap:
        #fig.subplots_adjust(hspace=0)
        gs.tight_layout(fig,rect = [0.0,0.0,0.875,1])
        ax = fig.add_subplot(1,22,22)
        Rng = np.linspace(start,end,1000)
        for i in range(1000):
            ax.add_patch(patches.Rectangle((0,Rng[i]),1,(end-start)/1000.0,
                                           facecolor=b2r(Rng[i]),linewidth=0))
        ax.set_ylim(bottom=start,top=end)
        adj_axis(ax,{'all off':True})
        ax.yaxis.set_label_position("right")
        adj_axis(ax,{'right spine':True,'right ytick':True,'labelright':True,
                     'tick labelsize':14,
                     'ylabel':('Normalized Feature Score',18)})
    plt.show()
    if save: fig.savefig(save)
    if show_feat_plot:
        feat_vs_feat(Net,True,'Hierarchical Feature Extraction.pdf')
    return Net

#VLpathT = auto_cluster(VLpathT,'Hierarchical Patient Clustering,
#                       4.8,True,False,'l10',True,'pos only')

def euc(x,y):
    return np.sqrt(sum((x-y)**2))

#def tsne_trial_and_error(VLpath):
#    results = []
#    errors = []
#    Data,p_ids = get_Matrix(VLpath,True,True)
#    for i in range(10):
#        R,e = tsne.tsne(Data)
#        results.append(R)
#        errors.append(errors)
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax.plot(range(len(errors)),errors)
#    return Data,p_ids,results,errors

#def tsne_analysis(VLpath,A=None,d_and_p=None,k=5,update=True,other=7,tf='l10'):
#    if A == None:
#        Data,dat,p_ids = get_Matrix(VLpath,True,True,True,tf)
#        if other == False:
#            A,e = tsne.tsne(Data)
#        else:
#            if other == True:
#                other = None
#            T = TSNE(init='pca',random_state=other)
#            T = T.fit(Data)
#            A,e = T.embedding_,T.kl_divergence_
#        print e
#    elif (d_and_p == None) and update:
#        raise AssertionError('Cannot update class without patient ids!')
#    else:
#        dat,p_ids = d_and_p
#    DB = DBSCAN(eps=k)
#    L = DB.fit_predict(A)
#    Col,class_colors,C = [],{},equi_color(len(np.unique(L)))
#    for i in range(len(L)):
#        l = L[i]
#        Col.append(C[l])
#        class_colors[l] = C[l]
#        if update:
#            VLpath.Nets[p_ids[i]]['Path'].update_class(l)
#    if update:
#        VLpath.set_class_colors(class_colors)
#        if update == 'only':
#            return VLpath
#    fig = plt.figure(figsize=(8.5,8.5))
#    ax = fig.add_subplot(111)
#    ax.scatter(A.T[0],A.T[1],color=Col,alpha=0.7)
#    plt.show()
#    feat_vs_feat(VLpath,(dat,Col))
#    return VLpath,A,dat,p_ids

#def unsupervised_clustering(VLpath,tform='l10',cluster_type='kMedoids',k=5):
#    M,D,p = get_Matrix(VLpath,True,True,True,tform)
#    if cluster_type == 'kMedoids':
#        kM = kMedoids(M)
#        L = kM.fit_predict(k)
#    elif cluster_type == 'kMeans':
#        kM = kMeans(k)
#        kM = kM.fit(M)
#        L = kM.predict(M)
#    C,Col,cc = equi_color(k),[],{}
#    for i in range(len(L)):
#        l = int(L[i])
#        Col.append(C[l])
#        cc[l] = C[l]
#        VLpath.Nets[p[i]]['Path'].update_class(l)
#    VLpath.set_class_colors(cc)
#    return D,Col
#    #feat_vs_feat(VLpath,(D,Col))

#VLpath2,R = tsne_analysis(VLpath2)

def VLplot(VLpath,mx=10000000.0,mn=-0.1,transform='l10',save=False):
    mn,mx = tls(mn,transform),tls(mx,transform)
    rows,i = int(np.ceil(len(VLpath.class_colors)*0.5)),-1
    fig = plt.figure(figsize=(15.5,4.5*rows))
    gs = gridspec.GridSpec(rows,2)
    for c,clr in VLpath.class_colors.items():
        i += 1
        ax = fig.add_subplot(gs[i])
        for p_id,path in VLpath.pat_by_class[c].items():
            VL,D = path.as_list(string=False,with_dates=True,clean=True)
            VL,D = tls(VL,transform),np.array(D)-D[0]
            ax.plot(D,VL,alpha=0.4,color=clr,lw=0.5)
            ax.scatter(D[-1],VL[-1],alpha=0.8,facecolor=clr)
        ax.set_ylim(bottom=mn,top=mx)
        adj_axis(ax,{'standard':True,'tick labelsize':14})
        if np.mod(i,2):
            adj_axis(ax,{'left spine':False,'left ytick':False,
                         'labelleft':False})
        else:
            Y = ax.get_yticks()
            aY,nY = [],[] #aY adjusts the position of y-axis, nY is label.
            logged = True if transform[0] == 'l' else False
            val = transform[1:]
            for y in Y:
                if y % 2: # If y is odd then we keep it
                    if logged:
                        aY.append(y)
                        nY.append(r'$'+val+'^{'+str(int(y))+'} - 10$')
                    else:
                        aY.append(y)
                        nY.append(r'$'+str(int(y))+'^{1/'+val+'}$')
            adj_axis(ax,{'yticks':aY,'yticklabels':nY})
    fig.text(0.04,0.5,'Viral Load (copies/ml)',rotation=90,size=18,
             horizontalalignment='center',verticalalignment='center')
    fig.text(0.5,0.075,'Days Since First Viral Load Measurement',size=18,
             horizontalalignment='center',verticalalignment='center')
    add_legend(VLpath,fig,'line',pad=0.25,size=18)
    plt.show()
    if save: fig.savefig(save)
    
def VLplot_binned(VLpath,mx=10000000.0,mn=-0.1,transform='l10',save=False):
    mn,mx = tls(mn,transform),tls(mx,transform)
    rows,i = int(np.ceil(len(VLpath.class_colors)*0.5)),-1
    fig = plt.figure(figsize=(15.5,4.5*rows))
    gs = gridspec.GridSpec(rows,2)
    axs = []
    for c,clr in VLpath.class_colors.items():
        i += 1
        Ds,VLs = [],[]
        ax = fig.add_subplot(gs[i])
        for p_id,path in VLpath.pat_by_class[c].items():
            VL,D = path.as_list(string=False,with_dates=True,clean=True)
            VL,D = tls(VL,transform),np.array(D)-D[0]
            Ds.append(D),VLs.append(VL)
        ax.set_title(c,fontsize=18)
        axs.append(binned_time_series(np.array(Ds),np.array(VLs),ax,
                        ytl=rev_log10_p10,halfyticks=True,plot_last=True))
    same_axis(axs,rows,2,yspine=True)
    fig.text(0.04,0.5,'Viral Load (copies/ml)',rotation=90,size=18,
             horizontalalignment='center',verticalalignment='center')
    fig.text(0.5,0.075,'Days Since First Viral Load Measurement',size=18,
             horizontalalignment='center',verticalalignment='center')
    plt.show()
    if save: fig.savefig(save)
    
def uniformity_of_last(VLpath):
    for c in VLpath.classes:
        last = []
        for path in VLpath.pat_by_class[c].values():
            VL,D = path.as_list(string=False,with_dates=True,clean=True)
            last.append(D[-1] - D[0])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(last)
        print c,stats.kstest(last,'uniform')

def log(value, base = np.e):
    if base == None:
        return value
    if (type(value) is list) or (type(value) is np.ndarray):
        return np.array([math.log(v,base) for v in value])
    return math.log(value,base)

def manual_assign(VLpath):
    for path in VLpath:
        VL,D = path.as_list(string=False,with_dates=True,clean=True)
        if len(VL) < 3:
            continue
        D,VLo,lg,Mvl = np.array(D)-D[0],np.array(VL),5.5,10000000.0
        VL10,VLt = np.log10(VLo+1),tls(VLo,lg,'root')
        VLs,lgs = [VLo,VL10,VLt],[None,'l10','r'+str(lg)]
        fig = plt.figure(figsize=(18,4))
        for i in range(len(VLs)):
            ax = fig.add_subplot(1,3,i+1)
            ax.plot(D,VLs[i],'--',marker='o',color=[0,100.0/255,50.0/255])
            ax.plot([0,D[-1]],[tls(20,lgs[i]),tls(20,lgs[i])],':',lw=0.5)
            ax.plot([0,D[-1]],[tls(48,lgs[i]),tls(48,lgs[i])],':',lw=0.5)
            ax.plot([0,D[-1]],[tls(1000,lgs[i]),tls(1000,lgs[i])],':',lw=0.5)
            ax.set_xlim(right=max([600,D[-1]]))
            ax.set_ylim(bottom=-0.1,top=tls(Mvl,lgs[i]))
            a = np.round(area_of_severity(D,VLs[i],tls(Mvl,lgs[i])),4)
            d = np.round((VLs[i][-1]-VLs[i][0])/tls(Mvl,lgs[i]),4)
            ax.set_title(str(lgs[i])+'- A: '+str(a)+' D: '+str(d))
        plt.show()
        prmpt = 'For Patient '+path.patient_id+' pick class (s,l,h,r,e,stop): '
        trns,abrv_c = {'s':'SHVL','l':'SLVL','h':'DSVL','r':'Supression',
                'e':'Emergence'},'start'
        accepted = {'s','l','h','r','e','stop',''}
        while abrv_c not in accepted:
            abrv_c = raw_input(prmpt)
        if abrv_c == '':
            continue
        elif abrv_c == 'stop':
            break
        path.update_class(trns[abrv_c])
        
def Greub_LLVR(D,VL):
    consec,new_class = False,'Unspecified'
    for i in range(len(VL)):
        if VL[i] < 50:
            if consec:
                if ((len(VL)-(i+1)) >= 2) and ((D[i]-D[i-1]) <= 168):
                    if (D[len(VL)-1] - D[i+1]) >= 168:
                        M = np.max(VL[(i+1):])
                        if M > 500:
                            new_class = 'Viral Failure'
                        elif M > 51:
                            new_class = 'LLVR'
                        else:
                            new_class = 'DSVL'
                        break
            else:
                consec = True
        else:
            consec = False
    return new_class
    
def Rose_SMVL_omit(D,VL,w=24,ws=6):
    e,bnd = w*30,ws*30
    if VL[0] < 200:
        return 'Baseline < 200'
    dist = np.abs(D-e)
    best_i = np.argmin(dist)
    if dist[best_i] <= bnd:
        if VL[best_i] < 200:
            new_class = 'Suppressed'
        else:
            new_class = 'Not Suppressed'
    else:
        new_class = 'Omitted'
    return new_class
    
def Rose_SMVL_setfailure(D,VL,w=24,ws=6):
    e,bnd = w*30,ws*30
    if VL[0] < 200:
        return 'Baseline < 200'
    dist = np.abs(D-e)
    best_i = np.argmin(dist)
    if dist[best_i] <= bnd:
        if VL[best_i] < 200:
            new_class = 'Suppressed'
        else:
            new_class = 'Not Suppressed'
    else:
        new_class = 'Not Suppressed'
    return new_class
    
def Rose_SMVL_closest(D,VL,w=24,ws=6):
    e = w*30
    if VL[0] < 200:
        return 'Baseline < 200'
    dist = np.abs(D-e)
    best_i = np.argmin(dist)
    if VL[best_i] < 200:
        new_class = 'Suppressed'
    else:
        new_class = 'Not Suppressed'
    return new_class
    
def Rose_RMVL_continuous(D,VL,w=24,ws=6):
    e = w*30
    if VL[0] < 200:
        return 'Baseline < 200'
    VL = np.log10(VL+1)
    VLa = VL - VL[0]
    D2,VL2 = D.reshape((-1,1)),VLa.reshape((-1,1))
    LR = LinearRegression(False)
    LR.fit(D2,VL2)
    window_VL = LR.coef_[0][0]*e + VL[0]
    if window_VL < np.log10(201):
        new_class = 'Suppressed'
    else:
        new_class = 'Not Suppressed'
    return new_class
    
def Terzian_SHVL(D,VL):
    if np.max(VL) <= 400:
        return 'DSVL'
    consec,new_class = False,'Unspecified'
    for i in range(len(VL)):
        if VL[i] >= 100000:
            if consec:
                new_class = 'SHVL'
                break
            else:
                consec = True
        else:
            consec = False
    return new_class
    
def Phillips_rebound(D,VL):
    lwbnd,upbnd = 24*7,32*7
    VL2,D2 = VL[D <= upbnd],D[D <= upbnd]
    if len(D2) == 0:
        new_class = 'Omitted'
    elif np.max(D2-lwbnd) < 0:
        new_class = 'Omitted'
    elif np.min(VL2) >= 500:
        new_class = 'Viral Failure'
    else:
        if VL2[-1] < 500:
            new_class,consec = 'Suppressed',False
            for i in range(len(D2)):
                if VL2[i] >= 500:
                    if consec:
                        new_class = 'Viral Rebound'
                        break
                    else:
                        consec = True
                else:
                    consec = False
        else:
            new_class = 'Omitted'
    return new_class

def axis_test():
    fig = plt.figure()
    for i in range(4):
        ax = fig.add_subplot(2,2,i+1)
        ax.text(0.5,0.5,i)
    plt.show()

def custom_grid_lines(ax,xcoords,ycoords,color,lw = 0.5,ls ='-',a=0.5):
    xl,xr = ax.get_xlim()
    yb,yt = ax.get_ylim()
    for x in xcoords:
        ax.plot([x,x],[yb,yt],ls,color=color,lw=lw,alpha=a)
    for y in ycoords:
        ax.plot([xl,xr],[y,y],ls,color=color,lw=lw,alpha=a)
    ax.set_xlim(left=xl,right=xr)
    ax.set_ylim(bottom=yb,top=yt)
    
def same_axis(axs,rows,cols,samex=True,samey=True,xspine=True,yspine=False,
              order=None):
    if type(order) is types.NoneType:
        order = range(1,len(axs)+1)
    order_map = {order[i]:i for i in range(len(order))}
    if samey:
        for i in range(rows):
            found_first = False
            for j in range(cols):
                k = i*cols + j + 1
                if k in order_map:
                    if found_first:
                        adj_axis(axs[order_map[k]],{'left ytick':False,
                         'labelleft':False,'left spine':yspine})
                    else:
                        found_first = True
    if samex:
        for j in range(cols):
            found_first = False
            for i in range(rows)[::-1]:
                k = i*cols + j + 1
                if k in order_map:
                    if found_first:
                        adj_axis(axs[order_map[k]],{'bottom xtick':False,
                         'labelbottom':False,'bottom spine':xspine})
                    else:
                        found_first = True

def rev_log10_p10(nvl):
    int_nvl = int(nvl)
    if int_nvl == nvl: nvl = int_nvl
    return r'$10^{'+str(nvl)+'}-10$'

def recommend_opacity(lT,lY):
    mx,D = 2.0,{}
    for i in range(len(lT)):
        s = str((lT[i],lY[i]))
        try:
            D[s] += 1.0
        except KeyError:
            D[s] = 1.0
        if D[s] > mx:
            mx = D[s]
    return np.max([0.3,1/np.log2(mx)])

def my_cmaps(M, my_cmap = 'pink'):
    params = my_cmap.split(',')
    sm = np.min(M[M != 0])
    if params[0] == 'pink':
        cdict = {'red':  ((0.0, 0.5, 0.5), # maybe change 1.0 to 0.5
                          (sm, 1.0, 1.0),
                          (1.0, 1.0, 1.0)),
                 'green':((0.0, 1.0, 1.0),
                          (sm, 1.0, 1.0),
                          (1.0, 0.004, 0.004)),
                 'blue': ((0.0, 1.0, 1.0),
                          (sm, 1.0, 1.0),
                          (1.0, 0.573, 0.573))}
    elif params[0] == 'cool':
        cdict = {'red':  ((0.0, 1.0, 1.0),
                          (sm, 0.0, 0.0),
                          (1.0, 1.0, 1.0)),
                 'green':((0.0, 1.0, 1.0),
                          (sm, 1.0, 1.0),
                          (1.0, 0.0, 0.0)),
                 'blue': ((0.0, 1.0, 1.0),
                          (1.0, 1.0, 1.0))}
    elif params[0] == 'lightautumn':
        cdict = {'red':  ((0.0, 140.0/255, 140.0/255),
                          (sm, 1.0, 1.0),
                          (1.0, 1.0, 1.0)),
                 'green':((0.0, 1.0, 1.0),
                          (sm, 1.0, 1.0),
                          (0.5,175./255,175./255),
                          (1.0, 105./255, 105./255)),
                 'blue': ((0.0, 1.0, 1.0),
                          (sm, 175./255, 175./255),
                          (0.5, 105./255, 105./255),
                          (1.0, 105./255, 105./255))}
    elif params[0] == 'autumn':
        cdict = {'red':  ((0.0, 0.75, 0.75),
                          (0.15, 1.0, 1.0),
                          (1.0, 1.0, 1.0)),
                 'green':((0.0, 1.0, 1.0),
                          (0.15, 1.0, 1.0),
                          (1.0, 105./255, 105./255)),
                 'blue': ((0.0, 1.0, 1.0),
                          (0.15, 105./255, 105./255),
                          (1.0, 105./255, 105./255))}
    elif params[0] == 'teal':
        cdict = {'red':  ((0.0, 1.0, 1.0),
                          (sm, 1.0, 1.0),
                          (1.0, 0.0, 0.0)),
                 'green':((0.0, 0.78, 0.78),
                          (sm, 1.0, 1.0),
                          (1.0,135./255,135./255)),
                 'blue': ((0.0, 0.5, 0.5),
                          (sm, 1.0, 1.0),
                          (1.0,135./255,135./255))}
    elif params[0] == 'lightpurple':
        cdict = {'red':  ((0.0, 0.75, 0.75),
                          (0.35, 1.0, 1.0),
                          (1.0, 0.55, 0.55)),
                 'green':((0.0, 1.0, 1.0),
                          (0.35, 0.7, 0.7),
                          (1.0, 0.4, 0.4)),
                 'blue': ((0.0, 1.0,1.0),
                         (1.0, 1.0,1.0))}
    cmap = LinearSegmentedColormap(params[0]+'_cmap',cdict)
    C = {'green':[0.125, 0.75, 0.125],'black':[0,0,0],'gray':[0.4,0.4,0.4],
         'lime':[0.8, 1.0, 0.41],'lightviolet':[105./255,105./255,1.0],
         'lightcyan':[140./255,1.0,1.0],'red':[1.0,0,0],
         'mgold':[212./255,175./255,55./255],'copper':[1.0,0.78,0.5],
         'bloodred':[137./255,0.0,0.0],'brown':[0.545,0.27,0.075]}
    if len(params) > 1:
        crgb = C[params[1]]
        return cmap,crgb
    return cmap

def binned_time_series(T,Y,ax,mint=0.0,maxt=1863.2,miny=1.0,maxy=7.0,
        matrix_shape=(12,28),xtl=lambda x:x,ytl=lambda x:x,xts=14,yts=14,
        halfxticks=False,halfyticks=False,plot_last=False,cmap='default'):
    m,n = matrix_shape
    tadj,yadj = maxt-mint,maxy-miny
    cM = np.zeros(matrix_shape)
    lT,lY = [],[]
    for i in range(len(T)):
        for j in range(len(T[i])):
            rw = int(np.floor(((Y[i][j]-miny)/yadj)*matrix_shape[0]))
            if rw >= matrix_shape[0]: rw = matrix_shape[0]-1
            rw = (matrix_shape[0]-1)-rw
            cl = int(np.floor(((T[i][j]-mint)/tadj)*matrix_shape[1]))
            if cl >= matrix_shape[1]: cl = matrix_shape[1]-1
            cM[rw,cl] += 1.0
            if (j+1) == len(T[i]) and plot_last:
                lT.append(cl),lY.append(rw)
    for j in range(matrix_shape[1]):
        cM[:,j] = np.log(cM[:,j]+1.0)
        mx = np.max(cM[:,j])
        if mx > 0:
            cM[:,j] /= np.max(cM[:,j])
    if type(cmap) is str:
        if cmap == 'default':
            ax.imshow(cM)
            scttr_clr = [1.0,0.25,0]
        else:
            ccmap,scttr_clr = my_cmaps(cM, cmap)
            ax.imshow(cM, interpolation='nearest', cmap = ccmap)
    else:
        ax.imshow(cM, interpolatin='nearest', cmap = cmap)
    if plot_last:
        opacity = recommend_opacity(lT,lY)
        ax.scatter(lT,lY,color=scttr_clr,alpha=opacity,marker='.')
    yt,xt = dc(ax.get_yticks()),dc(ax.get_xticks())
    if halfxticks:
        xt = [xt[2*i+1] for i in range(len(xt)/2)]
        ax.set_xticks(xt)
    if halfyticks:
        yt = [yt[2*i+1] for i in range(len(yt)/2)]
#    custom_grid_lines(ax,[i-0.5 for i in range(n)],
#                          [i-0.5 for i in range(m)],'w',a=0.2)
    ax.set_xticks([x-0.5 for x in xt])
    ax.set_yticks([y-0.5 for y in yt])
    ax.set_xticklabels([xtl(np.round((x/n)*tadj+mint,2)) for x in xt],
                        fontsize=xts)
    ax.set_yticklabels([ytl(((m-y)/m)*yadj+miny) for y in yt],fontsize=yts)
    ax.set_xlim((-0.5,n-0.5)),ax.set_ylim((m-0.5,-0.5))
    return ax

def matspace(start,stop,step,rows=1):
    L = np.linspace(start,stop,step)
    M = np.zeros((rows,len(L)))
    for r in range(rows): M[r] = L
    return M

def comparative_patterns(VLpathT,my_cmap = 'pink,green',save = False):
    groups = ['Greub','SMVL omit','SMVL s2f','SMVL closest','RMVL cont',
              'Terzian','Phillips','Our Analysis']
    funcs = [Greub_LLVR,Rose_SMVL_omit,Rose_SMVL_setfailure,
             Rose_SMVL_closest,Rose_RMVL_continuous,Terzian_SHVL,
             Phillips_rebound]
    order= [['DSVL','LLVR','Viral Failure','Unspecified'],
            ['Baseline < 200','Suppressed','Not Suppressed','Omitted'],
            ['Baseline < 200','Suppressed','Not Suppressed'],
            ['Baseline < 200','Suppressed','Not Suppressed'],
            ['Baseline < 200','Suppressed','Not Suppressed'],
            ['DSVL','SHVL','Unspecified'],
            ['Suppressed','Viral Rebound','Viral Failure','Omitted'],
            ['DSVL','SLVL','SHVL','HVLS','RVL']]
    window = [[],[720,180],[720,180],[720,0],[720,0],[],[224,56],[]]
    wp = [False,'both','both','single','single',False,'lower',False]
    rows,cols = len(order),max([len(order[i]) for i in range(len(order))])
    Pvl,Pd,Pvllog = {},{},{}
    for path in VLpathT:
        VL,D = path.as_list(None,False,True,True)
        VL = np.array(VL)
        Pd[path.patient_id] = np.array(D) - D[0]
        Pvl[path.patient_id] = np.array(VL)
        Pvllog[path.patient_id] = tls(VL,'l10')
    fig = plt.figure(figsize=(19,18)) # 16,16 without ours.
    axs,a_order,lst_axs = [],[],[]
    ax = fig.add_subplot(rows*2,1,1)
    Ml = matspace(0,1,28*cols,2)
    ccmap,scttr = my_cmaps(Ml,my_cmap)
    ax.imshow(Ml,interpolation='nearest',cmap=ccmap,zorder=2)
    adj_axis(ax,{'labelleft':False,'labelbottom':False,'left ytick':False,
                 'bottom xtick':False,'xticks':[0,28*cols-1],'top xtick':True,
                 'xticklabels':(['0%','100%'],14),'labeltop':True})
    ax.set_title('Percent of log(VLM counts + 1) relative'+\
                 ' to the max(log(VLM counts + 1)) in column',fontsize=16)
    pos1 = ax.get_position()
    ax.set_position([pos1.x0,pos1.y0,pos1.width*0.8,pos1.height])
    ax = fig.add_subplot(rows*2,cols,cols*2)
    ax.scatter(0.015,0.02,color=scttr,marker='.')
    ax.text(0.1,0.02,'A terminal VLM\n'+r'(opacity $\geq$ 0.3)',fontsize=14,
            verticalalignment='center',zorder=1)
    adj_axis(ax,{'all ticks':False,'labelbottom':False,'labelleft':False})
    ax.set_xlim(left=-.05,right=0.65)
    p2 = ax.get_position()
    p=[p2.x0+p2.width*0.1,pos1.y0+p2.height*0.3,p2.width,p2.height*0.8]
    ax.set_position(p)
    for i in range(len(groups)):
        VLpath = dc(VLpathT)
        if i == (len(groups)-1):
            class2patient={c:VLpathT.pat_by_class[c].keys() for c in order[i]}
        else:
            class2patient = {c:[] for c in order[i]}
            for path in VLpath:
                D,VL = Pd[path.patient_id],Pvl[path.patient_id]
                class2patient[funcs[i](D,VL)].append(path.patient_id)
        #axi = i + 1 if i == (len(groups)-1) else i
        for j in range(len(order[i])):
            ax = fig.add_subplot(rows,cols,cols*i+j+1)
            a_order.append(cols*i+j+1)
            clsVL,clsD = [],[]
            for p_id in class2patient[order[i][j]]:
                clsVL.append(Pvllog[p_id]),clsD.append(Pd[p_id])
            axs.append(binned_time_series(np.array(clsD),np.array(clsVL),ax,
                ytl=rev_log10_p10,halfyticks=True,halfxticks=True,
                plot_last=True,cmap=my_cmap))
            if i == (len(groups)-1):
                ax.set_title(order[i][j],fontsize=16,color='red')
                dv = 1.0
            else:
                ax.set_title(order[i][j],fontsize=16)
                dv = 2.0
            pos1 = ax.get_position()
            pos2=[pos1.x0,pos1.y0-(pos1.height/dv),pos1.width,pos1.height]
            ax.set_position(pos2)
            if (j+1) == len(order[i]):
                lst_axs.append(ax)
    same_axis(axs,rows,cols,yspine=True,order=a_order)
    bbx0,rose = lst_axs[-1].get_position(),[]
    for i in range(len(lst_axs)):
        bbx = lst_axs[i].get_position()
        y = (bbx.intervaly[0]+bbx.intervaly[1])/2.0
        xa,skip = 0.04,False
        if (i >= 1) and (i <=4):
            xa = 0.0175
            rose.append(y)
            skip = True
        x = bbx0.intervalx[1]+xa # 0.025
        fig.text(x,y,groups[i],rotation=270,size=16,
             horizontalalignment='center',verticalalignment='center')
        if not skip:
            mlt = 13 if i == (len(lst_axs)-1) else 10
            fig.text(x-0.005,y,'_'*mlt,rotation=270,size=16,color='red',
                     horizontalalignment='center',verticalalignment='center',
                     weight='heavy')
    x,y = bbx0.intervalx[1]+xa,np.mean(rose)
    fig.text(x,y,'Rose',verticalalignment='center',rotation=270,size=16,
             horizontalalignment='center')
    fig.text(x-0.005,y,'_'*58,rotation=270,size=16,weight='heavy',color='red',
             horizontalalignment='center',verticalalignment='center')
    fig.text(0.04,0.5,'Viral Load (copies/ml)',rotation=90,size=18,
             horizontalalignment='center',verticalalignment='center')
    fig.text(0.5,0.01,'Days Since First Viral Load Measurement',size=18,
             horizontalalignment='center',verticalalignment='center')
    plt.show()
    if save: fig.savefig(save,bbox_inches='tight')


###############################################
######## Classification Stability #############
###############################################

def MLmodel(alg = 'DecisionTree', trees = 150, w = get_feat_weights()):
    if alg == 'Logistic':
        MLmodel = LogisticRegression()
    elif alg[0:2] == 'RF':
        crit = 'gini' if len(alg) == 2 else alg[2:]
        MLmodel = RFC(n_estimators=trees,criterion = crit, n_jobs=-1)
    elif alg == 'SVC':
        MLmodel = svm.SVC(probability = True,kernel='linear')
    elif alg == 'LDA':
        MLmodel = LDA()
    elif alg == 'QDA':
        MLmodel = QDA()
    elif alg[0:2] == 'kN':
        k = 7 if len(alg) == 3 else int(alg[3:])
        MLmodel = kNN(n_neighbors = k)
    elif alg == 'AdaBoost':
        MLmodel = ABC(n_estimators=trees)
    elif alg == 'DecisionTree':
        MLmodel = DTC()
    elif alg == 'NeuralNet':
        MLmodel = Backpropogation(max_iter = trees*50)
    elif alg[0:2] == 'Ce':
        params = alg.split(' ')
        method = 'average'
        predictor = 'radius'
        if len(params) >= 2: 
            method = params[1]
            if len(params) == 3:
                predictor = params[2]
        MLmodel = Centroid(method,weights = w,predictor = predictor)
    else:
        raise AssertionError('There is no such algorithm as: '+alg)
    return MLmodel

def LOOCV(alg,Matrix,Classes,pred = 'predict'):
    result,Q = [],range(len(Matrix))
    for i in range(len(Matrix)):
        q = Q.pop()
        ML = MLmodel(alg)
        ML = ML.fit(Matrix[Q,:],Classes[Q])
        if pred == 'predict':
            r = ML.predict(Matrix[q].reshape(1,-1))
        elif pred == 'predict_proba':
            r = ML.predict_proba(Matrix[q].reshape(1,-1))[0]
        result.append(r)
        Q = [q] + Q
    return np.array(result[::-1])

def get_training_data(VLpath,return_dummy=False,upsample=False):
    M,true_c,dummy_c,T,C=[],[],[],{},[k for k in VLpath.classes if k != None]
    for i in range(len(C)):
        T[i],T[C[i]] = C[i],i
    for c in C:
        p_ids = VLpath.pat_by_class[c].keys()
        P = np.random.choice(p_ids,upsample) if upsample else p_ids
        for p_id in P:
            feat = patient_feats(VLpath,p_id)
            M.append(feat),true_c.append(c),dummy_c.append(T[c])
    M,true_c,dummy_c = np.array(M),np.array(true_c),np.array(dummy_c)
    if return_dummy:
        return M,true_c,dummy_c,T
    return M,true_c

def roudn(FLOAT,decimal_point,as_string = False):
    if (type(FLOAT) is list) or (type(FLOAT) is np.ndarray):
        new_L = []
        for i in range(len(FLOAT)):
            new_L.append(roudn(FLOAT[i],decimal_point,as_string))
        if type(FLOAT) is list:
            return new_L
        else:
            return np.array(new_L)
    if type(FLOAT) is not float: FLOAT = float(FLOAT)
    S = str(FLOAT)
    p = S.find('.')
    if len(S) <= p+decimal_point+1:
        if as_string: return str(FLOAT)
        return float(S)
    if int(S[p+decimal_point+1]) < 5: 
        if as_string: return S[0:p+decimal_point+1]
        return float(S[0:p+decimal_point+1])
    i = p+decimal_point
    while (i == p) or ((S[i] == '9') and (i != 0)): i -= 1
    if as_string: return S[0:i]+str(int(S[i])+1)
    return float(S[0:i]+str(int(S[i])+1))

def covariance_matrix(M,correlation=True,save_table=False):
    mean_diff,stds = [],[]
    m,n = np.shape(M)
    covariance = np.zeros((int(n),int(n)))
    for j in range(n):
        mean = np.mean(M[:,j])
        mean_diff.append(M[:,j] - mean)
        if correlation:
            stds.append(np.sqrt(sum(mean_diff[j]**2.0)/(m-1)))
    for i in range(n):
        for j in range(i,n):
            cov = sum(mean_diff[i]*mean_diff[j])/(m-1)
            if correlation:
                cov = cov / (stds[i]*stds[j])
            covariance[i,j] = cov
            covariance[j,i] = cov
    if save_table:
        a = open("Table S1.csv",'wb')
        b = csv.writer(a)
        fn = get_feat_names()
        b.writerow([''] + fn)
        for i in range(len(fn)):
            b.writerow(np.append([fn[i]],roudn(covariance[i],4)))
    return covariance

def trainingVL_plot(VLpath,M,true_c,pred_c,save):
    names = get_feat_names()
    save = False if type(save) is bool else save
    markers,D = [],{n:[] for n in names}
    for feat in M: 
        for i in range(len(names)): D[names[i]].append(feat[i])
    for i in range(len(true_c)):
        markers.append('o') if true_c[i] == pred_c[i] else markers.append('x')
    feat_vs_feat(VLpath, (D,pred_c), save, markers, True)
    return np.array(markers)

def dummy_to_list(dummy,mxdummy):
    L = [0 for i in range(mxdummy)]
    L[dummy] = 1
    return L

def trainingVL(VLpath,alg='kNN7',plot=False,use_LOOCV=False,given=False):
    if given:
        M,true_c,dummy_c,T = given
    else:
        M,true_c,dummy_c,T = get_training_data(VLpath,True)
    if use_LOOCV:
        if plot: pred_c = LOOCV(alg,M,true_c)
        else: pred_c = LOOCV(alg,M,dummy_c,'predict_proba')
    else:
        ML = MLmodel(alg)
        ML = ML.fit(M,true_c) if plot else ML.fit(M,dummy_c)
        pred_c = ML.predict(M) if plot else ML.predict_proba(M)
    if plot: return trainingVL_plot(VLpath,M,true_c,pred_c,plot)
    scores,TP,FP,Trues = {},{},{},[]
    for i in range(len(pred_c.T)): scores[T[i]],TP[i],FP[i] = [],0.0,0.0
    for i in range(len(pred_c)):
        Trues.append(dummy_to_list(dummy_c[i],len(pred_c.T)))
        pred = np.argmax(pred_c[i])
        if pred == dummy_c[i]: TP[pred] += 1.0
        else: FP[pred] += 1.0
    Trues = np.array(Trues)
    for i in range(len(pred_c.T)):
        scores[T[i]].append(roudn(roc_auc_score(Trues.T[i],pred_c.T[i]),4))
        precision,recall = TP[i] / (TP[i] + FP[i]),TP[i] / VLpath.classes[T[i]]
        F1 = (2*precision*recall)/(precision+recall)
        scores[T[i]].append(roudn(precision,4))
        scores[T[i]].append(roudn(recall,4))
        scores[T[i]].append(roudn(F1,4))
    return scores

def generate_table2(VLpath,specifically=False,Cpredictor='radius'):
    algs = np.array(['median','average','boundingbox','smallestdisk',
            'bestrep','poly','push&pull','kNN5','kNN7','kNN9',
            'SVC','DecisionTree','AdaBoost','RF'])
    if type(specifically) is list:
        algs = algs[specifically[0]:specifically[1]]
        alg_num = specifically[0]
    elif type(specifically) is int:
        algs = [algs[specifically]]
        alg_num = specifically
    else:
        alg_num = 0
    M,true_c,dummy_c,T = get_training_data(VLpath,True)
    ML = MLmodel('Centroid LT')
    ML = ML.fit(M,true_c)
    nM = ML.LT(M,True)
    given_norm = (nM,true_c,dummy_c,T)
    given_unnorm = (M,true_c,dummy_c,T)
    Scores = {}
    for alg in algs:
        print "----------------------------------------------"
        print "Currently performing LOOCV on method: "+alg
        if alg_num >= 7:
            A = alg
            given = given_norm
        else:
            A = 'Centroid '+alg+' '+Cpredictor
            given = given_unnorm
        s = time()
        try:
            print A
            Scores[alg] = trainingVL(VLpath,A,False,True,given)
        except AssertionError:
            print "The center is not inside convex hull, skipping..."
            alg_num += 1
            continue
        e = time()-s
        m,h = e / 60.0, e / 3600.0
        print "Finished in.. "
        print str(e)+" seconds | "+str(m)+" minutes | "+str(h)+" hours."
        alg_num += 1
    return Scores

def write_table2(Scores):
    a = open('Table2.csv','wb')
    b = csv.writer(a)
    Names = ['DSVL','SLVL','SHVL','HVLS','Rebounding']
    Avg_scores = {}
    for alg,data in Scores.items():
        avg = 0.0
        for L in data.values():
            avg += L[3]
        avg /= len(Names)
        Avg_scores[alg] = avg
    srtd = sorted(Avg_scores.items(),key=op.itemgetter(1))
    for i in range(len(Scores)):
        alg,avg = srtd.pop()
        L = [alg]
        for name in Names:
            L.append(Scores[alg][name][3])
        L.append(avg)
        b.writerow(L)

def generate_table3(VLpath):
    algs = ['median','average','bounding box','smallest disk','best rep',
            'poly','push and pull']
    Centroids = {}
    w = get_feat_weights()
    M,true_c = get_training_data(VLpath)
    for alg in algs:
        print "----------------------------------------------"
        print "Currently finding centers and radii on method: "+alg
        C = Centroid(alg,w)
        C = C.fit(M,true_c)
        Centroids[alg] = [C.centers_,C.radii_,C.get_inv_centers(),C.LT_]
    return Centroids

def write_table3(Centroids_Res,alg_of_choice):
    D = Centroids_Res[alg_of_choice]
    Names = ['DSVL','SLVL','SHVL','HVLS','Rebounding']
    ft_nms = get_feat_names()
    blank_line = ['' for i in range(len(ft_nms))]
    a = open('Table3.csv','wb')
    b = csv.writer(a)
    ft_nms = get_feat_names()
    b.writerow(ft_nms)
    for name in Names:
        b.writerow(roudn(D[2][name],4,True))
    b.writerow(blank_line)
    if type(D[3]) is not dict:
        for j in range(2):
            b.writerow(roudn(D[3][j],4,True))
    else:
        for line_type in ['p','n']:
            for j in range(2):
                line = []
                for i in range(len(ft_nms)):
                    try:
                        line.append(roudn(D[3][i][line_type][j],4,True))
                    except KeyError:
                        line.append('-')
                b.writerow(line)
    blank_line.append('Radius')
    b.writerow(blank_line)
    for name in Names:
        row = list(D[0][name])
        row.append(D[1][name])
        b.writerow(roudn(row,4,True))

def write_supplmentarycenters(Centroids_Res):
    a = open('Supplementary_Centers.csv','wb')
    b = csv.writer(a)
    algs = ['smallest disk','best rep','bounding box','poly','average',
            'median','push and pull']
    Names = ['DSVL','SLVL','SHVL','HVLS','Rebounding']
    l = len(Centroids_Res[algs[0]][0][Names[0]])
    for alg in algs:
        for i in range(l+1):
            row = []
            radius = True if i == l else False
            for name in Names:
                if radius:
                    row.append(roudn(Centroids_Res[alg][1][name],4))
                else:
                    row.append(roudn(Centroids_Res[alg][0][name][i],4))
            b.writerow(row)

def empty_count(C):
    return {str(c):0 for c in C}

def stacked_barplot(x,y,ax,C,thisC,include_thisC=False,avg=False,title=True):
    try:
        step,max_y,total_pats = x[1]-x[0],0,[]
    except IndexError:
        print x
    string_C = [str(c) for c in C]
    st_tc = str(thisC)
    for i in range(len(x)):
        y_i,total = 0,float(sum(y[i].values())) if avg else 1
        total_pats.append(total)
        if avg and (total == 0): continue
        if avg and include_thisC:
            ax.add_patch(patches.Rectangle((x[i],y_i),step,y[i][st_tc]/total,
                         facecolor=thisC,linewidth=0))
            y_i += y[i][st_tc]/total
            max_y = y_i if y_i > max_y else max_y
        for j in range(len(C)):
            s,c = string_C[j],C[j]
            if (y[i][s] == 0) or ((s == st_tc) and (not include_thisC or avg)):
                continue
            ax.add_patch(patches.Rectangle((x[i],y_i),step,y[i][s]/total,
                         facecolor=c,linewidth=0))
            y_i += y[i][s]/total
        max_y = y_i if y_i > max_y else max_y
    if avg: ax.plot([x[0],x[-1]],[0.8,0.8],'--',color='w')
    ax.set_xlim(left=x[0],right=x[-1]+step)
    ax.set_ylim(bottom = 0,top=max_y)
    if title:
        if type(title) is bool: ax.set_title(thisC,size=14)
        else: ax.set_title(title,size=14)
    adj_axis(ax,{'standard':True,'tick labelsize':14})
    return ax,total_pats

def validatingVL(VLpath,disp='auc',alg='kNN7',upsmp=False,step=15,mxdays=1876,
                 specific_save_name = False,return_scores = False):
    auc,P=True if 'auc' in disp else False,True if mxdays == 'prop' else False
    days_list=np.linspace(0,1,1001) if P else range(90,mxdays,step)
    M,true_c,dummy_c,trns = get_training_data(VLpath,True,upsmp)
    ML,C,inst = MLmodel(alg),[k for k in VLpath.classes if k != None],-1
    Clr_L = [VLpath.class_colors[trns[i]] for i in range(len(C))]
    params = alg.split(' ')
    if params[0] != 'Centroid':
        LT = MLmodel('Centroid LT')
        LT = LT.fit(M,true_c)
        M = LT.LT(M)
    ML,Cs,E = ML.fit(M,dummy_c),len(C),True if 'incomplete' in disp else False
    scores,patients_used = [[] for i in range(len(C))],[]
    X = 'Days Since First Viral Load Measurement'
    if P: X = r'Proportion of Retained Information ($ri$)'
    for i in range(len(days_list)):
        vM,vC,days = [],[],days_list[i]
        for path in VLpath:
            feat = patient_feats(VLpath,path.patient_id,days,E=E)
            if not feat: continue
            vM.append(feat)
            if auc:
                vC.append(dummy_to_list(trns[path.Class],len(C)))
            else:
                vC.append(trns[path.Class])
        vM,vC = np.array(vM),np.array(vC)
        patients_used.append(len(vM))
        if len(vM) == 0: 
            for i in range(len(scores)):
                if auc:
                    scores[i].append(0)
                else:
                    scores[i].append(empty_count(Clr_L))
            inst += 1
            continue
        if params[0] != 'Centroid':
            vM = LT.LT(vM)
        if auc:
            vP = ML.predict_proba(vM)
            vC,vP = vC.T,vP.T
            for i in range(len(C)):
                scores[i].append(roc_auc_score(vC[i],vP[i]))
        else:
            vP = ML.predict(vM) if 'abs' in disp else ML.predict_proba(vM)
            for i in range(len(scores)):
                scores[i].append(empty_count(Clr_L))
            inst += 1
            if 'abs' in disp:
                if type(vP) is not np.ndarray: vP = [vP]
                for i in range(len(vC)):
                    scores[vC[i]][inst][str(Clr_L[vP[i]])] += 1
            else:
                for i in range(len(vC)):
                    for j in range(len(C)):
                        scores[vC[i]][inst][str(Clr_L[j])] += vP[i][j]
    if return_scores == 'only':
        return scores
    if auc:
        fig = plt.figure(figsize=(15.5,8.5))
        ax = fig.add_subplot(111)
        for i in range(len(C)):
            ax.plot(days_list,scores[i],color=trns[i],lw=2)
        adj_axis(ax,{'tick labelsize':14,'standard':True})
        ax.set_xlim(left=days_list[0],right=days_list[-1])
        ax = fig.add_subplot(111,frame_on = False)
        ax.plot(days_list,patients_used,'--',color='gray')
        adj_axis(ax,{'all off':True,'labelright':True,'right spine':True,
                     'right ytick':True,'tick labelsize':14})
        ax.set_xlim(left=days_list[0],right=days_list[-1])
    else:
        rows = int(np.ceil(len(C)/2.0))
        fig = plt.figure(figsize=(15.5,4.5*rows))
        avg = True if 'avg' in disp else False
        for i in range(len(C)):
            ax = fig.add_subplot(rows,2,i+1)
            ax,T=stacked_barplot(days_list,scores[i],ax,Clr_L,
                 VLpath.class_colors[trns[i]],True,avg,trns[i])
            if i % 2: adj_axis(ax,{'labelleft':False,'left spine':False,
                                   'left ytick':False})
            if i<Cs-2: adj_axis(ax,{'labelbottom':False,'bottom xtick':False})
            ax2 = fig.add_subplot(rows,2,i+1,frame_on = False)
            ax2.plot(days_list,T,color='w',lw=2)
            ax2.set_xlim(left=days_list[0],right=days_list[-1])
            adj_axis(ax2,{'all off':True})
            adj_axis(ax2,{'labelright':True,'right spine':True,
                          'right ytick':True,'tick labelsize':14})
        Y = ' Probability' if 'avg' in disp else ''
        fig.text(0.08,0.5,'Membership Assignment'+Y,rotation=90,size=18,
                 verticalalignment='center',horizontalalignment='center')
        fig.text(0.5,0.075,X,size=18,
                 horizontalalignment='center',verticalalignment='center')
        fig.text(0.945,0.5,'Number of Patients in Class',rotation=270,size=18,
                 verticalalignment='center',horizontalalignment='center')
        add_legend(VLpath,fig,size=18)
    plt.show()
    ncplt,P = ' incomplete' if E else '',' proportion' if P else ''
    if specific_save_name:
        fig.savefig(specific_save_name)
    else:
        fig.savefig('Class Validation with '+alg+ncplt+P+'.pdf')
    if return_scores:
        return scores

# validatingVL(VLpathT,'abs avg','Centroid poly',mxdays='prop')

def get_scores(VLpath):
    print "Retrieving scores for: Centroid Polyhedron Radial Norm"
    cpr = validatingVL(VLpath,'abs avg','Centroid poly',mxdays='prop',
                       return_scores = 'only')
    print "Retrieving scores for: Decision Tree"
    dt = validatingVL(VLpath,'abs avg','DecisionTree',mxdays='prop',
                      return_scores = 'only')
    print "Retrieving scores for: SVM"
    svm = validatingVL(VLpath,'abs avg','SVC',mxdays='prop',
                       return_scores = 'only')
    print "Retrieving scores for: k-Nearest Neighbors, k = 5"
    kNN5 = validatingVL(VLpath,'abs avg','kNN5',mxdays='prop',
                        return_scores = 'only')
    print "Retrieving scores for: Centroid Polyhedron Projected Hyp Norm"
    cpp = validatingVL(VLpath,'abs avg','Centroid poly projection',
                       mxdays='prop',return_scores='only')
    Names = ['Polyhedral CMuRN','Decision Tree','SVM','k-NN,k=5',
             'Polyhedral CMuPHN']
    return [cpr,dt,svm,kNN5,cpp],Names

#scores,score_names = get_scores(VLpathT)

def sig_range_wilcoxon(x, rep = 1000):
    # There is currently something wrong with this function... nonetheless
    # it is not needed.
    s,p0 = stats.wilcoxon(x)
    already_sig = True if p0 < 0.05 else False
    median = np.median(x)
    bottom,top = None,None
    if (median > 0) or already_sig:
        mx = max(x)
        I = np.linspace(0.0,mx,rep)
        for i in range(1,rep):
            s,p = stats.wilcoxon(x-I[i])
            if (p < 0.05 and not already_sig) or (p >= 0.05 and already_sig):
                break
        if not already_sig:
            bottom = I[i]
            for j in range(i,rep):
                s,p = stats.wilcoxon(x-I[j])
                if (p >= 0.05):
                    break
            top = I[j-1]
        else:
            top = I[i-1]
    if (median < 0) or already_sig:
        mn = min(x)
        I = np.linspace(0.0,mn,rep)
        for i in range(1,rep):
            s,p = stats.wilcoxon(x+I[i])
            if (p < 0.05 and not already_sig) or (p >= 0.05 and already_sig):
                break
        if not already_sig:
            top = I[i]
            for j in range(i,rep):
                s,p = stats.wilcoxon(x+I[i])
                if (p >= 0.05):
                    break
            bottom = I[j-1]
        else:
            bottom = I[i-1]
    return bottom,top
        
def wilcoxon_test(VLpath,scores,score_names,save=False,save_table=False):
    rd = {}
    for c,color in VLpath.class_colors.items():
        rd[str(color)] = c
    M,true_c,dummy_c,trns = get_training_data(VLpath,True)
    data = {n:{} for n in score_names}
    st_tc,IQR_results = [],{c:[c] for c in VLpath.classes}
    titles = dc(score_names)
    vs = titles[0]
    titles[0] == ""
    for i in range(1,len(titles)):
        titles[i] = r'$x = $'+titles[i]
    if save_table:
        a = open('Table S2.csv','wb')
        b = csv.writer(a)
        table_titles = dc(score_names)
        table_titles[0] = ""
        b.writerow(table_titles)
    for c in VLpath.classes:
        if c != None: st_tc.append(str(VLpath.class_colors[c]))
    Diff = {ni:{} for ni in range(1,len(score_names))}
    fig = plt.figure(figsize=(12,9))
    nr,nc = num_of_rows_and_cols(len(score_names)-1)
    for ni in range(len(score_names)):
        bxs,cs,p_vals,rng = [],[],[],[]
        for c in VLpath.classes:
            if c == None:
                continue
            arr = np.array([])
            thisC = str(VLpath.class_colors[c])
            for i in range(len(scores[0][0])):
                summ = 0.0
                for st in st_tc:
                    summ += scores[ni][trns[c]][i][st]
                if summ >= 20:
                    arr = np.append(arr,scores[ni][trns[c]][i][thisC]/summ)
            data[score_names[ni]][c] = arr
            if ni > 0:
                Diff[ni][c] = data[score_names[ni]][c]-data[score_names[0]][c]
                bxs.append(Diff[ni][c]),cs.append(c)
                stat,p = stats.wilcoxon(Diff[ni][c])
                Q1 = roudn(np.percentile(Diff[ni][c],25),4)
                Q3 = roudn(np.percentile(Diff[ni][c],75),4)
                p_vals.append(p)
                rng.append((Q1,Q3))
        if ni > 0:
            print score_names[ni]
            print '================'
            for i in range(len(cs)):
                print cs[i] + ' p-val: '+str(p_vals[i])+' w/ IQR: '+str(rng[i])
                IQR_results[cs[i]].append(rng[i])
            print ''
            ax = fig.add_subplot(nr,nc,ni)
            bp = ax.boxplot(bxs,labels=cs,patch_artist=True,notch=True,
                            zorder=3)
            for i in range(len(cs)):
                bp['boxes'][i].set(facecolor=VLpath.class_colors[cs[i]])
            X = ax.get_xlim()
            Y = ax.get_ylim()
            ax.plot(X,[0,0],color=[0.7,0.7,0.7],zorder=2)
            ax.add_patch(patches.Rectangle((X[0],Y[0]),X[1]-X[0],0-Y[0],
                         facecolor=[0.9,0.9,0.9],alpha=0.3,zorder=1))
            ax.set_xlim(left=X[0],right=X[1])
            ax.set_ylim(bottom=Y[0],top=Y[1])
            ax.set_title(titles[ni],fontsize=18)
            adj_axis(ax,{'xtick labelsize':14,'ytick labelsize':14,
                         'neat':('x',nr,nc,len(score_names)-1,ni)})
    fig.text(0.04,0.5,r'$x - $'+vs,rotation=90,size=18,
                 verticalalignment='center',horizontalalignment='center')
    if save: fig.savefig(save)
    if save_table:
        for c in VLpath.classes:
            b.writerow(IQR_results[c])

def rrL(L):
    return [L[0],L[6],L[1],L[7],L[2],L[8],L[3],L[4],L[5]]
    #return [L[0],L[6],L[1],L[7],L[2],L[8],L[3],L[9],L[4],L[5]]

def sphere(ax,xcoord,ycoord,zcoord,radius,color,alpha):
    u = np.linspace(0, 2 * np.pi, 1000)
    v = np.linspace(0, np.pi, 1000)
    x = radius * np.outer(np.cos(u), np.sin(v)) + xcoord
    y = radius * np.outer(np.sin(u), np.sin(v)) + ycoord
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + zcoord
    
    # Plot the surface
    ax.plot_surface(x, y, z, color=color,alpha = alpha)

def centroid_plot(VLpath, Centroid, save = False):
    names,plot_num,iC,L,m = get_feat_names(),1,Centroid.get_inv_centers(),[],[]
    Ct = Centroid.centers_
    try: R = {c:[Centroid.radii_[c] for i in range(len(names))] for c in iC}
    except AttributeError: R = {c:[0.5 for i in range(len(names))] for c in iC}
    M,true_c = get_training_data(VLpath)
    pred_c = Centroid.predict(M)
    for i in range(len(M.T)): M.T[i] = one_transform(M.T[i])
    cc = VLpath.class_colors
    for i in range(len(true_c)):
        m.append('.') if true_c[i] == pred_c[i] else m.append('x')
    m = np.array(m)
    for i in range(len(names)):
        L.append([min(M.T[i]),max(M.T[i])])
    for c in R: R[c] = Centroid.inv_LT(R[c])
    amt = (len(names)*(len(names)-1))/2.0
    fig = plt.figure(figsize=(14,10))
    r,c = num_of_rows_and_cols(amt,False)
    dots,misl = m == '.',m=='x'
    dotc,misc = [cc[i] for i in true_c[dots]],[cc[i] for i in true_c[misl]]
    for i in range(len(names)):
        n1 = names[i]
        for j in range(len(names))[(i+1):]:
            ax,n2 = fig.add_subplot(r,c,plot_num),names[j]
            for C in Ct: 
                ax.add_patch(patches.Ellipse((Ct[C][i],Ct[C][j]),
                             Centroid.radii_[C],Centroid.radii_[C],
                                            color=cc[C],alpha=0.4))
                #ax.add_patch(patches.Ellipse((iC[C][i],iC[C][j]),
                #             R[C][i],R[C][j],color=cc[C],alpha=0.4))
            #ax.scatter(M.T[i][dots],M.T[j][dots],
            #           color=dotc,alpha=0.1,marker='.')
            ax.scatter(M.T[i][misl],M.T[j][misl],alpha=0.6,
                       color=misc,marker='x')
            #for x,y,clr,mrk in zip(M.T[i],M.T[j],true_c,m):
            #    a = 0.1 if mrk == '.' else 0.6
            #    ax.scatter(x,y,color=cc[clr],marker=mrk,alpha=a)
            #for C in iC: ax.scatter(iC[C][i],iC[C][j],marker=r'$+$',s=64,
            #                        color=cc[C])
            for C in Ct: ax.scatter(Ct[C][i],Ct[C][j],marker=r'$+$',s=64,
                                    color=cc[C])
            adj_axis(ax,{'xlabel':(n1,14),'ylabel':(n2,14),'tick labelsize':14,
                     'standard':True})
            ax.set_xlim(left=L[i][0],right=L[i][1])
            ax.set_ylim(bottom=L[j][0],top=L[j][1])
            plot_num += 1
    if amt == 3:
        ax = fig.add_subplot(2,2,4,projection='3d')
        for C in Ct:
            sphere(ax,Ct[C][0],Ct[C][1],Ct[C][2],Centroid.radii_[C],cc[C],0.4)
    ax = fig.add_subplot(111)
    Artists,Labels = generate_class_artists(VLpath)
    Artists += [plt.scatter([0],[0],c='gray',marker='.'),
                plt.scatter([0],[0],c='gray',marker='x'),
                plt.scatter([0],[0],64,c='gray',marker=r'$+$'),
                plt.scatter([0],[0],400,'gray',alpha=0.4)]
    Labels += ['correct','mislabeled','cluster center',
               'transformed cluster radius']
    fig.delaxes(ax)
    ax = fig.add_subplot(111)
    ax.legend(rrL(Artists),rrL(Labels),bbox_to_anchor=(0.,1.02,1., .102),\
       loc=3,ncol=len(Artists)-4,mode='expand',borderaxespad=0.25,fontsize=14)
    ax.axis('off')
    plt.show()
    if save: fig.savefig(save)

def rgb2gray(rgb):
    if np.any(rgb > 1):
        if type(rgb) is not np.ndarray:
            rgb = np.array(rgb)
        rgb = rgb/255.0
    return 0.2989 * rgb[0] + 0.5870 * rgb[1] + 0.1140 * rgb[2]

def generate_DTpdf(VLpath,color_change=False,save='FigS4.pdf'):
    M,true_c,dummy_c,T = get_training_data(VLpath,True)
    LT = MLmodel('Centroid LT')
    LT = LT.fit(M,dummy_c)
    nM = LT.LT(M,True)
    DT = DTC()
    DT = DT.fit(nM,dummy_c)
    # For new graphviz installers go to the website:
    # http://www.graphviz.org/Download_windows.php
    # Download .zip file then create a new folder (ie. Graphviz2.38)
    # Then run the next 2 lines to place it in your path:
    #os.environ["PATH"] += os.pathsep + \
    #'C:/Program Files (x86)/Graphviz2.38/bin/' #Change this line if elsewhere
    VLpatterns = []
    for i in range(len(VLpath.class_colors)):
        VLpatterns.append(T[i])
    dot_data = tree.export_graphviz(DT,out_file=None,rounded=True,label='root',
                                    feature_names=get_feat_names(),
                                    class_names=VLpatterns,filled=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    for node in graph.get_node_list():
        S = node.get_label()
        if S == None:
            continue
        S2 = S.split('\\n')
        S2 = S2[-1].split(' ')
        color = VLpath.class_colors[S2[-1][:-1]]
        if color_change:
            try:
                color = color_change[color]
            except KeyError:
                pass
        intcolor = [int(cl*255) for cl in color]
        node.set_fillcolor(webcolors.rgb_to_hex(intcolor))
        if rgb2gray(color) < 0.4:
            node.set_fontcolor('white')
    if save: graph.write_pdf(save)
    
#generate_DTpdf(VLpathT,
#               {'g':'green','y':'yellow','m':'magenta','r':'red','c':'cyan'})

    
    
