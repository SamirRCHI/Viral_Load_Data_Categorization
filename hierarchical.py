# -*- coding: utf-8 -*-
"""
Created on Fri Feb 02 14:40:02 2018

@author: sfarooq1
"""
import numpy as np
import matplotlib.pyplot as plt
from axishelper import adj_axis
from matplotlib.patches import Rectangle as Rect
import colorsys
import random

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

def gnu_colors(n):
    cmap = plt.get_cmap('gnuplot')
    colors = []
    for i in np.linspace(0,1,n):
        colors.append(cmap(i))
    return colors

def cluster_probabilities(predictor, cluster):
    T,count = float(len(cluster)),{}
    for c in cluster:
        try:
            count[predictor[c]] += 1
        except KeyError:
            count[predictor[c]] = 1
    probabilities = []
    for predictor_count in count.values():
        probabilities.append(predictor_count / T)
    return np.array(probabilities)
    
def summed_cluster_entropy(predictor, Clusters):
    e = 0
    for cluster in Clusters:
        p = cluster_probabilities(predictor, cluster)
        e += (len(cluster) / float(len(predictor))) * (-sum(p*np.log2(p)))
    return e

def norm_area(x,y):
    area = 0.0
    triangle = 0.5*y[-1]*x[-1]
    for i in range(1,len(x)): area += (y[i] + y[i-1])*(x[i]-x[i-1])/2.0
    return (area - triangle)/triangle

class linkage_clustering:
    def __init__(self, Z, thresh = None, classes = None, label = None):
        self.Clusters = []
        self.Z = Z
        self.p = len(self.Z)
        self.classes = classes
        self.thresh = thresh
        if type(label) is type(None):
            self.label = np.arange(self.p+1)
        else:
            self.label = np.array(label)
        self.perform_clustering()
    def perform_clustering(self, new_thresh = False):
        if new_thresh != False:
            self.thresh = new_thresh
        self.S = set(range(self.p+1))
        if self.thresh == None:
            self.Zt = self.Z
            self.thresh = np.max(self.Z[:,2])
        self.Zt = self.Z[self.Z[:,2] <= self.thresh, :]
        self.C,self.B,self.cn = set(range(len(self.Zt))),{},0
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
            clrs = gnu_colors(len(self.Clusters))
        for i in range(len(clrs)):
            self.colors[i] = clrs[i]
        self.def_bracket_color = np.array([0.65,0.65,0.65])
    def plot_segment(self,x,bot,top,color):
        if bot > self.thresh:
            if self.xorient:
                X = [x,x]
                Y = [bot,top]
            else:
                X = [bot,top]
                Y = [x,x]
            self.ax.plot(X,Y,color=self.def_bracket_color)
        elif top <= self.thresh:
            if self.xorient:
                X = [x,x]
                Y = [bot,top]
            else:
                X = [bot,top]
                Y = [x,x]
            self.ax.plot(X,Y,color=color)
        else:
            if self.xorient:
                X1,X2 = [x,x],[x,x]
                Y1,Y2 = [bot,self.thresh],[self.thresh,top]
            else:
                X1,X2 = [bot,self.thresh],[self.thresh,top]
                Y1,Y2 = [x,x],[x,x]
            self.ax.plot(X1,Y1,color=color)
            self.ax.plot(X2,Y2,color=self.def_bracket_color)
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
        if self.xorient:
            X,Y = [x1,x2],[top,top]
        else:
            X,Y = [top,top],[x1,x2]
        self.ax.plot(X,Y,color=new_color)
    def plot_dendrogram(self, ax = None, title = False, class_color = None,
                        label = False, new_thresh = False,orientation = 'top',
                        reverse_labelflush=False,color_label=True,yflip=True,
                        xtick_rotate = 0):
        if new_thresh != False:
            self.perform_clustering(new_thresh)
            cbf = True if len(self.Clusters) <= 8 else False
            self.generate_cluster_colors(cbf)
        self.lo = []
        self.clus_pos = {i:[float('inf'),0] for i in range(len(self.Clusters))}
        try:
            self.colors
        except AttributeError:
            cbf = True if len(self.Clusters) <= 8 else False
            self.generate_cluster_colors(cbf)
        self.pos,self.brackets = -5.0,{}
        if (ax == None) or (ax == 'small'):
            fig = plt.figure()
            self.ax = fig.add_subplot(111)
        elif ax == 'big':
            fig = plt.figure(figsize=(15.5,8.5))
            self.ax = fig.add_subplot(111)
        elif ax == 'med':
            fig = plt.figure(figsize=(10,6))
            self.ax = fig.add_subplot(111)
        else:
            self.ax = ax
        self.xorient = True
        if (orientation == 'right') or (orientation == 'left'):
            self.xorient = False
        self.update_bracket(self.p-1)
        lx = 10+10*self.p
        if self.xorient:
            self.ax.plot([-5,lx],[self.thresh,self.thresh],'--',
                         color=self.def_bracket_color)
            self.ax.set_xlim(left=0,right=lx)
        else:
            self.ax.plot([self.thresh,self.thresh],[-5,lx],'--',
                         color=self.def_bracket_color)
            self.ax.set_ylim(bottom=0,top=lx)
        if (self.classes != None) and (class_color != None):
            if self.xorient:
                h = 0.15*np.max(self.Z[:,2])
                y = 0.0 - h
                w = 10
            else:
                w = 0.15*np.max(self.Z[:,2])
                x = 0.0 - w
                h = 10
            for i in range(len(self.classes)):
                clr = class_color[self.classes[i]]
                if self.xorient:
                    x = self.brackets[i][0] - 5
                else:
                    y = self.brackets[i][0] - 5
                self.ax.add_patch(Rect((x,y),w,h,facecolor=clr,linewidth=0))
        elif self.classes != None:
            y = 0
        if self.xorient:
            self.ax.set_ylim(bottom = 0 if self.classes == None else y)
            adj_axis(self.ax,{'xticks':range(5,lx+1,10),'bottom xtick':False})
            if orientation == 'bottom':
                self.ax.invert_yaxis()
                if label:
                    adj_axis(self.ax,{'xticklabels':self.label[self.lo],
                                      'labeltop':True,'labelbottom':False})
                    if xtick_rotate:
                        for tick in self.ax.get_xticklabels():
                            tick.set_rotation(xtick_rotate)
            elif label:
                adj_axis(self.ax,{'xticklabels':self.label[self.lo]})
                if xtick_rotate:
                    for tick in self.ax.get_xticklabels():
                        tick.set_rotation(xtick_rotate)
            if not label:
                adj_axis(self.ax,{'xticklabels':False})
            if label and color_label:
                t = self.ax.xaxis.get_ticklabels()
                for i in range(len(t)):
                    if type(self.label[0]) is int:
                        txt = int(t[i].get_text())
                    else:
                        txt = t[i].get_text()
                    t[i].set_color(self.colors[self.B[txt]])
        else:
            self.ax.set_xlim(left = 0 if self.classes == None else x)
            adj_axis(self.ax,{'yticks':range(5,lx+1,10),'left ytick':False})
            if yflip:
                self.ax.invert_yaxis()
            if orientation == 'left':
                self.ax.invert_xaxis()
                if label:
                    side = 'right' if reverse_labelflush else 'left'
                    adj_axis(self.ax,{'yticklabels':(self.label[self.lo],side),
                                      'labelleft':False,'labelright':True})
                    if reverse_labelflush:
                        yax = self.ax.get_yaxis()
                        pad = max(T.label.get_window_extent(renderer=\
                            reverse_labelflush).width for T in yax.majorTicks)
                        yax.set_tick_params(pad = pad)
            elif label:
                side = 'left' if reverse_labelflush else 'right'
                adj_axis(self.ax,{'yticklabels':(self.label[self.lo],side)})
            if not label:
                adj_axis(self.ax,{'yticklabels':False})
            if label and color_label:
                t = ax.yaxis.get_ticklabels()
                for i in range(len(t)):
                    if type(self.label[0]) is int:
                        txt = int(t[i].get_text())
                    else:
                        txt = t[i].get_text()
                    t[i].set_color(self.colors[self.B[txt]])
        if title: self.ax.set_title(title, fontsize=14)
    def get_plot_boundaries(self):
        boundaries = {c:[float('inf'),None,-float('inf')] for c in self.colors}
        for i in range(self.p+1):
            b,c = self.brackets[i][0],self.B[self.label[i]]
            if b < boundaries[c][0]:
                boundaries[c][0] = b
            if b > boundaries[c][2]:
                boundaries[c][2] = b
        for c in boundaries:
            boundaries[c][1] = (boundaries[c][2] + boundaries[c][0])*0.5
        return boundaries

def cluster_gains(Z, classes, plot = False):
    original_e = summed_cluster_entropy(classes,[set(range(len(classes)))])
    uniq_thresh = sorted(Z[:,2],reverse=True)[1:]
    Cluster_len,Gains = [1],[0]
    for thresh in uniq_thresh:
        LC = linkage_clustering(Z, thresh)
        e = summed_cluster_entropy(classes, LC.Clusters)
        Cluster_len.append(len(LC.Clusters))
        Gains.append(original_e - e)
    Cluster_len.append(len(Z)+1)
    Gains.append(original_e)
    area = norm_area(Cluster_len,Gains)
    if plot:
        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(111)
        ax.plot(Cluster_len,Gains)
        ax.fill_between([1,Cluster_len[-1]],0,[0,Gains[-1]],
                        color='gray',alpha=0.2)
        AUC = 'Area under the curve: '+str(np.round(area*100,2))+'%'
        adj_axis(ax,{'xlabel':('Number of Clusters',14),'xtick labelsize':14,
                     'ylabel':('Information Gain',14),'ytick labelsize':14,
                     'standard':True,'title':(AUC,14)})
    return area
    