# -*- coding: utf-8 -*-
"""
Created on Tue May 22 14:54:37 2018

@author: sfarooq1
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as Rect
from matplotlib.patches import Polygon

class Node:
    def __init__(self, item):
        self.item = item
        self.reset()
    def reset(self):
        self.full_list = None
        self.is_head = True
        self.is_tail = True
        self.prev_node = None
        self.next_node = None
    def remove_self(self):
        if self.full_list != None:
            self.full_list.length -= 1
            if self.is_head:
                if self.next_node == None:
                    self.full_list.head = None
                else:
                    self.next_node.prev_node = None
                    self.next_node.is_head = True
                    self.full_list.head = self.next_node
            elif self.is_tail:
                self.prev_node.is_tail = True
                self.prev_node.next_node = None
                self.full_list.tail = self.prev_node
            else:
                self.prev_node.next_node = self.next_node
                self.next_node.prev_node = self.prev_node
            self.reset()
 
class Queue:
    def __init__(self):
        self.head = None
        self.tail = None
        self.length = 0
        self.stage = 0
        self.cur_node = None
    def __getitem__(self,key):
        if key >= 0:
            if key >= self.length:
                raise IndexError("Index Out of Bounds.")
            cur_item = 0
            cur_node = self.head
            while cur_item < key:
                cur_node = cur_node.next_node
                cur_item += 1
            return cur_node.item
        else:
            if key < -self.length:
                raise IndexError("Index Out of Bounds.")
            cur_item = -1
            cur_node = self.tail
            while cur_item > key:
                cur_node = cur_node.prev_node
                cur_item -= 1
            return cur_node.item
    def __iter__(self):
        return self
    def __next__(self):
        if self.stage == 0:
            if self.length == 0:
                raise StopIteration
            else:
                self.cur_node = self.head
                self.stage = 1
                return self.cur_node.item
        else:
            self.cur_node = self.cur_node.next_node
            if self.cur_node == None:
                self.stage = 0
                raise StopIteration
            else:
                return self.cur_node.item
    def add(self, node):
        self.length += 1
        node.full_list = self
        if self.head == None:
            self.head = node
            self.tail = node
        else:
            node.is_head = False
            self.tail.is_tail = False
            self.tail.next_node = node
            node.prev_node = self.tail
            self.tail = node
    def pop(self):
        if self.head == None:
            raise AssertionError("Nothing left to pop!")
        self.length -= 1
        old_head = self.head
        self.head = self.head.next_node
        if self.head != None:
            self.head.is_head = True
            self.head.prev_node.reset()
            self.head.prev_node = None
        return old_head.item
    def display(self):
        node = self.head
        while node != None:
            print(node.item)
            node = node.next_node

class Tree_Node:
    def __init__(self, feature, threshold, value):
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.class_choice = np.argmax(value)
        self.leaf_class_choices = {self.class_choice}
        self.is_left = False
        self.is_right = False
        self.left = None
        self.right = None
        self.parent = None
        self.sibling = None
        self.is_leaf = True
        self.depth = 0
        self.Node = Node(self)
    def set_children(self, left, right):
        left.sibling = right
        right.sibling = left
        self.is_leaf = False
        self.left = left
        self.right = right
        self.left.parent = self
        self.right.parent = self
        self.left.depth = self.depth + 1
        self.right.depth = self.depth + 1

def min_max_norm(v,oldmin,oldmax,newmin,newmax):
    return ((v - oldmin)/(oldmax - oldmin))*(newmax - newmin) + newmin

def Parentheses(xy,width,height,left_direction=True,fc='w',fill=True,ec='none',
                pts=50):
    s = 1 if left_direction else -1
    x,y = xy
    c1,c2,rx,ry = x, x + s*width/2, width/2, height/2
    d1 = np.pi/2 if left_direction else -np.pi/2
    d2 = 3*np.pi/2 if left_direction else np.pi/2
    theta = np.linspace(d1,d2,pts)
    x1 = rx*np.cos(theta) + c1
    x2 = rx*np.cos(theta[::-1]) + c2
    y1 = ry*np.sin(theta) + (y + height/2)
    y2 = y1[::-1]
    N = []
    for i in range(len(theta)):
        N.append([x1[i], y1[i]])
    N.append([c2, y1[-1]])
    for i in range(len(theta)):
        N.append([x2[i], y2[i]])
    N = np.array(N)
    return Polygon(N,fc=fc,fill=fill,ec=ec)

        
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

def txt(x,y,s,size_val,clr,ax,r):
    t = ax.text(x,y,s,size=size_val,color = clr)
    transf = ax.transData.inverted()
    bb = t.get_window_extent(renderer=r)
    bb_datacoords = bb.transformed(transf)
    x_axis = bb_datacoords.intervalx
    y_axis = bb_datacoords.intervaly
    coords = [x_axis[0],x_axis[1],y_axis[0],y_axis[1]]
    return t,coords

class DTrule_extraction:
    def __init__(self, DT):
        self.DT = DT
        self.set_up_tree_structure()
        self.update_class_choices()
        self.generate_rules()
    def set_up_tree_structure(self):
        left = self.DT.tree_.children_left
        right = self.DT.tree_.children_right
        feat = self.DT.tree_.feature
        thresh = self.DT.tree_.threshold
        values = self.DT.tree_.value
        self.leaves = Queue()
        self.head_node = Tree_Node(feat[0],thresh[0],values[0][0])
        self.max_depth = 0
        stack = [(0,self.head_node)]
        while len(stack) > 0:
            node_id,node = stack.pop()
            lid, rid = left[node_id], right[node_id]
            if lid != rid:
                left_node = Tree_Node(feat[lid],thresh[lid],values[lid][0])
                left_node.is_left = True
                right_node = Tree_Node(feat[rid],thresh[rid],values[rid][0])
                right_node.is_right = True
                node.set_children(left_node, right_node)
                stack.append((lid,left_node))
                stack.append((rid,right_node))
            else:
                self.leaves.add(node.Node)
    def update_class_choices(self):
        for leaf in self.leaves:
            node = leaf.parent
            all_choices = {leaf.class_choice}
            while node != None:
                all_choices = all_choices.union(node.leaf_class_choices)
                node.leaf_class_choices = all_choices
                node = node.parent
    def rule_of_node(self, input_node):
        L = [[None,None] for i in range(self.DT.max_features_)]
        node = input_node
        while True:
            if node.is_left:
                ft = node.parent.feature
                if L[ft][1] == None:
                    L[ft][1] = node.parent.threshold
                elif node.parent.threshold < L[ft][1]:
                    L[ft][1] = node.parent.threshold
            elif node.is_right:
                ft = node.parent.feature
                if L[ft][0] == None:
                    L[ft][0] = node.parent.threshold
                elif node.parent.threshold > L[ft][0]:
                    L[ft][0] = node.parent.threshold
            else:
                break
            node = node.parent
        support = float(input_node.value[input_node.class_choice])
        L.append(support/self.head_node.value[input_node.class_choice])
        L.append(support/sum(input_node.value))
        L.append(input_node.depth)
        return L
    def generate_rules(self):
        self.unclean_rulebook = {c:[] for c in range(self.DT.n_classes_)}
        while self.leaves.length > 0:
            leaf = self.leaves.pop()
            if leaf.sibling.is_leaf:
                if leaf.class_choice != leaf.sibling.class_choice:
                    L = self.rule_of_node(leaf)
                    self.unclean_rulebook[leaf.class_choice].append(L)
                else:
                    self.leaves.add(leaf.parent.Node)
                    leaf.sibling.Node.remove_self()
                    leaf.parent.is_leaf = True
            else:
                if (len(leaf.sibling.leaf_class_choices) == 1) and \
                (leaf.class_choice in leaf.sibling.leaf_class_choices):
                    self.leaves.add(leaf.Node) # his sibling hasn't bubbled up
                else:
                    L = self.rule_of_node(leaf)
                    self.unclean_rulebook[leaf.class_choice].append(L)
    def unclean2cleanrule(self, rule, feat_names = None):
        if np.all(feat_names == None):
            feat_names = [str(i) for i in range(self.DT.max_features_)]
        l = []
        for i in range(self.DT.max_features_):
            s = ''
            grtr,less,updated = rule[i][0],rule[i][1],False
            if grtr != None:
                if type(grtr) is float:
                    grtr = str(np.round(grtr,3))
                else:
                    grtr = str(grtr)
                s += grtr + ' < '
                updated = True
            s += feat_names[i]
            if less != None:
                if type(less) is float:
                    less = str(np.round(less,3))
                else:
                    less = str(less)
                s += '<= ' + less
                updated = True
            if updated:
                l.append(s)
        return ', '.join(l)
    def plot(self,feat_names = None,minimum = 0,maximum = 1,class_colors = {},
             c2f=0.05,f2b=0.01,b2b=0.15,b2c=0.05,b_size=0.05,b2s_ratio=0.9,
             b2u=0.005,c_size=18,f_size=10,u_size=10,s_size=12,dr2cr=0.075,
             title="Decision Tree Rules",Centroid=None,save='Fig7.pdf'):
        if np.all(feat_names == None):
            feat_names = [str(i) for i in range(self.DT.max_features_)]
        if type(minimum) is not dict:
            minimum = {c:minimum for c in self.unclean_rulebook.keys()}
        if type(maximum) is not dict:
            maximum = {c:maximum for c in self.unclean_rulebook.keys()}
        fig = plt.figure(figsize=(11.5,14.5))
        ax = fig.add_subplot(111)
        r = fig.canvas.get_renderer()
        y = 1.0
        inst = 0
        for c,rules in self.unclean_rulebook.items():
            try:
                try:
                    c2 = self.DT.classes_[c]
                except AttributeError:
                    c2 = c
                color = class_colors[c2]
            except KeyError:
                color = 'b'
            ax.text(0,y,c2,fontsize=c_size,color=color)
            X = np.linspace(0,1.0,self.DT.max_features_+1)
            sep = (X[1]-X[0])*b2s_ratio
            if inst == 0:
                y -= c2f
                for i in range(len(X)-1):
                    x = (X[i] + X[i] + sep)/2.0
                    ax.text(x, y, feat_names[i],ha='center', fontsize = f_size)
                l = 'Support, Purity, Depth'
                t, coords = txt(X[-1],y+c2f,l,f_size,'k',ax,r)
                y -= (f2b + b_size)
            else:
                y -= (c2f + b_size)
            for rule in rules:
                for i in range(self.DT.max_features_):
                    if (rule[i][0] != None) or (rule[i][1] != None): 
                        ax.add_patch(Rect((X[i],y),sep,b_size,
                                          fill=False,ec=[0.8,0.8,0.8],
                                          fc = 'none', capstyle='round'))
                        if rule[i][0] == None:
                            xs = X[i]
                        else:
                            xs = min_max_norm(rule[i][0],minimum[c],maximum[c],
                                              X[i],X[i]+sep)
                        if rule[i][1] == None:
                            xw = (X[i] + sep) - xs
                        else:
                            xw = min_max_norm(rule[i][1],minimum[c],maximum[c],
                                              X[i],X[i]+sep) - xs
                        ax.add_patch(Rect((xs,y),xw,b_size,fc=color,ec='none'))
                        if rule[i][0] == None:
                            left = str(minimum[c])
                        else:
                            left = str(np.round(rule[i][0],2))
                        if rule[i][1] == None:
                            right = str(maximum[c])
                        else:
                            right = str(np.round(rule[i][1],2))
                        ax.text(xs+xw/2,y-b2u,left+' -- '+right,
                                fontsize=u_size,ha='center',va='top')
                l = [str(np.round(i,2)) for i in rule[-3:]]
                ax.text(X[-1],y,', '.join(l),fontsize=s_size)
                y -= b2b
            if Centroid != None:
                y -= dr2cr
                for i in range(self.DT.max_features_):
                    ax.add_patch(Rect((X[i],y),sep,b_size,fill=False,
                                  ec=[0.8,0.8,0.8],fc='none',capstyle='round'))
                    cent = Centroid.centers_[c2][i]
                    if cent < minimum[c]:
                        ax.add_patch(Indicator((X[i],y),sep/40,b_size,
                                               fc=color))
                        centa = X[i]
                    elif cent > maximum[c]:
                        ax.add_patch(Indicator((X[i]+sep,y),sep/40,b_size,
                                               fc=color))
                        centa = X[i]+sep
                    else:
                        centa = min_max_norm(cent,minimum[c],maximum[c],X[i],
                                            X[i]+sep) - sep/80
                        ax.add_patch(Rect((centa,y),sep/40,b_size,fc=color,
                                          ec='none',fill=True))
                    ax.text(centa,y-b2u,np.round(cent,2),fontsize=u_size,
                            ha='center',va='top')
                    try:
                        if type(Centroid.radii_[c2]) is np.ndarray:
                            r = Centroid.raddi_[c2][i]
                        else:
                            r = Centroid.radii_[c2]
                        left = min_max_norm(cent-r,minimum[c],maximum[c],
                                            X[i],X[i]+sep)
                        right = min_max_norm(cent+r,minimum[c],maximum[c],
                                             X[i],X[i]+sep)
                        if left <= (X[i] + sep):
                            if left >= X[i]:
                                ax.add_patch(Parentheses((left,y),sep/40,
                                                         b_size,fc=color))
                            else:
                                left = X[i]
                            ax.plot([centa,left],[y+b_size/2,y+b_size/2],
                                    color=color)
                        if right >= X[i]:
                            if right <= (X[i]+sep):
                                ax.add_patch(Parentheses((right,y),sep/40,
                                                b_size,False,fc=color))
                            else:
                                right = X[i]+sep
                            ax.plot([centa,right],[y+b_size/2,y+b_size/2],
                                    color=color)
                    except KeyError:
                        pass
                ax.text(X[-1],y,'Centroid Method',fontsize=s_size)
                y -= b2b
            if inst != (len(self.unclean_rulebook)-1):
                y -= b2c
                inst += 1
        ax.set_ylim(bottom=y)
        ax.set_xlim(left=-.02,right=coords[1])
        ax.axis("off")
        ax.set_title(title,fontsize=c_size)
        plt.show()
        if save:
            fig.savefig(save,bbox_inches='tight',pad_inches=0)