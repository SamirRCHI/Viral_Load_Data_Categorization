# -*- coding: utf-8 -*-
"""
Created on Tue Nov 01 14:49:01 2016

Company: University of Rochester Medical Center
Team: Rochester Center for Health Informatics
Supervisor: Dr. Martin Zand
Author: Samir Farooq

Documentation Available on GitHub
"""

import datetime
import csv
from copy import deepcopy as dc
import math
import matplotlib
import matplotlib.pylab as py
import matplotlib.pyplot as plt
import matplotlib.dates as dts
import matplotlib.patches as patches
from matplotlib import gridspec
import numpy as np
import operator
from tempfile import NamedTemporaryFile
#from FPtreeGrowth import *
import networkx as nx
import shutil
#from PrefixSpan import *
from scipy.cluster.hierarchy import dendrogram,linkage
import random
from time import time as timer

class Qnode:
    def __init__(self, data):
        self.child = None
        self.data = data
     
class Queue:
    def __init__(self):
        self.head = None
        self.tail = None
        self.count = 0
    def __len__(self):
        return self.count
    def is_empty(self):
        if self.count == 0:
            return True
        else:
            return False
    def add(self, item):
        I = Qnode(item)
        if self.head == None:
            self.head = I
            self.tail = I
        else:
            self.tail.child = I
            self.tail = I
        self.count += 1
    def pop(self):
        if self.count == 0:
            raise ValueError("Nothing left to pop!")
        D = self.head.data
        self.head = self.head.child
        self.count -= 1
        return D
    

def conv_date(date_str):
    year = int(date_str[0:4])
    month = int(date_str[5:7])
    day = int(date_str[8:10])
    hour = int(date_str[11:13])
    minute = int(date_str[14:16])
    sec = int(date_str[17:19])
    return datetime.datetime(year,month,day,hour,minute,sec)
    
def date_to_str(num,with_time = False):
    D = dts.num2date(num)
    if with_time:
        return str(D.month)+'/'+str(D.day)+'/'+str(D.year)+' '+str(D.hour)+':'+str(D.minute)
    else:
        return str(D.month)+'/'+str(D.day)+'/'+str(D.year)
        
class Node:
    def __init__(self,date = '9999-12-31 23:59:59', primary_field = 'NPI'):
        self.parent = None
        self.child = None
        self.primary_field = primary_field
        self.tag = None
        self.data = {}
        self.sibling = {}
        self.up_fields = []
        self.sibling_checked = False
        self.date = dts.date2num(conv_date(date))
        self.is_class = False
        self.Class = None
        self.color = 'white'
    def __eq__(self,other_node):
        if isinstance(other_node,self.__class__):
            self_field = self.get_field(self.primary_field)
            other_field = other_node.get_field(other_node.primary_field)
            if self_field is not dict:
                self_field = {self_field:1}
            if other_field is not dict:
                other_field = {other_field:1}
            return self_field == other_field
        else:
            return False
    def __ne__(self,other_node):
        return not self.__eq__(other_node)
    def __len__(self):
        d = self.data[self.primary_field]
        if type(d) is dict:
            return len(d)
        else:
            return 1
    def is_multinode(self):
        d = self.data[self.primary_field]
        if type(d) is dict:
            if len(d) > 1:
                return True
        return False
    def set_color(self,color):
        self.color = color
    def get_color(self):
        return self.color
    def get_date(self):
        return self.date
    def add_field(self,fieldname,data):
        self.data[fieldname] = data
    def get_field(self,fieldname = None,string = False):
        if fieldname == None:
            fieldname = self.primary_field
        if string:
            try:
                f = self.data[fieldname]
                if type(f) is dict:
                    L = []
                    for key in f:
                        for number_of_times in range(f[key]): # To adjust for self connections
                            if key == '':
                                L.append('UNKNOWN')
                            else:
                                L.append(key)
                    f = sorted(L) # For the purpose of consistency
                    if len(f) == 1:
                        return f[0]
                    else:
                        c = ','
                        s = '('+c.join(f)+')' # Create Multinode String
                else:
                    if f == '':
                        s = 'UNKNOWN'
                    else:
                        s = str(f)
            except KeyError:
                s = 'NOT AVAILABLE'
            return s        
        else:
            try:
                f = self.data[fieldname]
            except KeyError:
                f = {}
            return f
    def get_data(self):
        return self.data
    def has_parent(self):
        if self.parent == None:
            return False
        else:
            return True
    def has_child(self):
        if self.child == None:
            return False
        else:
            return True
    def has_siblings(self):
        if self.siblings > 0:
            return True
        else:
            return False
    def get_all_siblings(self):
        siblings_list = []
        for n in range(self.siblings):
            node = Node(self.date)
            node.add_parent(self.parent)
            node.add_child(self.child)
            for key in self.data:
                node.add_field(self.data[key][n])
            siblings_list.append(node)
        return siblings_list
    def add_parent(self,node):
        if self.parent == None:
            self.parent = node
        else:
            node.child = self
            node.parent = self.parent
            self.parent.child = node
            self.parent = node
        self.parent = node
    def add_child(self,node):
        if self.child == None:
            self.child = node
        else:
            node.parent = self
            node.child = self.child
            self.child.parent = node
            self.child = node
    def check_sibling(self,field_properties):
        PFv = self.data[self.primary_field]
        self.sibling[PFv] = {}
        self.data[self.primary_field] = {PFv:1}
        for field in self.data:
            if field == self.primary_field:
                continue
            try:
                field_property = field_properties[field]
            except KeyError:
                field_property = 'n'
            if field_property == 'up':
                self.sibling[PFv][field] = set([self.data[field]])
                self.up_fields.append(field)
            self.data[field] = {self.data[field]:1}
        self.sibling_checked = True
    def create_blank_sibling(self,primary_data):
        self.sibling[primary_data] = {}
        for field in self.up_fields:
            self.sibling[primary_data][field] = set()
    def sibling_decision(self,node,field,field_properties):
        try:
            field_property = field_properties[field]
        except KeyError:
            field_property = 'n'
        if field_property == 'u':
            self.data[field][node.data[field]] = 1
        elif field_property == 'up':
            if node.data[field] not in self.sibling[node.data[self.primary_field]][field]:
                try:
                    self.data[field][node.data[field]] += 1
                except KeyError:
                    self.data[field][node.data[field]] = 1
                self.sibling[node.data[self.primary_field]][field].add(node.data[field])
        elif field_property == 'np':
            if len(self.data[self.primary_field]) <= 1:
                try:
                    self.data[field][node.data[field]] += 1
                except KeyError:
                    self.data[field][node.data[field]] = 1
        else:
            try:
                self.data[field][node.data[field]] += 1
            except KeyError:
                self.data[field][node.data[field]] = 1
    def add_sibling(self,node,field_properties = {}):
        if not self.sibling_checked:
            self.check_sibling(field_properties)
        if node.data[self.primary_field] not in self.sibling:
            self.create_blank_sibling(node.data[self.primary_field])
        self.sibling_decision(node,self.primary_field,field_properties)
        for field in self.data:
            if field == self.primary_field:
                continue
            self.sibling_decision(node,field,field_properties)
    def get_child(self):
        return self.child
    def get_parent(self):
        return self.parent
    def display(self,field = None):
        if field == None:
            field = self.primary_field
        print self.get_field(field)

class PatientPath:
    def __init__(self,node,primary_field = 'NPI'):
        self.path_head = node
        self.primary_field = primary_field
        self.has_multinode = False
        self.patient_id = None
        self.stage = 0
        self.cur_node = None
        self.length = 1
        self.Class = None
        self.parent_Network = None
    def __len__(self):
        return self.length
    def __iter__(self):
        return self
    def next(self):
        if self.stage == 0:
            self.cur_node = self.path_head
            self.stage = 1
            return self.cur_node
        elif self.cur_node.get_child() == None:
            self.stage = 0
            self.cur_node = self.path_head
            raise StopIteration
        elif self.stage == 1:
            self.cur_node = self.cur_node.get_child()
            return self.cur_node
    def as_list(self,field=None,string=True,with_dates=False,clean=False):
        if field == None:
            field = self.primary_field
        path = []
        dates = []
        for node in self:
            if node.is_class:
                item = node.Class
            else:
                item = node.get_field(field,string)
                if clean:
                    if not string and (item == {}):
                        continue
                    elif string and (item == 'NOT AVAILABLE'):
                        continue
                if item == '':
                    item = 'UNKNOWN'
            path.append(item)
            if with_dates:
                #D = dts.date2num(node.date)
                dates.append(node.date)
        if with_dates:
            return path,dates
        return path
    def get_dates(self):
        D = []
        for node in self:
            D.append(node.date)
        return D
    def update_len(self):
        self.length += 1
        if self.length > self.parent_Network.max_len:
            self.parent_Network.max_len = self.length
    def set_patient_id(self,p_id):
        self.patient_id = p_id
    def get_patient_id(self):
        return self.patient_id
    def get_head(self):
        return self.path_head
    def display(self,field = None):
        if field == None:
            field = self.primary_field
        node = self.path_head
        while node.get_child() != None:
            print str(node.get_field(field))+'->'
            node = node.get_child()
        print str(node.get_field(field))
    def set_parent_Network(self,Network):
        self.parent_Network = Network
    def update_class(self,new_class):
        self.parent_Network.classes[self.Class] -= 1
        self.parent_Network.pat_by_class[self.Class].pop(self.patient_id)
        if self.parent_Network.classes[self.Class] == 0:
            self.parent_Network.classes.pop(self.Class)
            self.parent_Network.pat_by_class.pop(self.Class)
        self.Class = new_class
        try:
            self.parent_Network.classes[new_class] += 1
            self.parent_Network.pat_by_class[new_class][self.patient_id] = self
        except KeyError:
            self.parent_Network.classes[new_class] = 1
            self.parent_Network.pat_by_class[new_class] = {self.patient_id:self}

class Networks:
    def __init__(self,filepath=None,primary_field='NPI',date='visit_date',p_id='id',
                 tag='NPIpath',lastnames=True,conversion_rules = {},
                 field_properties = {'Classification':'up','proc':'np',
                                     'provider_name':'up','specific_proc':'u'}):
        self.Nets = {}
        self.Overview = {}
        self.last_update = '01/17/2018 12:24'
        self.tag = tag
        self.T_0 = dts.date2num(datetime.datetime(9999,12,31,23,59,59)) # first date
        self.T_final = 0 # Last date in database
        self.date = date
        self.p_id = p_id
        if primary_field not in field_properties:
            field_properties[primary_field] = 'u'
        self.field_properties = field_properties
        self.idtoname = {}
        self.dupnames = set()
        self.primary_field = primary_field
        self.classes = {}
        self.pat_by_class = {None:{}}
        self.class_colors = {}
        self.max_len = 0
        self.cur_key = -1
        self.max = {}
        self.min = {}
        if filepath != None:
            self.buildNetworks(filepath, lastnames)
            if conversion_rules != {}:
                self.apply_conversion_rules(conversion_rules)
    def __iter__(self):
        return self
    def next(self):
        self.cur_key += 1
        if self.cur_key == len(self.Nets.keys()):
            self.cur_key = -1
            raise StopIteration
        else:
            return self.Nets[self.Nets.keys()[self.cur_key]]['Path']
    def __getitem__(self,key):
        return self.Nets[key]['Path']
    def get_max_len(self):
        return self.max_len
    def get_max(self,field = None):
        if field == None:
            field = self.primary_field
        try:
            return self.max[field]
        except KeyError:
            return None
    def get_min(self,field = None):
        if field == None:
            field = self.primary_field
        try:
            return self.min[field]
        except KeyError:
            return None    
    def apply_conversion_rules(self,conversion_rules):
        for f in conversion_rules:
            if conversion_rules[f] == 'int':
                M = 0
                m = float('inf')
                for path in self:
                    for node in path:
                        d = node.data[f]
                        if type(d) is dict:
                            D = []
                            for i in d:
                                D.append(int(i))
                            d = np.mean(D)
                        d = int(d)
                        node.data[f] = d
                        if d > M:
                            M = d
                        if d < m:
                            m = d
                self.max[f] = M
                self.min[f] = m
            elif conversion_rules[f] == 'float':
                M = 0
                m = float('inf')
                for path in self:
                    for node in path:
                        d = node.data[f]
                        if type(d) is dict:
                            D = []
                            for i in d:
                                D.append(float(i))
                            d = np.mean(D)
                        d = float(d)
                        node.data[f] = d
                        if d > M:
                            M = d
                        if d < m:
                            m = d
                self.max[f] = M
                self.min[f] = m
            else:
                print "Sorry, only 'int' and 'float' are accepted as conversions at this time."
    def count_Overview(self):
        for path in self:
            self.Overview[path.patient_id] = {}
            for node in path:
                for field,items in node.data.items():
                    if field not in self.Overview[path.patient_id]:
                        self.Overview[path.patient_id][field] = {}
                    if field not in self.Overview:
                        self.Overview[field] = {}
                    if type(items) is dict:
                        for item,count in items.items():
                            try:
                                self.Overview[path.patient_id][field][item] += count
                            except KeyError:
                                self.Overview[path.patient_id][field][item] = count
                            try:
                                self.Overview[field][item] += count
                            except KeyError:
                                self.Overview[field][item] = count
                    else:
                        try:
                            self.Overview[path.patient_id][field][items] += 1
                        except KeyError:
                            self.Overview[path.patient_id][field][items] = 1
                        try:
                            self.Overview[field][items] += 1
                        except KeyError:
                            self.Overview[field][items] = 1
    def count_occurrence(self,item,field = None,count_style = 'both'):
        # Count style can take 3 options:
        # 'singlenode only', 'multinode only', or 'both'.
        if field == None:
            field = self.primary_field
        multicount = 0
        singlcount = 0
        for path in self:
            for node in path:
                if node.is_multinode():
                    if item in node.get_field(field):
                        multicount += 1
                else:
                    if node.get_field(field) == item:
                        singlcount += 1
        if count_style == 'singlenode only':
            return singlcount
        elif count_style == 'multinode only':
            return multicount
        elif count_style == 'both':
            return (singlcount,multicount)
        else:
            raise UserWarning
            print "Invalid count_style inputted- returning both"
            return singlcount,multicount                       
    def set_tag(self,tag):
        self.tag = tag
    def makeNode(self,h,r, seen_a_name):
        node = Node(r[h[self.date]], self.primary_field)
        node.tag = self.tag
        if seen_a_name != False:
            r[seen_a_name] = self.idtoname[r[h[self.primary_field]]]
        for item in h:
            if (item != self.date) and (item != self.p_id):
                if r[h[item]] == '':
                    appending_item = 'UNKNOWN'
                else:
                    appending_item = r[h[item]]
                node.add_field(item,appending_item)           
        return node
    def buildNetworks(self, filepath, lastnames = True):
        if type(filepath) is str:
            f = open(filepath,'r')
            fcsv = csv.reader(f)
        else:
            fcsv = filepath
        rownum = 0
        name_count = {}
        idtoname = {}
        if lastnames:
            if type(lastnames) is bool:
                lastnames = 'name'
            for row in fcsv:
                if rownum == 0:
                    seen_a_name = False
                    for i in range(len(row)):
                        if row[i] == self.primary_field:
                            id_idx = i
                        if lastnames in row[i]:
                            seen_a_name = i
                    rownum = 1
                elif seen_a_name != False:
                    if row[id_idx] not in idtoname:
                        names = row[seen_a_name].split(',')
                        lastname = names[0]
                        idtoname[row[id_idx]] = names
                        try:
                            name_count[lastname] += 1
                        except KeyError:
                            name_count[lastname] = 1
                else:
                    break
            if seen_a_name != False:
                for ID in idtoname:
                    name_list = idtoname[ID]
                    if name_count[name_list[0]] == 1:
                        idtoname[ID] = name_list[0]
                    else:
                        f_lname = name_list[1][1:]+' '+name_list[0]
                        idtoname[ID] = f_lname
                self.idtoname = idtoname      
            f.seek(0)
            fcsv = csv.reader(f)           
        rownum = 0
        header = {}            
        for row in fcsv:
            if rownum == 0:
                # Build Header information with first row
                column = 0
                seen_a_name = False
                for head in row:
                    header[head] = column
                    if (lastnames in head): #lastnames was 'name'
                        seen_a_name = column
                    column += 1 # We store the header information this way
                if lastnames == False:
                    seen_a_name = False
                prev_p_id = None
                prev_node = None
            else:
                # Building the network
                cur_p_id = row[header[self.p_id]]
                if (cur_p_id != '') and (cur_p_id != None):
                    cur_node = self.makeNode(header,row, seen_a_name)
                    if cur_node.get_date() < self.T_0:
                        self.T_0 = cur_node.get_date() # Update T_0
                    if cur_node.get_date() > self.T_final:
                        self.T_final = cur_node.get_date()
                    if prev_p_id != cur_p_id:
                        # Assuming that the current patient hasn't been added yet
                        cur_Path = PatientPath(cur_node, self.primary_field)
                        cur_Path.set_patient_id(cur_p_id)
                        self.Nets[cur_p_id] = {'Path':cur_Path}
                        self.pat_by_class[None][cur_p_id]=cur_Path
                        cur_Path.set_parent_Network(self)
                        prev_node = self.Nets[cur_p_id]['Path'].get_head()
                        prev_p_id = cur_p_id
                    else:
                        # This means we are still working with same patient
                        if prev_node.get_date() != cur_node.get_date():   
                            prev_node.add_child(cur_node)
                            prev_node = cur_node
                            cur_Path.update_len()
                        else:
                            prev_node.add_sibling(cur_node,self.field_properties)
                            boolean = prev_node.is_multinode()
                            if boolean:
                                cur_Path.has_multinode = True
            rownum += 1
        self.classes[None] = len(self.Nets)
        f.close()
        self.Plot = Plot_Network(self)
    def add_class_nodes(self,filepath):
        if type(filepath) is str:
            f = open(filepath,'r')
            fcsv = csv.reader(f)
        else:
            fcsv = filepath
        inst = 0
        for r in fcsv:
            if inst == 0:
                inst += 1
                prev_id = None
                continue
            if prev_id != r[0]:
                D = dts.date2num(conv_date(r[2]))
                try:
                    curPath = self[r[0]]
                except KeyError:
                    continue
                curNode = curPath.path_head
                while (D > curNode.date) and (curNode.child != None):
                    curNode = curNode.child
                classNode = Node(r[2])
                classNode.is_class = True
                classNode.Class = r[1]
                if D > curNode.date:
                    curNode.add_child(classNode)
                else:
                    curNode.add_parent(classNode)
                prev_id = r[0]
            else:
                D = dts.date2num(conv_date(r[2]))
                while (D > curNode.date) and (curNode.child != None):
                    curNode = curNode.child
                classNode = Node(r[2])
                classNode.is_class = True
                classNode.Class = r[1]
                if D > curNode.date:
                    curNode.add_child(classNode)
                else:
                    curNode.add_parent(classNode)
                prev_id = r[0]
    def set_class_colors(self,dict_or_input = 'input'):
        if type(dict_or_input) is dict:
            self.class_colors = dict_or_input
        else:
            for c in self.classes:
                self.class_colors[c] = raw_input('Color of '+str(c)+': ')
    def class_lists(self,field = None,Tau = float('inf'),as_dict = True,string=True):
        if field == None:
            field = self.primary_field
        if as_dict:
            cL = {}
            for c in self.classes:
                cL[c] = []
        else:
            cL = []
        for path in self:
            c = path.Class
            prev_date = -1
            l = []
            for node in path:
                cur_date = node.date
                if (prev_date == -1) or ((cur_date - prev_date)>Tau):
                    if l != []:
                        if as_dict:
                            cL[c].append(l)
                        else:
                            l.append(c)
                            cL.append(l)
                    l = [node.get_field(field,string)]
                else:
                    l.append(node.get_field(field,string))
                prev_date = cur_date
            if as_dict:
                cL[c].append(l)
            else:
                l.append(c)
                cL.append(l)
        return cL
    def add_classes(self,determine_class,Net='self',optional_threshold = 'default',
                    class_field=None,string=False,print_result = True):
        if Net == 'self':
            Net = self
            on_self = True
        else:
            on_self = False
        if class_field == None:
            class_field = Net.primary_field
        count_nonexisting = 0
        for path in Net:
            L = path.as_list(class_field,string)
            c = determine_class(L,optional_threshold)
            if not on_self:
                path.update_class(c)
            try:
                self.Nets[path.patient_id]['Path'].update_class(c)
            except KeyError:
                count_nonexisting += 1
        if print_result:
            print "There were "+str(self.classes[None])+" unaccounted for patients"
            print "There were "+str(count_nonexisting)+" non-existing patients"
    def write_classes(self,filename):
        id_class = [['id','class']]
        for path in self:
            id_class.append([path.patient_id,path.Class])
        b = open(filename,'wb')
        a = csv.writer(b)
        a.writerows(id_class)
        b.close()
    def add_classes_from_dict(self,class_dict):
        count_un = len(self.Nets.keys())
        count = 0
        for c in class_dict:
            for p in class_dict[c]:
                try:
                    self.Nets[p]['Path'].update_class(c)
                    count_un -= 1
                except KeyError:
                    count += 1
        print "There were "+str(count_un)+" unaccounted for patients"
        print "There were "+str(count)+" non-existing patients"
    def export_as_csv(self,field = None, round_decimals=4):
        if field == None:
            field = self.primary_field
        if self.tag == None:
            filename = 'HIV_patientpath_based_on_'+field+'.csv'
        else:
            filename = self.tag+'_based_on_'+field+'.csv'
        export_data = []
        for key in self.Nets:
            patient_row = [key]
            instance = 1
            for node in self.Nets[key]['Path']:
                if instance == 1:
                    instance = 2
                    prev_date = node.get_date()
                    delta = prev_date - self.T_0
                else:
                    delta = node.get_date() - prev_date
                    prev_date = node.get_date()
                node_info = node.get_field(field)
                if (type(node_info) is float) and (round_decimals != False):
                    node_info = [delta, np.round(node_info,round_decimals)]
                elif type(node_info) is not dict:
                    node_info = [delta, node_info]
                else:
                    node_info = list(node_info.keys())
                    temp = []
                    for e in range(len(node_info)):
                        if e == 0:
                            temp.append(delta) # Append the edge weight between nodes
                            temp.append(node_info[e])
                        else:
                            temp.append(0) # Edge weight of 0 within the same node
                            temp.append(node_info[e])
                    node_info = temp
                patient_row = patient_row + node_info
            export_data.append(patient_row)
        b = open(filename,'wb')
        a = csv.writer(b)
        a.writerows(export_data)
        b.close()
    def generate_class_artists(self,style='rectangle',which='all',fig=None):
        if which == 'all': L = self.classes.keys()
        elif (type(which) is list) or (type(which) is np.ndarray): L = which
        elif type(which is str) and (which in self.classes): L = [which]
        else: raise AssertionError("Unrecognized classes inputted")
        Artists,used_labels = [],[]
        for c in L:
            try: clr = self.class_colors[c]
            except KeyError: continue
            used_labels.append(c)
            if style == 'rectangle': 
                Artists.append(patches.Rectangle((0,0),1,1,
                                facecolor=clr,linewidth=0))
            elif style == 'scatter': 
                ax = fig.add_subplot(111)
                Artists.append(plt.scatter([0],[0],c=clr))
                fig.delaxes(ax)
            elif style == 'line': Artists.append(plt.Line2D(0,0,color=clr))
            else:
                ax = fig.add_subplot(111)
                Artists.append(plt.scatter([0],[0],c=clr,marker=style))
                fig.delaxes(ax)
        return Artists,used_labels
    def display_paths(self,field = None, display_num = 'inf'):
        if field == None:
            field = self.primary_field
        if display_num == 'inf':
            display_num = len(self.Nets.keys())
        key_num = 1
        for key in self.Nets:
            print "--- Path of: "+str(key)+" ---"
            self.Nets[key]['Path'].display(field)
            if key_num == display_num:
                break
            key_num += 1
            
def merge_Networks(parent_Network, other_Networks = [], tag_colors = []):
    if len(tag_colors) > 0:
        parent_Network.tag_colors = {parent_Network.tag: tag_colors[0]}
        for i in range(1,len(tag_colors)):
            parent_Network.tag_colors[other_Networks[i-1].tag]=tag_colors[i]
    parent_Network.tag = {parent_Network.tag}
    for N in other_Networks:
        parent_Network.tag.add(N.tag)
        if N.T_0 < parent_Network.T_0:
            parent_Network.T_0 = N.T_0
        if N.T_final > parent_Network.T_final:
            parent_Network.T_final = N.T_final
    for path in parent_Network:
        dates = path.get_dates()
        nodes = [node for node in path]
        update = 0
        for N in other_Networks:
            try:
                path2 = N.Nets[path.patient_id]['Path']
                update += 1
            except KeyError:
                continue
            dates += path2.get_dates()
            nodes += [node for node in path2]
            if path2.has_multinode:
                path.has_multinode = True
        if update == 0:
            continue
        dates = np.array(dates)
        nodes = np.array(nodes)
        I = np.argsort(dates)
        dates,nodes = dates[I],nodes[I]
        path.head = nodes[0]
        prv_node,path.length = path.head,1
        prv_node.tag = {prv_node.tag}
        for i in range(len(nodes))[1:]:
            cur_node = nodes[i]
            cur_node.tag = {cur_node.tag}
            if prv_node.get_date() == cur_node.get_date():
                prv_node.tag.add(cur_node.tag.pop())
                for k,t in cur_node.data.items():
                    prv_node.data[k] = t
            else:
                cur_node.parent = prv_node
                prv_node.child = cur_node
                prv_node = cur_node
                path.update_len()
                

def class_transfer(Info_Network,Updating_Network):
    for path in Info_Network:
        pid = path.patient_id
        try:
            Updating_Network.Nets[pid]['Path'].update_class(path.Class)
        except KeyError:
            pass
    for c in Updating_Network.classes.keys():
        if Updating_Network.classes[c] == 0:
            Updating_Network.classes.pop(c)
            Updating_Network.pat_by_class.pop(c)
    for c,color in Info_Network.class_colors.items():
        Updating_Network.class_colors[c] = color
            
def class_combine(Network,parent_class,child_classes):
    if type(child_classes) is str:
        for ID,path in Network.pat_by_class[child_classes].items():
            path.update_class(parent_class)
        return Network
    else:
        for child_class in child_classes:
            Network = class_combine(Network,parent_class,child_class)
        return Network

def class_writer(Network,filename):
    id_class = [['id','class']]
    for path in Network:
        id_class.append([path.patient_id,path.Class])
    b = open(filename,'wb')
    a = csv.writer(b)
    a.writerows(id_class)
    b.close()

def add_class_from_file(Network,filename,ID,Class):
    b = open(filename,'rU')
    a = csv.reader(b)
    i = 0
    for row in a:
        if i == 0:
            header = {row[h]:h for h in range(len(row))}
            i += 1
            continue
        cur_id = row[header[ID]]
        if cur_id not in Network.Nets:
            continue
        path = Network.Nets[row[header[ID]]]['Path']
        path.update_class(row[header[Class]])
    return(Network)

def class_subsetting(New_Net,Networks,class_dictionary,IMPURE='IMPURE',IMPURE_color='k'):
    trans,color_dict = {},{IMPURE:IMPURE_color}
    for subset_class_name,old_classes in class_dictionary.items():
        colors = {}
        for i in range(len(old_classes)):
            trans[old_classes[i]] = subset_class_name
            color = Networks[i].class_colors[old_classes[i]]
            try:
                colors[color] += 1
            except KeyError:
                colors[color] = 1
        if len(colors) == 1:
            color_dict[subset_class_name] = colors.keys()[0]
    for path in New_Net:
        P = path.patient_id
        try:
            classes = {trans[Net.Nets[P]['Path'].Class] for Net in Networks}
            if len(classes) == 1:
                path.update_class(classes.pop())
            else:
                path.update_class(IMPURE)
        except KeyError:
            path.update_class(IMPURE)
    return New_Net
            
def rename_classes(Net,old_to_new_dict):
    for path in Net:
        try: path.update_class(old_to_new_dict[path.Class])
        except KeyError: continue
    for key in Net.classes.keys():
        if Net.classes[key] == 0:
            Net.classes.pop(key)
    for key in old_to_new_dict:
        color = Net.class_colors.pop(key)
        Net.class_colors[old_to_new_dict[key]] = color
    return Net

#NPIpath = Networks('NPIpath.csv')


#########################################################
#########################################################
########### Plotting Network Class ######################
#########################################################
#########################################################

def quadratic_eq(a,b,c):
    a,b,c = float(a),float(b),float(c)
    x1 = (-b + (b**2 - 4*a*c)**(0.5))/(2*a)
    x2 = (-b - (b**2 - 4*a*c)**(0.5))/(2*a)
    return x1,x2
    
def circle_intersection(r,a=0,b=0,c=0,d=0):
    r,a,b,c,d = float(r),float(a),float(b),float(c),float(d) 
    a4,b4,c4,d4 = a**4,b**4,c**4,d**4
    a3,b3,c3,d3 = a**3,b**3,c**3,d**3
    r2,a2,b2,c2,d2 = r**2,a**2,b**2,c**2,d**2
    
    Qc = a4-4*a2*r2-4*a3*c-4*a*b2*c+8*a*c*r2+6*a2*c2+2*b2*c2-4*c2*r2+\
         2*a2*d2-4*a*c3-4*a*d2*c+2*a2*b2+b4-2*b2*d2+c4+2*c2*d2+d4
    Qb = 8*a*b*c-4*b*c2-4*a2*d+8*a*d*c-4*a2*b+4*b2*d-4*b3-4*c2*d-4*d3+4*b*d2
    Qa = 4*a2-8*a*c+4*c2+4*d2-8*b*d+4*b2
    print Qa,Qb,Qc
    
    y1,y2 = quadratic_eq(Qa,Qb,Qc)
    
    x1 = (-a2-b2+c2+d2-2*d*y1+2*b*y1)/(-2*a+2*c)
    x2 = (-a2-b2+c2+d2-2*d*y2+2*b*y2)/(-2*a+2*c)
    return [(x1,y1),(x2,y2)]
    
def arrow_points(r,a=0,b=0,c=0,d=0,h=0):
    r,a,b,c,d,h = float(r),float(a),float(b),float(c),float(d),float(h)
    r2,a2,b2,c2,d2 = r**2,a**2,b**2,c**2,d**2
    TOL = 10.0**(-8.0)
    
    if (c - a)**2 < TOL: # Deals with the case that c-a = 0
        x1,x2 = a,c
        if d >= b:
            y1,y2 = b+r,d-(r+h)
        else:
            y1,y2 = b-r,d+r+h
        dx = x2 - x1
        dy = y2 - y1
        return x1,y1,dx,dy
    
    m = (d-b)/(c-a)
    I = b - m*a
    # Beginning Arrow Point
    Qa = 1 + m**2
    Qb = -2*a+2*I*m-2*b*m
    Qc = a2+I**2-2*b*I+b2-r2
    
    x1,x2 = quadratic_eq(Qa,Qb,Qc)
    y1,y2 = m*(x1)+I, m*(x2)+I
    
    dis1,dis2 = ((c-x1)**2+(d-y1)**2)**(0.5) , ((c-x2)**2+(d-y2)**2)**(0.5)
    if dis2 < dis1:
        x1,y1 = x2,y2
        
    # Ending Arrow Point
    Qb = -2*c+2*I*m-2*d*m
    Qc = c2+I**2-2*d*I+d2-(r+h)**2
    
    x2,x3 = quadratic_eq(Qa,Qb,Qc)
    y2,y3 = m*(x2)+I, m*(x3)+I
    
    dis1,dis2 = ((a-x2)**2+(b-y2)**2)**(0.5) , ((a-x3)**2+(b-y3)**2)**(0.5)
    if dis2 < dis1:
        x2,y2 = x3,y3
    
    dx = x2 - x1
    dy = y2 - y1
    return x1,y1,dx,dy
    
def theta_finding(coords1,coords2):
    x1,y1 = coords1
    x2,y2 = coords2 # Try to find dx and dy and do it that way.
    TOL = .001
    x1,x2,y1,y2 = float(x1),float(x2),float(y1),float(y2)
    dx,dy = np.abs(x1-x2),np.abs(y1-y2)
    if x1 == x2:
        theta = np.arctan(float('inf'))
    else:
        theta = np.arctan((y2-y1)/(x2-x1))
    if theta >= 0:
        x2a,y2a = x1-dx,y1-dy
        if ((x2-x2a)**2 < TOL) and ((y2-y2a)**2 < TOL):
            theta += np.pi
    else:
        theta += 2*np.pi
        x2a,y2a = x1-dx,y1+dy
        if ((x2-x2a)**2 < TOL) and ((y2-y2a)**2 < TOL):
            theta -= np.pi
    return theta

def z2pi(theta):
    return theta - 2*np.pi*np.floor(theta/(2*np.pi))
    
def theta_intersect(center,radius,intersection):
    TOL = .00001
    ex,ey = center
    x0,y0 = intersection
    ct1 = np.arccos((x0-ex)/radius)
    ct2 = np.pi + (np.pi - ct1)
    st1 = np.arcsin((y0-ey)/radius)
    if st1 < 0:
        st1 += 2*np.pi
        st2 = 3*np.pi/2.0 - (st1 - 3*np.pi/2.0)
    else:
        st2 = np.pi/2.0 + (np.pi/2.0 - st1)
    if ((ct1 - st1)**2 < TOL) or ((ct1 - st2)**2 < TOL):
        return ct1
    elif ((ct2 - st1)**2 < TOL) or ((ct2 - st2)**2 < TOL):
        return ct2
    else:
        print ct1,ct2,st1,st2
        raise ValueError("No theta found, check inputs!")
        
def arrow_head_coords(center,slope,length):
    x,y = center
    x2,s2 = x ** 2,slope ** 2
    d = length/2.0 # Accounts for that slight slope change we need
    n = (d**2/(1+slope))**.5+x
    m = -slope*(x-n)+y
    Qa = 1 + s2
    Qb = -2*x - 2*x*s2
    Qc = x2 + s2*x2 - d**2
    n1,n2 = quadratic_eq(Qa,Qb,Qc)
    m1,m2 = -slope * (x - n1) + y, -slope * (x - n2) + y
    return n1,m1,(x-n1)/9999.0,(y-m1)/9999.0

def complexity_score(V,E):
    V,E = float(V),float(E)
    if V == 1:
        return 0
    return V + (E / (V - 1))**2 - 3
    
def score_and_order(L,N):
    score = {}
    for v in L:
        score[v] = 0 
    

class Plot_Network:
    def __init__(self,Network):
        self.Network = Network
        self.fig = None
        self.edges = {}
        self.neighbors = {}
        self.complexity = {}
        self.weighted_edges = {}
        self.uniq_edges = {}
        self.max_neighbor = {}
        self.edge_type = None
        self.directed = True
        self.node_color = [.5,.675,.675]
        self.arrow_color = 'r'
        self.arrow_alpha = 0.6
        self.max_node_radius = 0.25
        self.orbital_radius = 1.0
    def clear_data(self):
        if self.edges != {}:
            self.edges = {}
            self.neighbors = {}
            self.complexity = {}
            self.weighted_edges = {}
            self.uniq_edges = {}
            self.max_neighbor = {}
    def get_patient_edges(self,field = None,directed=True,classes=None):
        self.clear_data()
        self.edge_type = 'patient'
        arrow = '->'
        if not directed:
            arrow = '<->'
            self.directed = False
        for path in self.Network:
            if classes == None:
                pass
            elif path.Class not in classes:
                continue
            m_neighbor = 0
            L = path.as_list(field = field)
            self.weighted_edges[path.patient_id] = {}
            self.edges[path.patient_id] = []
            neighbors = {}
            arrows = {}
            for i in range(len(L)-1):
                if len(L) == 0:
                    continue
                if directed:
                    try:
                        neighbors[L[i]].add(L[i+1])
                        arrows[L[i]].add(L[i+1])
                    except KeyError:
                        neighbors[L[i]] = set([L[i+1]])
                        arrows[L[i]] = set([L[i+1]])
                    try:
                        neighbors[L[i+1]].add(L[i])
                    except KeyError:
                        neighbors[L[i+1]] = set([L[i]])
                else:
                    try:
                        neighbors[L[i]].add(L[i+1])
                    except KeyError:
                        neighbors[L[i]] = set([L[i+1]])
                    try:
                        neighbors[L[i+1]].add(L[i])
                    except KeyError:
                        neighbors[L[i+1]] = set([L[i]])
                if not directed:
                    arrows = dc(neighbors)
                self.neighbors[path.patient_id] = [neighbors, arrows, 1]
                if directed:
                    edge = arrow.join([L[i],L[i+1]])
                else:
                    edge = arrow.join(sorted[L[i],L[i+1]])
                try:
                    self.weighted_edges[path.patient_id][edge] += 1
                except KeyError:
                    self.weighted_edges[path.patient_id][edge] = 1
                self.edges[path.patient_id].append((L[i],L[i+1]))
            mv = L[0]
            for v,S in neighbors.items():
                amt = len(S)
                if v in S:
                    amt -= 1
                if amt > m_neighbor:
                    m_neighbor = amt
                    mv = v
            self.max_neighbor[path.patient_id] = [m_neighbor,mv]
            V = len(neighbors)
            E = len(self.edges[path.patient_id])
            C = complexity_score(V, E)
            self.complexity[path.patient_id] = C
    def get_isomorphic_edges(self,field = None,directed = True,classes=None):
        self.clear_data()
        self.edge_type = 'isomorphic'
        arrow = '->'
        if not directed:
            arrow = '<->'
            self.directed = False
        for path in self.Network:
            if classes == None:
                pass
            elif path.Class not in classes:
                continue
            L = path.as_list(field = field)
            self.weighted_edges[path.patient_id] = {}
            self.edges[path.patient_id] = []
            #self.neighbors[path.patient_id] = {}
            trans = {'1':L[0],L[0]:'1'}
            t = 2
            for i in range(1,len(L)):
                if L[i] not in trans:
                    trans[str(t)] = L[i]
                    trans[L[i]] = str(t)
                    t+=1
                Li = trans[L[i]]
                Lb = trans[L[i-1]]
                if directed:
                    edge = arrow.join([Lb,Li])
                else:
                    edge = arrow.join(sorted([Lb,Li]))
                try:
                    self.weighted_edges[path.patient_id][edge] += 1
                except KeyError:
                    self.weighted_edges[path.patient_id][edge] = 1
                self.edges[path.patient_id].append((Lb,Li))
            if self.weighted_edges[path.patient_id] == {}:
                continue
            K = str(sorted(self.weighted_edges[path.patient_id]))
            try:
                self.neighbors[K][2] += 1
            except KeyError:
                neighbors = {}
                arrows = {}
                for e in self.edges[path.patient_id]:
                    if directed:
                        if e[0] in neighbors:
                            neighbors[e[0]].add(e[1])
                        else:
                            neighbors[e[0]] = set([e[1]])
                        if e[1] in neighbors:
                            neighbors[e[1]].add(e[0])
                        else:
                            neighbors[e[1]] = set([e[0]])
                        if e[0] in arrows:
                            arrows[e[0]].add(e[1])
                        else:
                            arrows[e[0]] = set([e[1]])
                    else:
                        try:
                            neighbors[e[0]].add(e[1])
                        except KeyError:
                            neighbors[e[0]] = set([e[1]])
                        try:
                            neighbors[e[1]].add(e[0])
                        except KeyError:
                            neighbors[e[1]] = set([e[0]])
                if not directed:
                    arrows = dc(neighbors)
                self.neighbors[K] = [neighbors, arrows, 1]
                #self.uniq_edges[K] = [i.split(arrow) 
                #for i in self.weighted_edges[path.patient_id]]
                m_neighbor = 0
                mv = '1'
                for v,S in neighbors.items():
                    amt = len(S)
                    if v in S:
                        amt -= 1
                    if amt > m_neighbor:
                        m_neighbor = amt
                        mv = v
                self.max_neighbor[K] = [m_neighbor,mv]
                V = len(neighbors)
                E = 0
                for v,S in arrows.items():
                    E += len(S)
                C = complexity_score(V, E)
                self.complexity[K] = C
    def get_complexity_avg(self):
        if self.complexity == {}:
            raise AssertionError('Complexity has not yet been calculated!')
        C = 0
        for c in self.complexity.values():
            C += c
        C /= len(self.complexity)
        self.complexity_avg = C
        return C
    def self_arrow_plot(self,v,node_radius):
        a,b = self.coords[v]
        try:
            angles1 = self.all_angles[v]
        except KeyError:
            angles1 = [3.0*np.pi/2.0]
        if len(angles1) == 1:
            angles2 = np.array([angles1[0] + 2*np.pi])
            angles1 = np.array(angles1)
        else:
            angles1 = sorted(angles1)
            angles2 = np.array(angles1[1:]+[angles1[0]+2*np.pi])
            angles1 = np.array(angles1)
        distances = angles2-angles1
        i = np.argmax(distances)
        angle = (angles1[i]+angles2[i])/2.0
        clock = np.linspace(angle,angle+2*np.pi,13)
        x0,y0 = node_radius*np.cos(clock[11])+a,node_radius*np.sin(clock[11])+b
        x1,y1 = node_radius*np.cos(clock[1])+a,node_radius*np.sin(clock[1])+b
        ex,ey = node_radius*np.cos(angle)+a,node_radius*np.sin(angle)+b
        sr = ((x0-ex)**2+(y0-ey)**2)**.5
        sr2 = ((x1-ex)**2+(y1-ey)**2)**.5
        t1 = theta_intersect((ex,ey),sr,(x0,y0))
        t2 = theta_intersect((ex,ey),sr2,(x1,y1))
        if t1 > t2:
            t1 -= 2*np.pi
        theta = np.linspace(t1,t2,500)
        x,y = sr*np.cos(theta)+ex,sr*np.sin(theta)+ey
        self.ax.plot(x,y,color=self.arrow_color,lw=1.5,alpha=self.arrow_alpha)
        slope = -1.0/np.tan(angle)
        x2,y2 = (node_radius+sr)*np.cos(angle)+a,(node_radius+sr)*np.sin(angle)+b
        head = node_radius*0.4
        try:
            n,m,dn,dm = arrow_head_coords((x2,y2),slope,head)
            self.ax.arrow(n,m,dn,dm, head_width=node_radius*0.2,head_length = head,
                          fc=self.arrow_color,ec=self.arrow_color,alpha = self.arrow_alpha)
        except ValueError:
            pass
        mx,Mx,my,My = min(x),max(x),min(y),max(y)
        if mx < self.minx:
            self.minx = mx
        if Mx > self.maxx:
            self.maxx = Mx
        if my < self.miny:
            self.miny = my
        if My > self.maxy:
            self.maxy = My
    def recursiveplot(self,N,node_radius):
        if self.Q.is_empty():
            return 
        v = self.Q.pop()
        N[v] = N[v].difference(self.seen)
        v2L = list(N[v])
        if len(v2L) == 0:
            N.pop(v)
            return self.recursiveplot(N,node_radius)
        elif len(v2L) == 1:
            theta = np.array([self.orbit_angle[v]])
        else:
            th = self.orbit_angle[v]
            theta = np.linspace(th-np.pi/4.0,th+np.pi/4.0,len(v2L))
        a,b = self.coords[v]
        x_0 = self.orbital_radius * np.cos(theta) + a
        y_0 = self.orbital_radius * np.sin(theta) + b
        self.orbital[v] = np.array([theta,x_0,y_0])
        v_num = 0
        for v2 in v2L:
            N[v].remove(v2)
            N[v2].remove(v)
            self.Q.add(v2)
            self.seen.add(v2)
            #t,x,y = self.orbital[v][:,v_num]
            t,x,y = theta[v_num],x_0[v_num],y_0[v_num]
            if x > self.maxx:
                self.maxx = x
            elif x < self.minx:
                self.minx = x
            if y > self.maxy:
                self.maxy = y
            elif y < self.miny:
                self.miny = y
            self.coords[v2] = (x,y)
            self.orbit_angle[v2] = t
            self.ax.add_patch(patches.Circle((x,y),node_radius,color=self.node_color))
            #self.ax.text(x,y,v2)
            v_num += 1
        N.pop(v)
        return self.recursiveplot(N,node_radius)    
    def myplot(self,amt = 'all'):
        self.fig = plt.figure(figsize = (60,60))
        if amt == 'all':
            amt = len(self.neighbors)
        sbplts = int(np.ceil(amt**.5))
        gs = gridspec.GridSpec(sbplts,sbplts,hspace=0,wspace=0)
        i = 0
        for K,NAc in self.neighbors.items()[0:(0+amt)]:
            N,A,c = NAc
            N = dc(N)
            m,v = self.max_neighbor[K]
            self.ax = self.fig.add_subplot(gs[i])
            if m == 0:
                m += 1
            node_radius = min([self.max_node_radius,(2.0*self.orbital_radius)/m])
            if (m % 2) == 0:
                m += 1
            theta = np.linspace(0,2*np.pi,(m+1))[0:-1]
            x_0 = self.orbital_radius * np.cos(theta)
            y_0 = self.orbital_radius * np.sin(theta)
            self.minx,self.maxx,self.miny,self.maxy = 0,0,0,0
            self.orbital = {v:np.array([theta,x_0,y_0])}
            self.orbit_angle = {v:0}
            self.all_angles = {}
            self.coords = {v:(0,0)}
            self.seen = {v}
            self.Q = Queue()
            self.ax.add_patch(patches.Circle((0,0),node_radius,color=self.node_color))
            #self.ax.text(0,0,v)
            v_num = 0
            v2L = list(N[v])
            for v2 in v2L:
                N[v].remove(v2)
                if v == v2:
                    continue
                N[v2].remove(v)
                self.Q.add(v2)
                self.seen.add(v2)
                #t,x,y = self.orbital[v][:,v_num]
                t,x,y = theta[v_num],x_0[v_num],y_0[v_num]
                if x > self.maxx:
                    self.maxx = x
                elif x < self.minx:
                    self.minx = x
                if y > self.maxy:
                    self.maxy = y
                elif y < self.miny:
                    self.miny = y
                self.coords[v2] = (x,y)
                self.orbit_angle[v2] = t
                self.ax.add_patch(patches.Circle((x,y),node_radius,color=self.node_color))
                #self.ax.text(x,y,v2)
                v_num += 1
            N.pop(v)
            self.recursiveplot(N,node_radius)
            self.minx,self.maxx = self.minx - node_radius, self.maxx + node_radius
            self.miny,self.maxy = self.miny - node_radius, self.maxy + node_radius
            adj_axis(self.ax,{'all off':True})
            self_loop_vs = []
            for v1 in A:
                for v2 in A[v1]:
                    x1,y1 = self.coords[v1]
                    x2,y2 = self.coords[v2]
                    if v1 == v2:
                        self_loop_vs.append(v1)
                        continue
                    theta = theta_finding((x1,y1),(x2,y2))
                    try:
                        self.all_angles[v1].append(theta)
                    except KeyError:
                        self.all_angles[v1] = [theta]
                    try:
                        self.all_angles[v2].append(z2pi(theta+np.pi))
                    except KeyError:
                        self.all_angles[v2] = [z2pi(theta+np.pi)]
                    head = node_radius * 0.4
                    x1,y1,dx,dy = arrow_points(node_radius,x1,y1,x2,y2,head)
                    self.ax.arrow(x1,y1,dx,dy, head_width=node_radius*0.2,
                             head_length = head,fc=self.arrow_color,ec=self.arrow_color,
                             alpha = self.arrow_alpha)
            for v in self_loop_vs:
                self.self_arrow_plot(v,node_radius)
            Dx = (np.abs(self.minx)+np.abs(self.maxx))
            Dy = (np.abs(self.miny)+np.abs(self.maxy))
            ratio=Dx/Dy
            if ratio < 1:
                add = (Dy - Dx)/2.0
                self.minx -= add
                self.maxx += add
            elif ratio > 1:
                add = (Dx - Dy)/2.0
                self.miny -= add
                self.maxy += add
            self.ax.set_xlim(left = self.minx,right=self.maxx)
            self.ax.set_ylim(bottom=self.miny,top = self.maxy)
            #xadd,yadd = np.abs(self.maxx - self.minx),np.abs(self.maxy - self.miny)
            #self.ax.text(self.maxx - xadd/8.0,self.miny + yadd/8.0,str(c),size=14)
            self.ax.text((self.minx+self.maxx)/2.0,(self.miny+self.maxy)/2.0,str(c),size=14,
                         horizontalalignment = 'center', verticalalignment = 'center')
            i += 1
        plt.show()           
    def netplot(self):
        self.fig = plt.figure(figsize = (60,60))
        amt = len(self.neighbors)
        sbplts = int(np.ceil(amt**.5))
        gs = gridspec.GridSpec(sbplts,sbplts,hspace=0,wspace=0)
        i = 0
        for K in self.neighbors:
            E = self.uniq_edges[K]
            G = nx.DiGraph()
            G.add_edges_from(E)
            ax = self.fig.add_subplot(gs[i])
            nx.draw_spring(G)
            i += 1
        plt.show()
            
def adding_patient_info(Network,Plot_Network,info_type = 'complexity'):
    if info_type == 'complexity':
        items = Plot_Network.complexity.items()
    else:
        raise ValueError("Sorry the 'info_type' which you inputted is not recognized")
    for patient,value in items:
        Network.Nets[patient][info_type] = value
    return Network


#pNPIpath2 = Plot_Network(NPIpath2)
#pNPIpath2.get_patient_edges()
#NPIpath2 = adding_patient_info(NPIpath2,pNPIpath2)
#NPIpath = adding_patient_info(NPIpath,pNPIpath2)
#NPIpath3 = adding_patient_info(NPIpath3,pNPIpath2)

def adj_axis(ax,kwargs):
    for param,val in kwargs.items():
        if param == 'spines':
            ax.spines['right'].set_visible(val)
            ax.spines['top'].set_visible(val)
            ax.spines['bottom'].set_visible(val)
            ax.spines['left'].set_visible(val)
        elif param == 'left spine':
            ax.spines['left'].set_visible(val)
        elif param == 'bottom spine':
            ax.spines['bottom'].set_visible(val)
        elif param == 'top spine':
            ax.spines['top'].set_visible(val)
        elif param == 'right spine':
            ax.spines['right'].set_visible(val)
        elif param == 'yticks':
            ax.set_yticks(val)
        elif param == 'yticklabels':
            if type(val) is tuple:
                ax.set_yticklabels(val[0])
                ax.tick_params(axis='y',labelsize = val[1])
            else:
                ax.set_yticklabels(val)
        elif param == 'xticks':
            ax.set_xticks(val)
        elif param == 'xticklabels':
            if type(val) is tuple:
                ax.set_xticklabels(val[0])
                ax.tick_params(axis='x',labelsize = val[1])
            else:
                ax.set_xticklabels(val)
        elif param == 'tick labelsize':
            ax.tick_params(axis='both',which='major',labelsize = val)
        elif param == 'xtick labelsize':
            ax.tick_params(axis='x',labelsize = val)
        elif param == 'ytick labelsize':
            ax.tick_params(axis='y',labelsize = val)
        elif param == 'top xtick':
            ax.tick_params(axis = 'x',which='both',top=val)
        elif param == 'bottom xtick':
            ax.tick_params(axis = 'x',which='both',bottom=val)
        elif param == 'left ytick':
            ax.tick_params(axis='y',which='both',left=val)
        elif param == 'right ytick':
            ax.tick_params(axis='y',which='both',right=val)
        elif param == 'all ticks':
            ax.tick_params(axis='x',which='both',top=val,bottom=val)
            ax.tick_params(axis='y',which='both',left=val,right=val)
        elif param == 'labelbottom':
            ax.tick_params(axis='x',labelbottom=val)
        elif param == 'labelleft':
            ax.tick_params(axis='y',labelleft=val)
        elif param == 'labelright':
            ax.tick_params(axis='y',labelright=val)
        elif param == 'labeltop':
            ax.tick_params(axis='x',labeltop=val)
        elif param == 'xlabel':
            if type(val) is tuple:
                if len(val) == 3:
                    ax.set_xlabel(val[0],size=val[1],labelpad=val[2])
                else:
                    ax.set_xlabel(val[0],size=val[1])
            else:
                ax.set_xlabel(val)
        elif param == 'ylabel':
            if type(val) is tuple:
                if len(val) == 3:
                    ax.set_ylabel(val[0],size=val[1],labelpad=val[2])
                else:
                    ax.set_ylabel(val[0],size=val[1])
            else:
                ax.set_ylabel(val)
        elif param == 'title':
            if type(val) is tuple:
                ax.set_title(val[0],size=val[1])
            else:
                ax.set_title(val)
        elif param == 'standard':
            val = not val
            ax.spines['right'].set_visible(val)
            ax.spines['top'].set_visible(val)
            ax.tick_params(axis = 'both',which='major',top=val,right=val)
        elif param == 'neat':
            neat_mode,r,c,tp,i = val
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            if ((i-1) % c) and ('y' in neat_mode):
                ax.tick_params(axis='y',labelleft=False,left=False)
                ax.spines['left'].set_visible(False)
            if (not ((i > (c*(r-1))) or (i + c > tp))) and ('x' in neat_mode):
                ax.tick_params(axis='x',labelbottom=False,bottom=False)
                if 'y' not in neat_mode:
                    ax.spines['bottom'].set_visible(False)
        elif param == 'all off':
            val = not val
            ax.spines['right'].set_visible(val)
            ax.spines['top'].set_visible(val)
            ax.spines['bottom'].set_visible(val)
            ax.spines['left'].set_visible(val)
            ax.tick_params(axis='both',which='major',top=val,right=val,bottom=val,left=val,
                           labelleft=val,labelbottom=val)
                           
def apply_white_ticks(ax,side = 'y'):
    if side == 'y':
        adj_axis(ax,{'bottom spine':False,'top spine':False,
                     'right spine':False,'top xtick':False,'bottom xtick':False,
                     'right ytick':False,'left ytick':False})
        x1,x2 = ax.get_xlim()
        Y = ax.get_yticks()
        for y in Y:
            ax.plot([x1,x2],[y,y],color='w',lw=1)
    elif side == 'x':
        adj_axis(ax,{'left spine':False,'top spine':False,
                     'right spine':False,'top xtick':'off','bottom xtick':'off',
                     'right ytick':'off','left ytick':'off'})
        y1,y2 = ax.get_ylim()
        X = ax.get_xticks()
        for x in X:
            ax.plot([x,x],[y1,y2],color='w',lw=1)
