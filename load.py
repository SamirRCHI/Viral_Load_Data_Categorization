# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 08:43:54 2017

Company: University of Rochester Medical Center
Team: Rochester Center for Health Informatics
Supervisor: Dr. Martin Zand (martin_zand@urmc.rochester.edu)
Author: Samir Farooq (samir_farooq@urmc.rochester.edu)
"""

from Networking import *
from VLfeats import *
from copy import deepcopy as dc

#CD4path = Networks('CD4_counts.csv','CD4_result','lab_date',tag='CD4',
#                   conversion_rules={'CD4_result':'int'})
#Diagpath = Networks('Diagnosis Codes.csv','description','diag_date',tag='Diag')  
#
#NPIpath = Networks('NPIpath.csv')
#NPIpath1 = dc(NPIpath)
#NPIpath2 = dc(NPIpath)
#NPIpath3 = dc(NPIpath)

def load_VLpath(filename = 'VLpath.csv',
                rename_dict = {0:'HVLS',1:'RVL',2:'SLVL',3:'DSVL',4:'SHVL'}):
    VLpath = Networks(filename,'result','lab_date',tag='VL', # Load the viral load network
                      conversion_rules={'result':'int'})
    VLpath = auto_cluster(VLpath,False,4.8,'only',False,'l10',False,'pos only') # Compute hierarchical clustering
    VLpath = rename_classes(VLpath,rename_dict) # Give meaningful names to the clusters found.
    return VLpath

#VLpathT = load_VLpath()

def generate_figure(VLpath,fig_list=range(1,8),supp_fig_list=range(1,10)):
    EM = " - either doesn't exist or is generated outside of python."
    for fig_num in fig_list:
        if fig_num == 1: 
            PlotSingleAxEx(save='Fig1.pdf')
        elif fig_num == 2:
            auto_cluster(VLpath,'Hierarchical Patient Clustering',4.8,
                         False,False,'l10',True,'pos only',save='Fig2.pdf')
        elif fig_num == 3:
            VLplot(VLpath,save='Fig3.pdf')
        elif fig_num == 4:
            feat_vs_feat(VLpath,True,'Fig4.pdf')
        elif fig_num == 5:
            comparative_patterns(VLpath,'teal,black','Fig5.pdf')
        elif fig_num == 6:
            validatingVL(VLpath,'abs avg','Centroid poly',mxdays='prop',
                         specific_save_name='Fig6.pdf')
        elif fig_num == 7:
            print "If class assignment scores are already saved, it will"+\
                  " be faster to generate this figure by skipping"+\
                  " the call to 'get_scores'- otherwise expect 6-7 hours..."
            score,score_names = get_scores(VLpath)
            wilcoxon_test(VLpath,scores,score_names,'Fig7.pdf')
        else:
            print "Fig #: "+str(fig_num)+EM
    for fig_num in supp_fig_list:
        if fig_num == 1:
            before_and_after(VLpath,'FigS1.pdf')
        elif fig_num == 3:
            feat_vs_feat(VLpath,False,'FigS3.pdf')
        elif fig_num == 4:
            generate_DTpdf(VLpath,False,'FigS4.pdf')
        elif fig_num == 5:
            validatingVL(VLpath,'abs avg','DecisionTree',mxdays='prop',
                         specific_save_name='FigS5.pdf')
        elif fig_num == 6:
            validatingVL(VLpath,'abs avg','SVC',mxdays='prop',
                         specific_save_name='FigS6.pdf')
        elif fig_num == 7:
            validatingVL(VLpath,'abs avg','kNN5',mxdays='prop',
                         specific_save_name='FigS7.pdf')
        elif fig_num == 8:
            validatingVL(VLpath,'abs avg','Centroid poly projection',
                         mxdays='prop',specific_save_name='FigS8.pdf')
        else:
            print "Fig S#: "+str(fig_num)+EM
    

#NPIpath1.add_classes(VL_class,VLpath1,print_result = False)
#class_transfer(VLpath2,NPIpath2)
#NPIpath3.add_classes(Terzian_Classifier_New,VLpath3,print_result = False)
#NPIpath1.set_class_colors({'High Risk':'c','Remission':'m','Low Risk':'r','Healthy':'g'})
#NPIpath = class_subsetting(NPIpath,[NPIpath1,NPIpath2,NPIpath3],
#                           {'Pure SHVL':['High Risk','c','SHVL'],
#                            'Pure Remission':['Remission','m','SHVL Suppression']})
#class_transfer(NPIpath,VLpath1)
#
#Full_Net = dc(NPIpath)
#merge_Networks(Full_Net,[VLpath1,CD4path,Diagpath],equi_color(4))
#
#NPIpath.Plot.get_patient_edges()
#
#NPIpath1 = adding_patient_info(NPIpath1,NPIpath.Plot)