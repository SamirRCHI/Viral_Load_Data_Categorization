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
import Centroid as cntoid
import DTrules as dtr

def load_VLpath(filename = 'VL_based_on_result.csv',
                rename_dict = {0:'HVLS',1:'RVL',2:'SLVL',3:'DSVL',4:'SHVL'}):
    VLpath = Networks(None,'result','lab_date',tag='VL') 
    # 1) We have initated the Networks -- we do not build directly because
    #    the results are an exported replica of the original data.
    #    We have built a different algorithm to build from such data:
    VLpath.build_from_exported_csv(filename,'float')
    # 2) We built from the exported data.
    VLpath = auto_cluster(VLpath,False,4.8,'only',False,'l10',False,'pos only')
    # 3) We computted the hierarchical clustering.
    VLpath = rename_classes(VLpath,rename_dict) 
    # 4) We gave meaningful names to the clustering results.
    return VLpath

VLpath = load_VLpath()

def generate_figure(VLpath,fig_list=range(1,10),supp_fig_list=range(1,10)):
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
            M,true_c = get_training_data(VLpath)
            cntoid.comparative_plot(M,true_c,get_feat_names(),
                                    VLpath.class_colors)
        elif fig_num == 7:
            # Note: Decision Rules may be slightly different each time!
            M,true_c = get_training_data(VLpath) # First get the data
            Centroid = MLmodel('Centroid poly')
            Centroid.fit(M, true_c) # Fit the Centroid (featurs auto normalize)
            Centroid.learn_radii() # Radii must be learned.
            nM = Centroid.LT(M, True) # Normalize the features for DecisionTree
            DT = MLmodel('DecisionTree5')
            DT.fit(nM, true_c) # Fit to the normalized data
            DTR = dtr.DTrule_extraction(DT) # Extract rules.
            DTR.plot(get_feat_names(),class_colors=VLpath.class_colors, # Plot!
                     Centroid=Centroid,
                     title='Decision Tree Rules and Centroid Comparison')
        elif fig_num == 8:
            validatingVL(VLpath,'abs avg','Centroid poly',mxdays='prop',
                         specific_save_name='Fig8.pdf')
        elif fig_num == 9:
            print("If class assignment scores are already saved, it will"+\
                  " be faster to generate this figure by skipping"+\
                  " the call to 'get_scores'- otherwise expect 6-7 hours...")
            score,score_names = get_scores(VLpath)
            wilcoxon_test(VLpath,scores,score_names,'Fig9.pdf')
        else:
            print("Fig #: "+str(fig_num)+EM)
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
            print("Fig S#: "+str(fig_num)+EM)
    