author: Samir Farooq
company: University of Rochester Medical Center
team: Rochester Center for Health Informatics
email: samir_farooq@urmc.rochester.edu or martin_zand@urmc.rochester.edu
last revised: 18 June 2018

We have included 6 python scripts: VLfeats.py, Networking.py, Centroid.py, DTrules.py, hierarchical.py, and VLload.py.


VLfeats.py
__________

This is the script file containing all of the functions involved in our viral load research. An extensive (but not exhaustive) documentation has been written: read VLfeats.py documentation.




Networking.py
_____________

This is a data structure to handle patient paths. Read Networking.py documentation.




Centroid.py
___________

This is the algorithm which performs cluster summarization/generalization into centers and radii (or hyperplanes). Read Centroid.py documentation.




DTrules.py
__________

Contains the algorithm which mines sci-kit learn's Decision Tree results in order to extract decision rules. Please read the DTrules.py Documentation for more details.




hierarchical.py
_______________

Contains a more versatile method of plotting scipy's linkage results (i.e. an improved version of scipy's dendrogram function). Pleas read the hierarchical.py Documentation for more details.




VLload.py
_________

This script file uses all of the above mentioned scripts which shows an example of how to load the viral load network from a .csv file into the Networks class from Networking.py, followed by using the auto_cluster function to generate clusters, and then renaming the classes (lines 24-27).

The function generate_figure takes as input: VLpath, fig_list, and supp_fig_list:
-VLpath is the viral load patient paths loaded from the function: load_VLpath (line 22).
-fig_list takes a list of integers corresponding to the figure the user wants to generate (in regards to the figure order of the corresponding paper to these script files).
-supp_fig_list takes a list of integers corresponding to the supplementary figure the user wants to generate (in regards to the supplementary figure order of the corresponding paper to these script files).