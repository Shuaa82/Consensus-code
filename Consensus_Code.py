#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
from itertools import chain, repeat
import json
import math
import recordlinkage
import os.path



#Read native protein data (long type)
def NativeD(filename1): 
    with open(filename1) as f:
        df = json.load(f) 
    chainer = chain.from_iterable
    NATIVE = pd.DataFrame({'R1': list(chainer(repeat(k, len(v)) for k, v in df.items())), 'R2' : list (chainer(df.values()))})
    NATIVE['R1'] = NATIVE['R1'].astype(int)
    NATIVE['R2']= NATIVE['R2'].astype(int)
    NATIVE['true_class']= (NATIVE['R2'] != 0).astype(int)
    #save residue pairs of native data in new dataframe for evaluation process:
    NL=NATIVE[['R1', 'R2']]
    
    #comput the number of native contacts for evaluation process:
    Nc= []
    for index, row in NATIVE.iterrows():
        if row['R1'] > row['R2']:
            if (row['R1']-row['R2']) >= 24:
                Nc.append(row ['R2'])
            else:
                pass
        else:
            pass
    
    return (NATIVE, NL, Nc)


#recall native data 

Native_data = NativeD(sys.argv[1])

native = Native_data[0] #native data
RR_native = Native_data[1] # residue pairs
Nc_N = len(Native_data[2]) # the number of native contacts
print(Nc_N)




# A function to read and extract prediction data of individual methods:

def subset(filename):
    with open(filename, 'r') as f:
        df1 = json.load(f)
        for k in df1.values():# Extracting data and save as dataframe:
            key = [k]
            for j in key:
                values=[j] 
                S =list(values[0].values())
        L_10 = S[0] # top10 set
        L_5= S[1]   # L/5 set
        L_2= S[2]   # L/2 set
        FL_0 = S[3] # FL set 
        L_0 = S[4]  # L set

        # save the subsets as dataframes:

        L10 =pd.DataFrame(L_10, columns=['R1', 'R2', 'P'], dtype = int) # top 10 of predicted contact
        L5 = pd.DataFrame(L_5, columns=['R1', 'R2', 'P'], dtype= int)   # L/5 set
        L2 = pd.DataFrame(L_2, columns=['R1', 'R2', 'P'], dtype = int)  # L/2 set
        L = pd.DataFrame(L_0, columns=['R1', 'R2', 'P'], dtype= int)    # L set
        FL = pd.DataFrame(FL_0, columns=['R1', 'R2', 'P'], dtype = int) # Full list

        # convert all numbers of residues pairs from string to integer: 

        L10['R2']= L10['R2'].astype(int)
        L10['R1'] = L10['R1'].astype(int)
        L10['P'] = L10['P'].astype(float)
        L5['R2']= L5['R2'].astype(int)
        L5['R1'] = L5['R1'].astype(int)
        L5['P']= L5['P'].astype(float)
        L2['R2']= L2['R2'].astype(int)
        L2['R1'] = L2['R1'].astype(int)
        L2['P']= L2['P'].astype(float)
        L['R2']= L['R2'].astype(int)
        L['R1'] = L['R1'].astype(int)
        L['P']= L['P'].astype(float)
        FL['R2']= FL['R2'].astype(int)
        FL['R1'] = FL['R1'].astype(int)
        FL['P'] = FL['P'].astype(float)
        
        return (L10, L5, L2, L, FL)




# building a function for classifying consensus data into subsets (top10, L/5, L/2, L, FL):
def sets(length, data, p_value):
    for x in range(length):
        if x == 10:
            Top10=(data.nlargest(x, p_value))
            Top10= Top10.astype(str).values.tolist()
            L10 =pd.DataFrame(Top10, columns=['R1', 'R2', 'ConsP'], dtype = int) # top 10 of predicted contact
            L10['R1']= L10['R1'].astype(int)
            L10['R2'] = L10['R2'].astype(int)
            L10['ConsP']= L10['ConsP'].astype(float)
            
        elif x == math.ceil(length/5):
            L_5= (data.nlargest(x, p_value))
            L_5 = L_5.astype(str).values.tolist()
            L5 = pd.DataFrame(L_5, columns=['R1', 'R2', 'ConsP'], dtype= int)   # L/5 set
            L5['R1']= L5['R1'].astype(int)
            L5['R2'] = L5['R2'].astype(int)
            L5['ConsP']= L5['ConsP'].astype(float)


        elif x == math.ceil(length/2):
            L_2 = (data.nlargest(x, p_value))
            L_2 =L_2.astype(str).values.tolist() 
            L2 = pd.DataFrame(L_2, columns=['R1', 'R2', 'ConsP'], dtype = int)  # L/2 set
            L2['R1']= L2['R1'].astype(int)
            L2['R2'] = L2['R2'].astype(int)
            L2['ConsP']= L2['ConsP'].astype(float)

        else:
            L_1 = (data.nlargest(length, p_value))
            L_1 = L_1.astype(str).values.tolist()
            L = pd.DataFrame(L_1, columns=['R1', 'R2', 'ConsP'], dtype= int)    # L set
            L['R1']= L['R1'].astype(int)
            L['R2'] = L['R2'].astype(int)
            L['ConsP']= L['ConsP'].astype(float)


             
            FL_0=data.astype(str).values.tolist()
            FL = pd.DataFrame(FL_0, columns=['R1', 'R2', 'ConsP'], dtype = int) # Full list
            FL['R1']= FL['R1'].astype(int)
            FL['R2'] = FL['R2'].astype(int)
            FL['ConsP'] = FL['ConsP'].astype(float)
            FL = FL.drop_duplicates(subset=['R1', 'R2'], keep ='first')
    return (L10, L5, L2, L, FL) 

#Building function that computing scores of evaluation measurements:
def Scores(experD, predD, class1, class2, v):
    # experD is the native contact data
    # predD is the prediction contact data
    # class1 is the positive cases in prediction data (contacts)
    # class 2 is the true cases in native data
    # v is the number of all contacts in  native data
    #save the native and prediction dataframe with multiindex (R1, R2):

    native =pd.MultiIndex.from_frame(experD, names=('R1', 'R2'))
    pred =pd.MultiIndex.from_frame(predD, names=('R1', 'R2'))

    # compute confusion matrix using recordlinkage package:

    Conf= recordlinkage.confusion_matrix(native, pred)
    
    # count TP and FP from confusion matrix
    TP= Conf[0][0]
    FP= Conf[1][0]
    
    #count TN and FN according to the true cases of native and against to positive and negative cases of prediction data, where class1 = 0 represent negative cases (non-contact) and class1 = 1 represent positive cases (contact):
    TN = np.sum(np.logical_and(class1 == 0, class2 == 0))
    FN = np.sum(np.logical_and(class1 == 1, class2 == 0))
       

    # calculate precision, recall and f1_score:   
    if TP != 0:
        Precision =recordlinkage.precision(native, pred)
        Recall = (TP/v)
        f1_score= 2* Precision * Recall/(Precision + Recall)
                   
    else: 
        Precision =0
        Recall = 0
        f1_score = 0
    
    # save the scores as list:       
    scores=[TP, FP, FN, TN, Precision*100, Recall*100, f1_score*100]          
    return scores

    
 

# build a function to apply consensus method: 

#Consensus two methods: 

def Cons2(d, d1): 
    #first: merge the input data into one dataframe: 
    # d, d1 represent the input data: 
 
    data= pd.merge(d, d1, on=['R1', 'R2'], how='outer') 
   
   #padding p_value into zero and covert its type to float: 
    data['P_x'].fillna(0, inplace = True) 
    data['P_y'].fillna(0, inplace = True) 
    data['P_x'] = data['P_x'].astype(float) 
    data['P_y']= data['P_y'].astype(float) 
   
   #Second: Consensus prediction: 
    #calculating mean of probabilities for each residue pairs from two data 
    data['ConsP'] = data[['P_x', 'P_y']].mean(axis=1) 
    data['ConsP'].fillna(0, inplace = True) 
    pd.options.display.float_format = '{:.3f}'.format # display consensus p_value as 3 digits 
    
    #convert type of residue pairs into intger 
    data['R1'] = data['R1'].astype(int) 
    data['R2']= data['R2'].astype(int) 
    
    #save consensus data into a new dataframe 
    ConsD= data[['R1', 'R2', 'ConsP']] 
    return ConsD 
    


#Consensus three methods: 

def Cons3(d, d1, d2): 

    #first merge the input data into one datafram: 
    # d, d1, d2 represent the input data: 

    dfs=[d, d1, d2] 
    df =pd.merge(dfs[0], dfs[1], on=['R1','R2'], how='outer') 

    for d in dfs[2:]: 
        data=pd.merge(df, d, on=['R1','R2'], how='outer') 


    # padding p_value into zero and covert its type to float: 
    data['P_x'].fillna(0, inplace = True) 
    data['P_y'].fillna(0, inplace = True) 
    data['P'].fillna(0, inplace = True) 
    data['P_x'] = data['P_x'].astype(float) 
    data['P_y']= data['P_y'].astype(float) 
    data['P']= data['P'].astype(float) 


    #Consensus prediction: 
    # calculating mean of probabilities for each residue pairs from three data  
    data['ConsP'] = data[['P_x', 'P_y', 'P']].mean(axis=1) 

    #padding consensus p_value into zero and convert its type to float 
    data['ConsP'].fillna(0, inplace = True) 
    pd.options.display.float_format = '{:.3f}'.format # display consensus p_value as 3 digits 


    # convert type of residue pairs into intger 
    data['R1'] = data['R1'].astype(int) 
    data['R2']= data['R2'].astype(int) 

    #save consensus data into a new dataframe 
    ConsD= data[['R1', 'R2', 'ConsP']] 
    return ConsD 

  
def main(argv):
    # reading input data and save them as dataframes:
    input1= subset(sys.argv[2])
    input2 = subset(sys.argv[3]) 
    input3= subset(sys.argv[4]) 
    
    
    # Appling consensus method : 
    Cons2A = Cons2(input1[4], input2[4]) 
    Cons2B = Cons2(input1[4], input3[4]) 
    Cons2C = Cons2(input2[4],input3[4]) 
    Cons3M = Cons3(input1[4], input2[4], input3[4]) 
    
    
    # Categrizing consensus data into subsets(Top10, L/5, L/2, L, FL):
    # This step was repeated for each consensus method:
    # the fifth argument in command line will be the sequence length:

    sets1 = sets(int(sys.argv[5]), Cons3M, 'ConsP')

    # Top10

    Pred_top10 = sets1[0]
    top10= Pred_top10[['R1', 'R2']]

    # L/5 set
    Pred_L5= sets1[1]
    L5= Pred_L5[['R1', 'R2']]

    # L/2 set:
    Pred_L2 =sets1[2]
    L2 = Pred_L2[['R1', 'R2']]


    # L set:
    Pred_L = sets1[3]
    L = Pred_L[['R1', 'R2']]

    #FL set:
    Pred_FL = sets1[4]
    FL= Pred_FL[['R1', 'R2']]

    
    #save prediction data as rr format file for ConEva tool: 

    #Cons2A 
    Cons2A['D_min'] ='0' 
    Cons2A['D_max']= '8' 
    Cons2A=Cons2A[['R1', 'R2', 'D_min', 'D_max', 'ConsP']] 


    dirA= 'file path' 
    fileA='{}.rr'.format(str(sys.argv[6])) 
    file_path_A=os.path.join(dirA, fileA) 

    with open(file_path_A, 'w') as fa: 
        fa.write(Cons2A.to_string(header = False, index= False)) 

 

    #Cons2B 
    Cons2B['D_min'] ='0' 
    Cons2B['D_max']= '8' 
    Cons2B=Cons2B[['R1', 'R2', 'D_min', 'D_max', 'ConsP']] 

    dirB= 'file path' 
    fileB='{}.rr'.format(str(sys.argv[6])) 
    file_path_B=os.path.join(dirB, fileB) 

    with open(file_path_B, 'w') as fb: 
        fb.write(Cons2B.to_string(header = False, index= False)) 

    #Cons2C: 

    Cons2C['D_min'] ='0' 
    Cons2C['D_max']= '8' 
    Cons2C=Cons2C[['R1', 'R2', 'D_min', 'D_max', 'ConsP']] 

    dirC= 'file path' 
    fileC='{}.rr'.format(str(sys.argv[6])) 
    file_path_C=os.path.join(dirC, fileC) 

    with open(file_path_C, 'w') as fc: 
        fc.write(Cons2C.to_string(header = False, index= False)) 


    #Cons3: 
    Cons3M['D_min'] ='0' 
    Cons3M['D_max']= '8' 
    Cons3M=Cons3M[['R1', 'R2', 'D_min', 'D_max', 'ConsP']] 

    dir3M= 'file path' 
    file3M='{}.rr'.format(str(sys.argv[6])) 
    file_path_3M=os.path.join(dir3M, file3M) 

    with open(file_path_3M, 'w') as f3m: 
        f3m.write(Cons3M.to_string(header = False, index= False)) 



    
 # Third step:
    # Evaluation measures (Precision, Recall, F1_measure):
    # A- merge native contact data with predicted data based on the residue pairs of both data for compraison:
    

    # native contact with top 10 set of predicted contact data:
    F0= pd.merge(native, Pred_top10, on= ['R1', 'R2'], how='right' )
    
            
    F0['R2'].fillna(0, inplace = True)
    F0['R2'] = F0['R2'].astype(int)
    F0['ConsP'].fillna(0, inplace = True)
    F0['ConsP'] = F0['ConsP'].astype(float)
    F0['true_class'].fillna(0, inplace = True)
    F0['true_class'] = F0['true_class'].astype(int)


    #native contact with L/5 set od predicted contact data:
    F1 = pd.merge(native, Pred_L5, on=['R1', 'R2'], how='right')
    
    F1['R2'].fillna(0, inplace = True)
    F1['R2'] = F1['R2'].astype(int)
    F1['ConsP'].fillna(0, inplace = True)
    F1['ConsP'] = F1['ConsP'].astype(float)
    F1['true_class'].fillna(0, inplace = True)
    F1['true_class'] = F1['true_class'].astype(int)
    

    # save consensus prediction data in another file for Precision-Recall curve analysis
    F_1 =F1[['ConsP', 'true_class']]
    dirB= 'file path'
    fileB='{}.csv'.format(str(sys.argv[3]))
    file_path_B= os.path.join(dirB, fileB)
    F_1.to_csv(file_path_B, index=False)
    

    # native contact with L/2 set of predicted contact data:

    F2 = pd.merge(native, Pred_L2, on=['R1', 'R2'],  how='right')


    F2['R2'].fillna(0, inplace = True)
    F2['R2'] = F2['R2'].astype(int)
    F2['ConsP'].fillna(0, inplace = True)
    F2['ConsP'] = F2['ConsP'].astype(float)
    F2['true_class'].fillna(0, inplace = True)
    F2['true_class'] = F2['true_class'].astype(int)
                        
            
      #native contact with L set:
    F3 = pd.merge(native, Pred_L, on= ['R1', 'R2'], how = 'right')


    F3['R2'].fillna(0, inplace = True)
    F3['R2'] = F3['R2'].astype(int)
    F3['ConsP'].fillna(0, inplace = True)
    F3['ConsP'] = F3['ConsP'].astype(float)
    F3['true_class'].fillna(0, inplace = True)
    F3['true_class'] = F3['true_class'].astype(int) 
   
    
    # native contact with FL set: 
    F4 = pd.merge(native, Pred_FL, on= ['R1', 'R2'], how = 'right')


    F4['R2'].fillna(0, inplace = True)
    F4['R2'] = F4['R2'].astype(int)
    F4['ConsP'].fillna(0, inplace = True)
    F4['ConsP'] = F4['ConsP'].astype(float)
    F4['true_class'].fillna(0, inplace = True)
    F4['true_class'] = F4['true_class'].astype(int)
    
    
     # save consensus prediction data in another file for Precsion-Recall curve analysis
    F_4 = F4[['ConsP', 'true_class']]
        
    dirB= 'file path'
    fileB='{}.csv'.format(str(sys.argv[3]))
    file_path_B=os.path.join(dirB, fileB)
    F_4.to_csv(file_path_B, index=False)
    

     # Classification consensus prediction data into contact and noncontact at p-value >0:

    F0['Cor'] = (F0['ConsP'] > 0).astype(int)
    F1['Cor'] = (F1['ConsP'] > 0).astype(int)
    F2['Cor'] = (F2['ConsP'] > 0).astype(int)
    F3['Cor'] = (F3['ConsP'] > 0).astype(int)
    F4['Cor'] = (F4['ConsP'] > 0).astype(int)
     
     
     # C- Evaluation process:
       #Make confusion matrix for each subsets of predicted contact data to calculate tp, tn, fp, fn values:
       # Computing Precision, Recall and F1_measure:
       # depending on CASP assessores evaluation of contact prediction method: 
       # Precision = TP of subset / len of predicted contact set:
       # Recall = TP of subset / len of native contact data:
       # f1_score = 2 * Precision* Recall/ (Precision + Recall):
    
    # top 10 set:
    scores_top10= Scores(RR_native, top10, F0.Cor, F0.true_class, Nc_N)
    df_top10 =pd.DataFrame([scores_top10], columns=['TP','FP', 'FN', 'TN', 'Precision', 'Recall', 'f1_score'], index =['Top10'], dtype=int)
    df_top10.index.name = 'set'

    dir_top10='file path'
    file_top10='{}.csv'.format(str(sys.argv[6]))
    file_top10_path=os.path.join(dir_top10, file_top10)
    df_top10.to_csv(file_top10_path)

    
    #L/5 set:
    scores_L5= Scores(RR_native, L5, F1.Cor, F1.true_class, Nc_N)
    df_L5 =pd.DataFrame([scores_L5], columns=['TP','FP', 'FN', 'TN', 'Precision', 'Recall', 'f1_score'], index=['L/5'], dtype=int)
    df_L5.index.name ='set'


    dir_L5='file path'
    file_L5='{}.csv'.format(str(sys.argv[6]))
    file_L5_path=os.path.join(dir_L5, file_L5)
    df_L5.to_csv(file_L5_path)
            
    # L/2 set:
    scores_L2= Scores(RR_native, L2, F2.Cor, F2.true_class, Nc_N)
    df_L2 =pd.DataFrame([scores_L2], columns=['TP','FP', 'FN', 'TN', 'Precision', 'Recall', 'f1_score'], index = ['L/2'], dtype=int)
    df_L2.index.name = 'set'

    dir_L2='file path'
    file_L2='{}.csv'.format(str(sys.argv[6]))
    file_L2_path=os.path.join(dir_L2, file_L2)
    df_L2.to_csv(file_L2_path, index=False)


    # L set:
    scores_L= Scores(RR_native, L, F3.Cor, F3.true_class, Nc_N) 
    df_L =pd.DataFrame([scores_L], columns=['TP','FP', 'FN', 'TN', 'Precision', 'Recall', 'f1_score'], index= ['L'], dtype=int)
    df_L.index.name = 'set'

    dir_L='file path'
    file_L='{}.csv'.format(str(sys.argv[6]))
    file_L_path=os.path.join(dir_L, file_L)
    df_L.to_csv(file_L_path, index=False)

    #FL set:
    scores_FL = Scores(RR_native, FL, F4.Cor, F4.true_class, Nc_N)
    df_FL =pd.DataFrame([scores_FL], columns=['TP','FP', 'FN', 'TN', 'Precision', 'Recall', 'f1_score'], index=['FL'], dtype=int)
    df_FL.index.name = 'set'

    dir_FL='file path'
    file_FL='{}.csv'.format(str(sys.argv[6]))
    file_FL_path=os.path.join(dir_FL, file_FL)
    df_FL.to_csv(file_FL_path, index=False)

    #save the scores evaluation into dataframe:
    df =pd.DataFrame([scores_top10, scores_L5, scores_L2, scores_L, scores_FL], columns=['TP','FP', 'FN', 'TN', 'Precision', 'Recall', 'f1_score'], index=['Top10', 'L/5', 'L/2', 'L', 'FL'],  dtype=int)
    df.index.name = 'sets'

    
    dir_B='file path'
    file_B='{}.csv'.format(str(sys.argv[6]))
    file_B_path=os.path.join(dir_B, file_B)
    df.to_csv(file_B_path, index=False)
    print(df)
    
    
if __name__ =='__main__':
    main(sys.argv)


