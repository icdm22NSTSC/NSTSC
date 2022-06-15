# -*- coding: utf-8 -*-


import numpy as np
import torch
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from utils.Models import * 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Node phase classifier
class Node():
    def __init__(self, nodei):
       self.idx = nodei
       self.stoptrain = False
       self.trainidx = []
       self.testidx = []
       

# Assign train data
def Givetraintonode(Nodes, pronodenum, datanums):
    
    Nodes[pronodenum].trainidx = datanums
    
    return Nodes


# Assign test data
def Givevaltonode(Nodes, pronodenum, datanums):
    
    Nodes[pronodenum].validx = datanums
    
    return Nodes


# Construct a tree from node phase classifiers
def Build_tree(Xtrain, Xval, ytrain_raw, yval_raw, Epoch, classnum, \
               learnrate, savepath = './utils/'):
    
    Tree = {}
    pronodenum = 0
    maxnodenum = 0
    Tree[maxnodenum] = Node(maxnodenum)
    Tree = Givetraintonode(Tree, pronodenum, list(range(len(ytrain_raw))))
    Tree = Givevaltonode(Tree, pronodenum, list(range(len(yval_raw))))
    Modelnum = 7
    while pronodenum <= maxnodenum:
        if not Tree[pronodenum].stoptrain:
            Tree, trueidx, falseidx, trueidxt, falseidxt \
                = Trainnode(Tree, pronodenum, Epoch,\
            learnrate, Xtrain, ytrain_raw, Modelnum, savepath, \
                classnum, Xval, yval_raw)
        
            if maxnodenum < 32:
                if len(Tree[pronodenum].trueidx) > 0:
                    Tree, maxnodenum = Updateleftchd(Tree, pronodenum, \
                          maxnodenum, Xtrain, ytrain_raw, classnum,\
                              Xval, yval_raw)
                if len(Tree[pronodenum].falseidx) > 0:
                    Tree, maxnodenum = Updaterigtchd(Tree, pronodenum, \
                          maxnodenum, Xtrain, ytrain_raw, classnum,\
                              Xval, yval_raw)
    
        pronodenum += 1
        
    return Tree


# Train a node phase classifier
def Trainnode(Nodes, pronum, Epoch, lrt, X, y, Mdlnum, mdlpath, clsnum, Xv, yv):
    trainidx = Nodes[pronum].trainidx    
    Xori = X[trainidx,:]
    yori = y[trainidx]
    validx = Nodes[pronum].validx
    Xvori = Xv[validx,:]
    yvori = yv[validx]
    yoricount = County(yori, clsnum)
    yvoricount = County(yvori, clsnum)
    curclasses = np.where(yoricount!=0)[0]
    Nodes[pronum].ycount = yoricount
    Nodes[pronum].yvcount = yvoricount
    Nodes[pronum].predcls = yoricount.argmax()
    yori = np.array(yori)
    yori = torch.LongTensor(yori)
    yvori = torch.LongTensor(yvori)
    N, T = len(yori), int(Xori.shape[1]/3)
    ginibest = 1
    Xori = torch.Tensor(Xori)
    Xvori = torch.Tensor(Xvori)
    batch_size = N // 20
    if batch_size <= 1:
        batch_size = N
    tlnns = {}
    optimizers = {}
    X_rns = {}
    Losses = {}
    datadim = 3

    
    for mdli in range(1, Mdlnum):
        tlnn = eval('TL_NN' + str(mdli) + '(T)')
        tlnn = tlnn.to(device)
        optimizer = torch.optim.AdamW(tlnn.parameters(), lr = lrt)
        for epoch in range(Epoch):        
            rand_idx = np.array(range(N))           
            ytrain = yori
            IR = sum(ytrain==1)/sum(ytrain==0) 
            ytrain = torch.LongTensor(ytrain)
            X_batch = Variable(torch.Tensor(Xori[rand_idx,:])).to(device)
            y_batch = (ytrain[rand_idx]).to(device)
            w_batch = IR * (1-y_batch) 
            w_batch[w_batch==0] = 1
            X_rns = tlnn(X_batch[:,:T], X_batch[:,T:2*T],\
                                  X_batch[:,2*T:])
            Losses =  torch.sum(w_batch * (-y_batch * \
                          torch.log(X_rns + 1e-7) - (1-y_batch) * \
                          torch.log(1-X_rns + 1e-7)))
            
            optimizer.zero_grad()
            Losses.backward()
            optimizer.step()
            
            
            giniscores = torch.Tensor(Cptginisplit(tlnn, Xvori, yvori,\
                                                   T, clsnum))
            ginisminnum = int(giniscores.argmin().numpy())
            ginismin = giniscores.min()
            ginibest, Nodes = Update_gini(ginismin, ginibest, Nodes, tlnn, \
                              curclasses, ginisminnum, mdlpath, pronum)
            
                    
                    
    Nodes[pronum].bestmodel = torch.load(mdlpath + 'bestmodel.pkl')
                
    Xpred, accu, trueidx, falseidx = Cpt_Accuracy(Nodes[pronum].bestmodel,\
                                Xori, yori, T)
    Xpredv, accuv, trueidxv, falseidxv = Cpt_Accuracy(Nodes[pronum].bestmodel,\
                                Xvori, yvori, T)
    
    Nodes[pronum].trueidx = np.array(Nodes[pronum].trainidx)[trueidx]
    Nodes[pronum].falseidx = np.array(Nodes[pronum].trainidx)[falseidx]
    Nodes[pronum].trueidxv = np.array(Nodes[pronum].validx)[trueidxv]
    Nodes[pronum].falseidxv = np.array(Nodes[pronum].validx)[falseidxv]
    return Nodes, trueidx, falseidx, trueidxv, falseidxv  


# Expand left child node
def Updateleftchd(Nodes, pronum, maxnum, Xori, yori, clsnum, Xoriv, yoriv):
    Leftidx = Nodes[pronum].trueidx
    Leftidxv = Nodes[pronum].trueidxv
    yleft = yori[Leftidx]
    ylgini = Cptgininode(yleft, clsnum)
    yleftv = yoriv[Leftidxv]
    ylginiv = Cptgininode(yleftv, clsnum)
    maxnum += 1
    Nodes[maxnum] = Node(maxnum)
    Nodes = Givetraintonode(Nodes, maxnum, Leftidx)
    Nodes = Givevaltonode(Nodes, maxnum, Leftidxv)
    ylcount = County(yleftv, clsnum)
    Nodes[maxnum].ycount = ylcount
    Nodes[maxnum].predcls = ylcount.argmax()
    Nodes[maxnum].ginis = ylginiv
    if ylginiv == 0 or ylgini == 0:
        Nodes[maxnum].stoptrain = True
    else:
        Nodes[maxnum].stoptrain = False
    Nodes[pronum].leftchd = maxnum
    Nodes[maxnum].prntnb = pronum
    Nodes[maxnum].childtype = 'leftchild'
    
    return Nodes, maxnum
    

# Expand right child node
def Updaterigtchd(Nodes, pronum, maxnum, Xori, yori, clsnum, Xoriv, yoriv):
    Rightidx = Nodes[pronum].falseidx
    Rightidxv = Nodes[pronum].falseidxv
    yright = yori[Rightidx]
    yrgini = Cptgininode(yright, clsnum)
    yrightv = yoriv[Rightidxv]
    yrginiv = Cptgininode(yrightv, clsnum)
    maxnum += 1
    Nodes[maxnum] = Node(maxnum)
    Nodes = Givetraintonode(Nodes, maxnum, Rightidx)
    Nodes = Givevaltonode(Nodes, maxnum, Rightidxv)
    yrcount = County(yrightv, clsnum)
    Nodes[maxnum].ycount = yrcount
    Nodes[maxnum].predcls = yrcount.argmax()
    Nodes[maxnum].ginis = yrginiv
    if yrginiv == 0 or yrgini == 0:
        Nodes[maxnum].stoptrain = True
    else:
        Nodes[maxnum].stoptrain = False
    Nodes[pronum].rightchd = maxnum
    Nodes[maxnum].prntnb = pronum
    Nodes[maxnum].childtype = 'rightchild'
    
    return Nodes, maxnum


# Count the number of data in each class
def County(yori, clsnum):
    ycount = np.zeros((clsnum))
    for i in range(clsnum):
        ycount[i] = sum(yori == i)
    return ycount


# Gini index for classification at a node
def Cptginisplit(md, X, y, T, clsnum):
    ginis = []
    X = Variable(X).to(device)
    Xmd_preds = md(X[:,:T], X[:,T:2*T], X[:,2*T:]).cpu()
    Xmd_predsrd = torch.round(Xmd_preds)
    onesnum = torch.sum(Xmd_predsrd == 1.)
    ygroup1 = y[Xmd_predsrd == 1.]
    zerosnum = torch.sum(Xmd_predsrd == 0.)
    ygroup0 = y[Xmd_predsrd == 0.]
    ginimd = Cpt_ginigroup(onesnum, ygroup1, zerosnum, ygroup0, clsnum)
    ginis.append(ginimd)
    return ginis


# Gini index computation for each classifier
def Cpt_ginigroup(num1, y1, num0, y0, clsnum):
    y1prob = torch.zeros(clsnum)
    y0prob = torch.zeros(clsnum)
    y1N = len(y1)
    y0N = len(y0)
    nums = num1 + num0
    for i in range(clsnum):
        if y1N>0:
            y1prob[i] = sum(y1==i)/y1N
        if y0N>0:
            y0prob[i] = sum(y0==i)/y0N

    ginipt1 = 1 - torch.sum(y1prob**2)
    ginipt0 = 1 - torch.sum(y0prob**2)
    ginirt = (num1/nums) * ginipt1 + (num0/nums) * ginipt0
    return ginirt


# Gini index for a node
def Cptgininode(yori, clsn):
    yfrac = np.zeros(clsn)
    if len(yori)>0:
        for i in range(clsn):
            yfrac[i] = sum(yori==i)/len(yori)
        ginin = 1 - np.sum(yfrac ** 2)
    else:
        ginin = 0
    return ginin


def Update_gini(ginismin, ginibest, Nodes, tlnn, \
                              curclasses, num, mdlpath, pronum):
    if ginismin < ginibest:
        torch.save(tlnn, mdlpath + 'bestmodel.pkl')
        Nodes[pronum].ginis = ginismin
        ginibest = ginismin - 0
        Nodes[pronum].bstmdlclass = int(curclasses[num])
    return ginibest, Nodes


# Accuracy for a node phase classifier
def Cpt_Accuracy(mdl, X, y, T):
    X = Variable(torch.Tensor(X)).to(device)
    Xpreds = mdl(X[:,:T], X[:,T:2*T], X[:,2*T:])
    Xpredsnp = Xpreds.cpu().detach().numpy()
    Xpnprd = np.round(Xpredsnp)
    trueidx = np.where(Xpnprd == 1)[0]
    falseidx = np.where(Xpnprd == 0)[0]
    accup = accuracy_score(y, Xpnprd)
    
    return Xpredsnp, accup, trueidx, falseidx










