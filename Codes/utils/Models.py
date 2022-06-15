# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# conjunction of different predicates
class TL_NN1(nn.Module):
    def __init__(self, T):
        super(TL_NN1,self).__init__()
        self.t1 = torch.nn.Parameter(1e-5*torch.randn(1,T), requires_grad=True)
        self.t1_2 = torch.nn.Parameter(1e-5*torch.randn(1,T), requires_grad=True)
        self.t2 = torch.nn.Parameter(1e-5*torch.randn(1,T), requires_grad=True)
        self.t2_2 = torch.nn.Parameter(1e-5*torch.randn(1,T), requires_grad=True)
        self.t3 = torch.nn.Parameter(1e-5*torch.randn(1,T), requires_grad=True)
        self.t3_2 = torch.nn.Parameter(1e-5*torch.randn(1,T), requires_grad=True)
        self.b1 = torch.nn.Parameter(torch.randn(1,T), requires_grad=True)
        self.b1_2 = torch.nn.Parameter(1e-5*torch.randn(1,T), requires_grad=True)
        self.b2 = torch.nn.Parameter(torch.randn(1,T), requires_grad=True)
        self.b2_2 = torch.nn.Parameter(1e-5*torch.randn(1,T), requires_grad=True)
        self.b3 = torch.nn.Parameter(torch.randn(1,T), requires_grad=True)
        self.b3_2 = torch.nn.Parameter(1e-5*torch.randn(1,T), requires_grad=True)
        self.A1 = torch.nn.Parameter(torch.rand(1,T),requires_grad=True)
        self.A2 = torch.nn.Parameter(torch.rand(1,T),requires_grad=True)
        self.A3 = torch.nn.Parameter(torch.rand(1,T),requires_grad=True)
        self.A4 = torch.nn.Parameter(torch.rand(1,3),requires_grad=True)
        self.beta1 = torch.nn.Parameter(torch.tensor(1.),requires_grad=True)
        self.beta2 = torch.nn.Parameter(torch.tensor(1.),requires_grad=True)
        self.beta3 = torch.nn.Parameter(torch.tensor(1.),requires_grad=True)
        self.beta4 = torch.nn.Parameter(torch.tensor(1.),requires_grad=True)

#        
    def forward(self,x1, x2, x3):
         self.r_a1 = x1 * self.t1 - self.b1
         self.r_asgm1 = torch.sigmoid(self.r_a1) # convert to 0-1 range
         self.A_sm1 =  F.softmax(self.A1, dim = 1)
         self.weightbias1 = self.beta1 - torch.sum(self.A_sm1 * (1 - self.r_asgm1), 1)
         self.activate1 = clamp(self.weightbias1).reshape([-1,1])
         
         
         self.r_a2 = x2 * self.t2 - self.b2
         self.r_asgm2 = torch.sigmoid(self.r_a2) # convert to 0-1 range
         self.A_sm2 =  F.softmax(self.A2, dim = 1)
         self.weightbias2 = self.beta2 - torch.sum(self.A_sm2 * (1 - self.r_asgm2), 1)
         self.activate2 = clamp(self.weightbias2).reshape([-1,1])
        
         
         self.r_a3 = x3 * self.t3 - self.b3
         self.r_asgm3 = torch.sigmoid(self.r_a3) # convert to 0-1 range
         self.A_sm3 =  F.softmax(self.A3, dim = 1)
         self.weightbias3 = self.beta3 - torch.sum(self.A_sm3 * (1 - self.r_asgm3), 1)
         self.activate3 = clamp(self.weightbias3).reshape([-1,1])
        
         self.r_asgm4 = torch.cat((self.activate1, self.activate2, self.activate3),1)
         self.A_sm4 = F.softmax(self.A4, dim = 1)
         self.weightbias4 = self.beta4 - torch.sum(self.A_sm4 * (1 - self.r_asgm4), 1)
         self.activate4 = clamp(self.weightbias4).reshape([-1])
        
         return self.activate4.reshape([-1])

    def Const_loss(self, alpha = 0.5):
        loss_1 = 1 - torch.sum(self.A_sm1 ** 2)
        loss_2 = 1 - torch.sum(self.A_sm2 ** 2)
        loss_3 = 1 - torch.sum(self.A_sm3 ** 2)
        loss_4 = 1 - torch.sum(self.A_sm4 ** 2)
        
        loss_ret = loss_1 + loss_2 + loss_3 + loss_4
        
        return loss_ret


# disjunction of different predicates
class TL_NN2(nn.Module):
    def __init__(self, T):
        super(TL_NN2,self).__init__()
        self.t1 = torch.nn.Parameter(1e-5*torch.randn(1,T), requires_grad=True)
        self.t1_2 = torch.nn.Parameter(1e-5*torch.randn(1,T), requires_grad=True)
        self.t2 = torch.nn.Parameter(1e-5*torch.randn(1,T), requires_grad=True)
        self.t2_2 = torch.nn.Parameter(1e-5*torch.randn(1,T), requires_grad=True)
        self.t3 = torch.nn.Parameter(1e-5*torch.randn(1,T), requires_grad=True)
        self.t3_2 = torch.nn.Parameter(1e-5*torch.randn(1,T), requires_grad=True)
        self.b1 = torch.nn.Parameter(torch.randn(1,T), requires_grad=True)
        self.b1_2 = torch.nn.Parameter(1e-5*torch.randn(1,T), requires_grad=True)
        self.b2 = torch.nn.Parameter(torch.randn(1,T), requires_grad=True)
        self.b2_2 = torch.nn.Parameter(1e-5*torch.randn(1,T), requires_grad=True)
        self.b3 = torch.nn.Parameter(torch.randn(1,T), requires_grad=True)
        self.b3_2 = torch.nn.Parameter(1e-5*torch.randn(1,T), requires_grad=True)
        self.A1 = torch.nn.Parameter(torch.rand(1,T),requires_grad=True)
        self.A2 = torch.nn.Parameter(torch.rand(1,T),requires_grad=True)
        self.A3 = torch.nn.Parameter(torch.rand(1,T),requires_grad=True)
        self.A4 = torch.nn.Parameter(torch.rand(1,3),requires_grad=True)
        self.beta1 = torch.nn.Parameter(torch.tensor(1.),requires_grad=True)
        self.beta2 = torch.nn.Parameter(torch.tensor(1.),requires_grad=True)
        self.beta3 = torch.nn.Parameter(torch.tensor(1.),requires_grad=True)
        self.beta4 = torch.nn.Parameter(torch.tensor(1.),requires_grad=True)
        

    def forward(self,x1, x2, x3):
         
         
         self.r_a1 = x1 * self.t1 - self.b1
         self.r_asgm1 = torch.sigmoid(self.r_a1) # convert to 0-1 range
         self.A_sm1 =  F.softmax(self.A1, dim = 1)
         self.weightbias1 = 1-self.beta1 + torch.sum(self.A_sm1 * (self.r_asgm1), 1)
         self.activate1 = clamp(self.weightbias1).reshape([-1,1])
         
         
         self.r_a2 = x2 * self.t2 - self.b2
         self.r_asgm2 = torch.sigmoid(self.r_a2) # convert to 0-1 range
         self.A_sm2 =  F.softmax(self.A2, dim = 1)
         self.weightbias2 = 1-self.beta2 + torch.sum(self.A_sm2 * (self.r_asgm2), 1)
         self.activate2 = clamp(self.weightbias2).reshape([-1,1])
           
         
         self.r_a3 =  x3 * self.t3 - self.b3
         self.r_asgm3 = torch.sigmoid(self.r_a3) # convert to 0-1 range
         self.A_sm3 =  F.softmax(self.A3, dim = 1)
         self.weightbias3 = 1-self.beta3 + torch.sum(self.A_sm3 * (self.r_asgm3), 1)
         self.activate3 = clamp(self.weightbias3).reshape([-1,1])
           
           
         self.r_asgm4 = torch.cat((self.activate1, self.activate2, self.activate3),1)
         self.A_sm4 = F.softmax(self.A4, dim = 1)
         self.weightbias4 = 1 - self.beta4 + torch.sum(self.A_sm4 * (self.r_asgm4), 1)
         self.activate4 = clamp(self.weightbias4).reshape([-1])
         
         return self.activate4.reshape([-1])


    def Check_const(self, alpha = 0.5):
        dif1 = alpha * (1+self.A_sm1) - self.beta1
        dif2 = alpha * (1+self.A_sm2) - self.beta2
        dif3 = alpha * (1+self.A_sm3) - self.beta3
        dif4 = alpha * (1+self.A_sm4) - self.beta4
        
        As = [-np.sort(-self.A_sm1.cpu().detach().numpy().reshape([-1,1]), axis=0)]
        
        return [dif1.reshape((-1,1)).cpu().detach().numpy(), dif2.reshape((-1,1)).cpu().detach().numpy(), \
               dif3.reshape((-1,1)).cpu().detach().numpy(), dif4.reshape((-1,1)).cpu().detach().numpy()]


# Always one predicate
class TL_NN3(nn.Module):
    def __init__(self, T):
        super(TL_NN3,self).__init__()
        self.t1 = torch.nn.Parameter(1e-5*torch.randn(1,1), requires_grad=True)
        self.t1_2 = torch.nn.Parameter(1e-5*torch.randn(1,1), requires_grad=True)
        self.t2 = torch.nn.Parameter(1e-5*torch.randn(1,1), requires_grad=True)
        self.t2_2 = torch.nn.Parameter(1e-5*torch.randn(1,1), requires_grad=True)
        self.t3 = torch.nn.Parameter(1e-5*torch.randn(1,1), requires_grad=True)
        self.t3_2 = torch.nn.Parameter(1e-5*torch.randn(1,1), requires_grad=True)
        self.b1 = torch.nn.Parameter(torch.randn(1,1), requires_grad=True)
        self.b2 = torch.nn.Parameter(torch.randn(1,1), requires_grad=True)
        self.b3 = torch.nn.Parameter(torch.randn(1,1), requires_grad=True)
        self.A1 = torch.nn.Parameter(torch.rand(1,T),requires_grad=True)
        self.A2 = torch.nn.Parameter(torch.rand(1,T),requires_grad=True)
        self.A3 = torch.nn.Parameter(torch.rand(1,T),requires_grad=True)
        self.A4 = torch.nn.Parameter(torch.rand(1,3),requires_grad=True)
        self.beta1 = torch.nn.Parameter(torch.tensor(1.),requires_grad=True)
        self.beta2 = torch.nn.Parameter(torch.tensor(1.),requires_grad=True)
        self.beta3 = torch.nn.Parameter(torch.tensor(1.),requires_grad=True)
        self.beta4 = torch.nn.Parameter(torch.tensor(1.),requires_grad=True)

    def forward(self,x1, x2, x3):
         
        
        self.r_a1 = x1 * self.t1 - self.b1
        self.r_asgm1 = torch.sigmoid(self.r_a1) # convert to 0-1 range
        self.A_sm1 =  F.softmax(self.A1, dim = 1)
        self.weightbias1 = self.beta1 - torch.sum(self.A_sm1 * (1 - self.r_asgm1), 1)
        self.activate1 = clamp(self.weightbias1).reshape([-1,1])
                 
        
        self.r_a2 = x2 * self.t2 - self.b2
        self.r_asgm2 = torch.sigmoid(self.r_a2) # convert to 0-1 range
        self.A_sm2 =  F.softmax(self.A2, dim = 1)
        self.weightbias2 = self.beta2 - torch.sum(self.A_sm2 * (1 - self.r_asgm2), 1)
        self.activate2 = clamp(self.weightbias2).reshape([-1,1])
           
        
        self.r_a3 =  x3 * self.t3 - self.b3
        self.r_asgm3 = torch.sigmoid(self.r_a3) # convert to 0-1 range
        self.A_sm3 =  F.softmax(self.A3, dim = 1)
        self.weightbias3 = self.beta3 - torch.sum(self.A_sm3 * (1 - self.r_asgm3), 1)
        self.activate3 = clamp(self.weightbias3).reshape([-1,1])
           
           
        self.r_asgm4 = torch.cat((self.activate1, self.activate2, self.activate3),1)
        self.A_sm4 = F.softmax(self.A4, dim = 1)
        self.weightbias4 = self.beta4 - torch.sum(self.A_sm4 * (1 - self.r_asgm4), 1)
        self.activate4 = clamp(self.weightbias4).reshape([-1])
        
        return self.activate4.reshape([-1])




# Eventually one predicate
class TL_NN4(nn.Module):
    def __init__(self, T):
        super(TL_NN4,self).__init__()
        self.t1 = torch.nn.Parameter(1e-5*torch.randn(1,1), requires_grad=True)
        self.t1_2 = torch.nn.Parameter(1e-5*torch.randn(1,1), requires_grad=True)
        self.t1_3 = torch.nn.Parameter(1e-5*torch.randn(1,1), requires_grad=True)
        self.t2 = torch.nn.Parameter(1e-5*torch.randn(1,1), requires_grad=True)
        self.t2_2 = torch.nn.Parameter(1e-5*torch.randn(1,1), requires_grad=True)
        self.t2_3 = torch.nn.Parameter(1e-5*torch.randn(1,1), requires_grad=True)
        self.t3 = torch.nn.Parameter(1e-5*torch.randn(1,1), requires_grad=True)
        self.t3_2 = torch.nn.Parameter(1e-5*torch.randn(1,1), requires_grad=True)
        self.t3_3 = torch.nn.Parameter(1e-5*torch.randn(1,1), requires_grad=True)
        self.b1 = torch.nn.Parameter(torch.randn(1,1), requires_grad=True)
        self.b2 = torch.nn.Parameter(torch.randn(1,1), requires_grad=True)
        self.b3 = torch.nn.Parameter(torch.randn(1,1), requires_grad=True)
        self.A1 = torch.nn.Parameter(torch.rand(1,T),requires_grad=True)
        self.A2 = torch.nn.Parameter(torch.rand(1,T),requires_grad=True)
        self.A3 = torch.nn.Parameter(torch.rand(1,T),requires_grad=True)
        self.A4 = torch.nn.Parameter(torch.rand(1,3),requires_grad=True)
        self.beta1 = torch.nn.Parameter(torch.tensor(1.),requires_grad=True)
        self.beta2 = torch.nn.Parameter(torch.tensor(1.),requires_grad=True)
        self.beta3 = torch.nn.Parameter(torch.tensor(1.),requires_grad=True)
        self.beta4 = torch.nn.Parameter(torch.tensor(1.),requires_grad=True)

        
    def forward(self,x1, x2, x3):
        
        self.r_a1 =  x1 * self.t1 - self.b1
        self.r_asgm1 = torch.sigmoid(self.r_a1) # convert to 0-1 range
        self.A_sm1 =  F.softmax(self.A1, dim = 1)
        self.weightbias1 = 1-self.beta1 + torch.sum(self.A_sm1 * (self.r_asgm1), 1)
        self.activate1 = clamp(self.weightbias1).reshape([-1,1])
                 
        
        self.r_a2 =  x2 * self.t2 - self.b2
        self.r_asgm2 = torch.sigmoid(self.r_a2) # convert to 0-1 range
        self.A_sm2 =  F.softmax(self.A2, dim = 1)
        self.weightbias2 = 1-self.beta2 + torch.sum(self.A_sm2 * (self.r_asgm2), 1)
        self.activate2 = clamp(self.weightbias2).reshape([-1,1])
           
        
        self.r_a3 = x3 * self.t3 - self.b3
        self.r_asgm3 = torch.sigmoid(self.r_a3) # convert to 0-1 range
        self.A_sm3 =  F.softmax(self.A3, dim = 1)
        self.weightbias3 = 1-self.beta3 + torch.sum(self.A_sm3 * (self.r_asgm3), 1)
        self.activate3 = clamp(self.weightbias3).reshape([-1,1])
           
        self.r_asgm4 = torch.cat((self.activate1, self.activate2, self.activate3),1)
        self.A_sm4 = F.softmax(self.A4, dim = 1)
        self.weightbias4 = 1-self.beta4 + torch.sum(self.A_sm4 * (self.r_asgm4), 1)
        self.activate4 = clamp(self.weightbias4).reshape([-1])
                 
        return self.activate4.reshape([-1])
     
     

# always eventually one predicate
class TL_NN5(nn.Module):
    def __init__(self, T):
        super(TL_NN5,self).__init__()
        self.t1 = torch.nn.Parameter(1e-5*torch.randn(1,1), requires_grad=True)
        self.b1 = torch.nn.Parameter(1e-5*torch.randn(1,1), requires_grad=True)
        self.A1_1 = torch.nn.Parameter(torch.rand(1,1,T),requires_grad=True)
        self.beta1_1 = torch.nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.beta1_2 = torch.nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.A2_1 = torch.nn.Parameter(torch.rand((1,T),requires_grad=True))
        self.T = T
        
        self.t2 = torch.nn.Parameter(1e-5*torch.randn(1,1), requires_grad=True)
        self.b2 = torch.nn.Parameter(1e-5*torch.randn(1,1), requires_grad=True)
        self.A1_2 = torch.nn.Parameter(torch.rand(1,1,T),requires_grad=True)
        self.A2_2 = torch.nn.Parameter(torch.rand((1,T),requires_grad=True))
        self.beta2_1 = torch.nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.beta2_2 = torch.nn.Parameter(torch.tensor(1.), requires_grad=True)
        
        self.t3 = torch.nn.Parameter(1e-5*torch.randn(1,1), requires_grad=True)
        self.b3 = torch.nn.Parameter(1e-5*torch.randn(1,1), requires_grad=True)
        self.A1_3 = torch.nn.Parameter(torch.rand(1,1,T),requires_grad=True)
        self.A2_3 = torch.nn.Parameter(torch.rand((1,T),requires_grad=True))
        self.beta3_1 = torch.nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.beta3_2 = torch.nn.Parameter(torch.tensor(1.), requires_grad=True)
        
        self.A4 = torch.nn.Parameter(torch.rand((1,3),requires_grad=True))
        self.beta4 = torch.nn.Parameter(torch.tensor(1.), requires_grad=True)

    def forward(self,x1, x2, x3):
        # first layer for eventually
         x1 = Preprocess(x1, self.T, self.T)
         r_a1_1 =  (self.b1 - x1 * self.t1)
         b1pos = torch.where(r_a1_1 == self.b1)
         r_a1_1[r_a1_1 == self.b1] = -10
         self.r_a1_1 = r_a1_1
         self.r_asgm1_1 = torch.sigmoid(self.r_a1_1) # convert to 0-1 range
         self.A1_1sm =  F.softmax(self.A1_1, dim = 2)         
         self.weightbias1_1 = 1-self.beta1_1 + torch.sum(self.A1_1sm * (self.r_asgm1_1), 2)
         self.activate1_1 = torch.max(torch.zeros(1).to(device), \
                                      torch.min(torch.ones(1).to(device),self.weightbias1_1)).reshape([-1,self.T])         
         # second layer for always
         self.ra2_1 = self.activate1_1
         self.A2_1sm =  F.softmax(self.A2_1, dim = 1)        
         self.weightbias2_1 = self.beta1_2 - torch.sum(self.A2_1sm * (1-self.ra2_1), 1)
         self.xrtn2_1 = torch.max(torch.zeros_like(self.weightbias2_1), \
                        torch.min(torch.ones_like(self.weightbias2_1),self.weightbias2_1)).reshape([-1,1])
         
         
         x2 = Preprocess(x2, self.T, self.T)
         r_a1_2 =  (self.b2 - x2 * self.t2)
         b2pos = torch.where(r_a1_2 == self.b2)
         r_a1_2[b2pos] = -10
         self.r_a1_2 = r_a1_2
         self.r_asgm1_2 = torch.sigmoid(self.r_a1_2) # convert to 0-1 range
         self.A1_2sm =  F.softmax(self.A1_2, dim = 2)         
         self.weightbias1_2 = 1-self.beta2_1 + torch.sum(self.A1_2sm * (self.r_asgm1_2), 2)
         self.activate1_2 = torch.max(torch.zeros_like(self.weightbias1_2), \
                                      torch.min(torch.ones_like(self.weightbias1_2),self.weightbias1_2)).reshape([-1,self.T])         
         # second layer for always
         self.ra2_2 = self.activate1_2
         self.A2_2sm =  F.softmax(self.A2_2, dim = 1)        
         self.weightbias2_2 = self.beta2_2 - torch.sum(self.A2_2sm * (1-self.ra2_2), 1)
         self.xrtn2_2 = torch.max(torch.zeros_like(self.weightbias2_2), \
                        torch.min(torch.ones_like(self.weightbias2_2),self.weightbias2_2)).reshape([-1,1])
         
         
         x3 = Preprocess(x3, self.T, self.T)
         r_a1_3 =  (self.b3 - x3 * self.t3)
         b3pos = torch.where(r_a1_3 == self.b3)
         r_a1_3[b3pos] = -10
         self.r_a1_3 = r_a1_3
         self.r_asgm1_3 = torch.sigmoid(self.r_a1_3) # convert to 0-1 range
         self.A1_3sm =  F.softmax(self.A1_3, dim = 2)         
         self.weightbias1_3 = 1-self.beta3_1 + torch.sum(self.A1_3sm * (self.r_asgm1_3), 2)
         self.activate1_3 = torch.max(torch.zeros_like(self.weightbias1_3), \
                            torch.min(torch.ones_like(self.weightbias1_3),self.weightbias1_3)).reshape([-1,self.T])         
         # second layer for always
         self.ra2_3 = self.activate1_3
         self.A2_3sm =  F.softmax(self.A2_3, dim = 1)        
         self.weightbias2_3 = self.beta3_2 - torch.sum(self.A2_3sm * (1-self.ra2_3), 1)
         self.xrtn2_3 = torch.max(torch.zeros_like(self.weightbias2_3), \
                        torch.min(torch.ones_like(self.weightbias2_3),self.weightbias2_3)).reshape([-1,1])
     
         self.xrtn123 = torch.cat((self.xrtn2_1, self.xrtn2_2, self.xrtn2_3),1)
         self.A4_sm =  F.softmax(self.A4, dim = 1)        
         self.weightbias4 = self.beta4 - torch.sum(self.A4_sm * (1-self.xrtn123), 1)
         self.xrtn4 = torch.max(torch.zeros_like(self.weightbias4), \
                                torch.min(torch.ones_like(self.weightbias4),self.weightbias4)).reshape([-1])
                  
         return self.xrtn4



# eventually always one predicate
class TL_NN6(nn.Module):
    def __init__(self, T):
        super(TL_NN6,self).__init__()
        self.t1 = torch.nn.Parameter(1e-5*torch.randn(1,1), requires_grad=True)
        self.b1 = torch.nn.Parameter(1e-5*torch.randn(1,1), requires_grad=True)
        self.A1_1 = torch.nn.Parameter(torch.rand(1,1,T),requires_grad=True)
        self.beta1_1 = torch.nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.beta1_2 = torch.nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.A2_1 = torch.nn.Parameter(torch.rand((1,T),requires_grad=True))
        self.T = T
        
        self.t2 = torch.nn.Parameter(1e-5*torch.randn(1,1), requires_grad=True)
        self.b2 = torch.nn.Parameter(1e-5*torch.randn(1,1), requires_grad=True)
        self.A1_2 = torch.nn.Parameter(torch.rand(1,1,T),requires_grad=True)
        self.A2_2 = torch.nn.Parameter(torch.rand((1,T),requires_grad=True))
        self.beta2_1 = torch.nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.beta2_2 = torch.nn.Parameter(torch.tensor(1.), requires_grad=True)
        
        self.t3 = torch.nn.Parameter(1e-5*torch.randn(1,1), requires_grad=True)
        self.b3 = torch.nn.Parameter(1e-5*torch.randn(1,1), requires_grad=True)
        self.A1_3 = torch.nn.Parameter(torch.rand(1,1,T),requires_grad=True)
        self.A2_3 = torch.nn.Parameter(torch.rand((1,T),requires_grad=True))
        self.beta3_1 = torch.nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.beta3_2 = torch.nn.Parameter(torch.tensor(1.), requires_grad=True)
        
        self.A4 = torch.nn.Parameter(torch.rand((1,3),requires_grad=True))
        self.beta4 = torch.nn.Parameter(torch.tensor(1.), requires_grad=True)
        
    def forward(self,x1, x2, x3):
        # first layer for always
         x1 = Preprocess(x1, self.T, self.T)
         r_a1_1 =  (self.b1 - x1 * self.t1)
         b1pos = torch.where(r_a1_1 == self.b1)
         r_a1_1[b1pos] = 10
         self.r_a1_1 = r_a1_1
         self.r_asgm1_1 = torch.sigmoid(self.r_a1_1) # convert to 0-1 range
         self.A1_1sm =  F.softmax(self.A1_1, dim = 2)         
         self.weightbias1_1 = self.beta1_1 - torch.sum(self.A1_1sm * (1-self.r_asgm1_1), 2)
         self.activate1_1 = torch.max(torch.zeros_like(self.weightbias1_1), \
                        torch.min(torch.ones_like(self.weightbias1_1),self.weightbias1_1)).reshape([-1,self.T])         
         # second layer for eventually
         self.ra2_1 = self.activate1_1
         self.A2_1sm =  F.softmax(self.A2_1, dim = 1)        
         self.weightbias2_1 = 1 - self.beta2_1 + torch.sum(self.A2_1sm * (self.ra2_1), 1)
         self.xrtn2_1 = torch.max(torch.zeros_like(self.weightbias2_1), \
                                  torch.min(torch.ones_like(self.weightbias2_1),self.weightbias2_1)).reshape([-1, 1])
         
        
         x2 = Preprocess(x2, self.T, self.T)
         r_a1_2 =  (self.b2 - x2 * self.t2)
         b2pos = torch.where(r_a1_2 == self.b2)
         r_a1_2[b2pos] = 10
         self.r_a1_2 = r_a1_2
         self.r_asgm1_2 = torch.sigmoid(self.r_a1_2) # convert to 0-1 range
         self.A1_2sm =  F.softmax(self.A1_2, dim = 2)         
         self.weightbias1_2 = self.beta2_1 - torch.sum(self.A1_2sm * (1-self.r_asgm1_2), 2)
         self.activate1_2 = torch.max(torch.zeros_like(self.weightbias1_2), \
                        torch.min(torch.ones_like(self.weightbias1_2),self.weightbias1_2)).reshape([-1,self.T])         
         # second layer for eventually
         self.ra2_2 = self.activate1_2
         self.A2_2sm =  F.softmax(self.A2_2, dim = 1)        
         self.weightbias2_2 = 1 - self.beta2_2 + torch.sum(self.A2_2sm * (self.ra2_2), 1)
         self.xrtn2_2 = torch.max(torch.zeros_like(self.weightbias2_2), \
                        torch.min(torch.ones_like(self.weightbias2_2),self.weightbias2_2)).reshape([-1,1])
        
        
         x3 = Preprocess(x3, self.T, self.T)
         r_a1_3 =  (self.b3 - x3 * self.t3)
         b3pos = torch.where(r_a1_3 == self.b3)
         r_a1_3[b3pos] = 10
         self.r_a1_3 = r_a1_3
         self.r_asgm1_3 = torch.sigmoid(self.r_a1_3) # convert to 0-1 range
         self.A1_3sm =  F.softmax(self.A1_3, dim = 2)         
         self.weightbias1_3 = self.beta3_1 - torch.sum(self.A1_3sm * (1-self.r_asgm1_3), 2)
         self.activate1_3 = torch.max(torch.zeros_like(self.weightbias1_3), \
                            torch.min(torch.ones_like(self.weightbias1_3),self.weightbias1_3)).reshape([-1,self.T])         
         # second layer for always
         self.ra2_3 = self.activate1_3
         self.A2_3sm =  F.softmax(self.A2_3, dim = 1)        
         self.weightbias2_3 = 1 - self.beta3_2 + torch.sum(self.A2_3sm * (self.ra2_3), 1)
         self.xrtn2_3 = torch.max(torch.zeros_like(self.weightbias2_3), \
                        torch.min(torch.ones_like(self.weightbias2_3),self.weightbias2_3)).reshape([-1,1])
        
         self.xrtn123 = torch.cat((self.xrtn2_1, self.xrtn2_2, self.xrtn2_3),1)
         self.A4_sm =  F.softmax(self.A4, dim = 1)        
         self.weightbias4 = self.beta4 - torch.sum(self.A4_sm * (1-self.xrtn123), 1)
         self.xrtn4 = torch.max(torch.zeros_like(self.weightbias4), \
                        torch.min(torch.ones_like(self.weightbias4),self.weightbias4)).reshape([-1])
                  
         return self.xrtn4


def Preprocess(x, T1, T2):
    xnew = torch.zeros([x.shape[0], T1, T2]).to(device)
    for i in range(T1):
        xnew[:,i,:] = torch.cat((x[:, i:], torch.zeros([x.shape[0],i]).to(device)),1)
    return xnew

def clamp(x):
    return torch.max(torch.zeros_like(x), torch.min(torch.ones_like(x),x))

