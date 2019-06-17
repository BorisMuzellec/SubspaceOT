#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 13:33:51 2019

@author: boris
"""

import numpy as np

def sqrtm(A):
    vecs, vals, _ = np.linalg.svd(A)
    return vecs.dot(np.sqrt(vals[:, np.newaxis]) * vecs.T)

def bures(A,B):
    sA = sqrtm(A)
    return np.trace(A + B - 2 * sqrtm(sA.dot(B).dot(sA)))


#### Monge ####

def monge(A, B):

    sA = sqrtm(A)
    sA_inv = np.linalg.inv(sA)
  
    return sA_inv.dot(sqrtm(sA.dot(B).dot(sA))).dot(sA_inv)


def Vpi(A, B):
  
    sA = sqrtm(A)
    sA_inv = np.linalg.inv(sA)
    mid = sqrtm(sA.dot(B).dot(sA))
  
    T = sA_inv.dot(mid).dot(sA_inv)
    
    return A + B - (T.dot(A) + A.dot(T))

def fidelity(A, B):
    sA = sqrtm(A)
    return  np.trace(sqrtm(sA.dot(B).dot(sA)))


#### Monge-Knothe ####

def MK(A, B, k=2):
    d = A.shape[0]
  
    Ae = A[:k, :k]
    Aeet = A[:k, k:]
    Aet = A[k:, k:]
  
    schurA = Aet - Aeet.T.dot(np.linalg.inv(Ae)).dot(Aeet)
  
    Be = B[:k, :k]
    Beet = B[:k, k:]
    Bet = B[k:, k:]
  
    schurB = Bet - Beet.T.dot(np.linalg.inv(Be)).dot(Beet)
  
    Tee = monge(Ae, Be)
    Tschur = monge(schurA, schurB)
 
    return (np.hstack([np.vstack([Tee, (Beet.T.dot(np.linalg.inv(Tee)) - Tschur.dot(Aeet.T)).dot(np.linalg.inv(Ae))]), np.vstack([np.zeros((k, d-k)), Tschur])]))

def MK_dist(A, B, k = 2):
    T = MK(A, B, k)
    return np.trace(A + B - (T.dot(A) + A.dot(T.T)))

def MK_fidelity(A, B, k = 2):
    T = MK(A, B, k)
    return np.trace(T.dot(A) + A.dot(T.T)) / 2.



#### Monge-Independent ####

def Vpi_MI(A, B, k = 2):
    d = A.shape[0]
  
    Ae = A[:k, :k]
    Aeet = A[:k, k:]
  
    Be = B[:k, :k]
    Beet = B[:k, k:]
  
    I = np.eye(d)
    Ve = I[:, :k]
    Vet = I[:, k:]
  
    sAe = sqrtm(Ae)
    sAe_inv = np.linalg.inv(sAe)
    Te = sAe_inv.dot(sqrtm(sAe.dot(Be).dot(sAe))).dot(sAe_inv)
  
    C1 = Ve.dot(Ae).dot(Te).dot(Ve.T + (np.linalg.inv(Be)).dot(Beet).dot(Vet.T))
    C2 = Vet.dot(Aeet.T).dot(Te).dot(Ve.T + (np.linalg.inv(Be)).dot(Beet).dot(Vet.T))
  
    C = C1 + C2
  
    return A + B - (C + C.T) 

def MI_dist(A, B, k = 2):
    return np.trace(Vpi_MI(A, B, k))

def MI_fidelity(A, B, k = 2):
    return np.trace((A + B) - Vpi_MI(A, B, k)) / 2.



### Knothe-Rosenblatt ###

def KR_dist(A, B):
  
    La = np.linalg.cholesky(A)
    Lb = np.linalg.cholesky(B)
  
    return ((La - Lb)**2).sum()
