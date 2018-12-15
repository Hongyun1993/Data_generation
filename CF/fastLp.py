import cv2
import numpy as np


def fastLp(img,p,deltaInv,mu,radSize,D,nambda,D2,gamma):
    '''
    input: img:[row,col,3], p:[row,col], deltaInv:[row,col,3,3]
    mu:[row,col], radSize:int
    output: q:[row,col]
    valueInterval:
    img:[0,1], p:[0,1], mu:[0,1], radSize:[1,inf]
    '''
    row,col = np.shape(img)[:2]
    kSize = radSize*2 + 1
    imgPad = np.pad(img,((radSize,radSize),(radSize,radSize),(0,0)),'constant')
    pPad = np.pad(p,((radSize,radSize),(radSize,radSize)),'constant')
    pMu = cv2.blur(pPad,(kSize,kSize))
    pMu = pMu[radSize:radSize+row,radSize:radSize+col]

    Ip = imgPad*np.tile(pPad[:,:,np.newaxis],(1,1,3))
    IpSum = cv2.blur(Ip,(kSize,kSize))*(kSize*kSize)
    IpSum = IpSum[radSize:radSize+row,radSize:radSize+col,:]

    muP = mu*np.tile(pMu[:,:,np.newaxis],(1,1,3))

    media = IpSum/(kSize*kSize) - muP
    a = np.einsum('...ij,...j->...i',deltaInv,media)
    b = pMu - np.sum(a*mu,axis = 2)
    a = np.pad(a,((radSize,radSize),(radSize,radSize),(0,0)),'constant')
    b = np.pad(b,((radSize,radSize),(radSize,radSize)),'constant')
    aSum = cv2.blur(a,(kSize,kSize))*((kSize)*(kSize))
    bSum = cv2.blur(b,(kSize,kSize))*((kSize)*(kSize))
    aSum = aSum[radSize:radSize+row,radSize:radSize+col]
    bSum = bSum[radSize:radSize+row,radSize:radSize+col]
    Lp = (kSize*kSize)*p - np.sum(aSum*img,axis = 2) - bSum
    q = Lp + (nambda * D + gamma*D2) * p
    return q
