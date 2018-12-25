import cv2
import numpy as np
from fastLp import fastLp
def fastMatting(img,tri,tri2):
    '''
    input: img:[row,col,3], tri:[row,col]
    output: alpha:[row,col]
    value interval: [0,255]
    '''
    epsilon = 10**(-7)
    nambda = 100
    gamma = 50
    radSize = 3
    kSize = 2*radSize + 1
    img = img/255.0
    tri = tri/255.0
    tri2 = tri2/255.0
    row,col = np.shape(tri)
    D = (tri < 0.01) + (tri > 0.99)
    D = D.astype(np.uint8)
    D2 = (tri2 < 0.01) + (tri2 > 0.99)
    D2 = D2.astype(np.uint8)
    mu = cv2.blur(img,(kSize,kSize))
    imgPad = np.pad(img,((radSize,radSize),(radSize,radSize),(0,0)),'constant')
    deltaInv = np.zeros((row,col,3,3))
    for i in range(row):
        for ii in range(col):
            meanMu = mu[i,ii]
            meanMu = np.reshape(meanMu,(1,1,3))
            meanMu = np.tile(meanMu,(kSize,kSize,1))
            imgPart = imgPad[i:i+kSize,ii:ii+kSize,:]
            imgPart = imgPart - meanMu
            imgPart = imgPart.reshape((kSize*kSize,3))
            sigma = np.dot(imgPart.T,imgPart)/(kSize*kSize) + (epsilon/(kSize*kSize))*np.eye(3)
            deltaInv[i,ii] = np.linalg.pinv(sigma)
    alpha = np.random.rand(row,col)
    q = fastLp(img,alpha,deltaInv,mu,radSize,D,nambda,D2,gamma)
    b = nambda*D*tri + gamma*D2*tri2
    r = b - q
    p = r
    iterMax = 300
    for o in range(iterMax):
        q = fastLp(img,p,deltaInv,mu,radSize,D,nambda,D2,gamma)
        a = np.sum(r*r)/np.sum(p*q)
        alpha = alpha + a*p
        r_ = r - a*q
        beta = np.sum(r_*r_)/np.sum(r*r)
        p = r_ + beta*p
        r = r_.copy()
        rNorm = np.sqrt(np.sum(r*r))
        if rNorm < 10**(-3):
            break
    #    cv2.imshow('alpha',alpha)
    #    cv2.waitKey(10)
        print('-'*20+str(iterMax-o)+'-'*20)
        print('res:',rNorm)
        print('alpha_max:',np.max(alpha))
    return alpha
