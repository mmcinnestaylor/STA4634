import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

def draw(x):
    plt.scatter(x[:,0], x[:,1])

def kMeans(x, k=2):
    d=2
    miu=np.zeros((k, d))
    sigma=np.zeros((k, d,d))
    sigma[:]=np.identity(d)
    y=np.zeros(x.shape[0])
    miu[0]=np.mean(x[0::2, :], axis=0)
    miu[1]=np.mean(x[1::2, :], axis=0)
    miuOld=np.copy(miu)
    sigmaOld=np.copy(sigma)
    eps=1.0e-8
    pi=np.zeros(2)
    for i in range(100):
        sigmaInv=inv(sigma)
        dx=x-miu[0]
        temp0=np.sum(dx.dot(sigmaInv[0])*dx, axis=1)
        dx=x-miu[1]
        temp1=np.sum(dx.dot(sigmaInv[1])*dx, axis=1)
        temp=(temp1<temp0)
        y=temp.astype(int)
        miu[0]=np.mean(x[y==0], axis=0)
        miu[1]=np.mean(x[y==1], axis=0)
        #print(i, miu)

        dx=x[y==0]-miu[0]
        sigma[0]=dx.T.dot(dx)/(dx.shape[0]-1)
        dx=x[y==1]-miu[1]
        sigma[1]=dx.T.dot(dx)/(dx.shape[0]-1)

        dmiu=np.sum(np.absolute(miu-miuOld))
        dsigma=np.sum(np.absolute(sigma-sigmaOld))
        print(i, dmiu, dsigma)
        if dmiu<=eps and dsigma<=eps:
            print('dmiu and dsigma<', eps,', done!')
            break
        miuOld=np.copy(miu)
        sigmaOld=np.copy(sigma)
        pi[1]=np.mean(y)
        pi[0]=1.-pi[1]
    
    print('pi', pi)
    print('miu: ', miu)
    print('sigma: ', sigma)
    #print('y', y)
    return pi, miu, sigma

def EM_only(x, k=2):
    pi0, miu0, sigma0=kMeans(x, k)
    print('finish initialization with k-means')
    d=2
    pi=np.copy(pi0)
    miu=np.copy(miu0)
    sigma=np.copy(sigma0)
    y=np.zeros(x.shape[0])
    miuOld=np.copy(miu)
    sigmaOld=np.copy(sigma)
    eps=1.0e-6
    for i in range(500):
        #expectation
        sigmaInv=inv(sigma)
        dx=x-miu[0]
        temp0=(np.exp(-0.5*np.sum(dx.dot(sigmaInv[0])*dx, axis=1))/
                (2.*np.pi*np.sqrt(np.linalg.det(sigma[0]))))
        dx=x-miu[1]
        temp1=(np.exp(-0.5*np.sum(dx.dot(sigmaInv[1])*dx, axis=1))/
                (2.*np.pi*np.sqrt(np.linalg.det(sigma[1]))))

        temp0*=pi[0]
        temp1*=pi[1]
        y=temp1/(temp0+temp1)

        # maximization
        pi[0]=1.-np.mean(y)
        pi[1]=np.mean(y)
        miu[0]=np.sum( (x.T*(1-y)).T, axis=0 )/np.sum(1.-y)
        miu[1]=np.sum( (x.T*y).T, axis=0 )/np.sum(y)
        print(i, miu)

        dx=x-miu[0]
        sigma[0]=(dx.T*(1-y)).dot(dx)/(np.sum(1.-y))
        dx=x-miu[1]
        sigma[1]=(dx.T*y).dot(dx)/(np.sum(y))

        dmiu=np.sum(np.absolute(miu-miuOld))
        dsigma=np.sum(np.absolute(sigma-sigmaOld))
        print(i, dmiu, dsigma)
        if dmiu<=eps and dsigma<=eps:
            print('dmiu and dsigma<', eps,', done!')
            break
        miuOld=np.copy(miu)
        sigmaOld=np.copy(sigma)

    print('pi:',pi)
    print('miu:',miu)
    print('sigma:',sigma)

def EM(miu0, sigma0, weight0, x, eps=1.0e-6):
    print('miu0', miu0)
    print('sigma0', sigma0)
    print('weight0', weight0)
    n=x[0].shape[0]
    l=miu0.shape[0]
    m=x.shape[0]
    p=np.zeros((m,l))

    miu=np.copy(miu0)
    sigma=np.copy(sigma0)
    weight=np.copy(weight0)

    miuOld=np.copy(miu0)
    sigmaOld=np.copy(sigma)
    weightOld=np.copy(weight)

    for i in range(50):
        #expectation
        print('i, weight', i, weight)
        for j in range(l):
            dx=x-miu[j]
            #in case divide by 0
            if sigma[j]==0.:
                sigma[j]+=0.01*eps
            p[:,j]=(np.exp(-0.5/sigma[j]/sigma[j]*np.sum(dx*dx, axis=1))/
                    (2.*np.pi*sigma[j]*sigma[j]))
        #print('p', p[:10])
        p=p*weight
        #print('weight x p:',  p[:10])
        temp=np.sum(p, axis=1)
        temp[temp==0.]+=0.01*eps
        p=(p.T/temp).T
        #print('weight x p after normalization:', p[:10])

        # maximization
        weight=np.sum(p, axis=0)/m
        for j in range(l):
            #in case divide by 0
            if weight[j]==0.:
                #miu[j]=np.sum(x, axis=0)/m
                weight[j]+=0.01*eps
            #else:
            miu[j]=np.sum((x.T*p[:,j]).T,axis=0)/m/weight[j]
        for j in range(l):
            dx=x-miu[j]
            dx=np.sum(dx*dx, axis=1)
            sigma[j]=np.sqrt(np.sum(dx*p[:,j], axis=0)/m/n)
            #print('j, sigma:', j, sigma[j])
        print('EM:', i, miu)

        dmiu=np.sum((miu-miuOld)*(miu-miuOld))
        dsigma=np.sum((sigma-sigmaOld)*(sigma-sigmaOld))
        dweight=np.sum((weight-weightOld)*(weight-weightOld))
        print(i, dmiu, dsigma, dweight)
        if dmiu<=eps and dsigma<=eps and dweight<=eps:
            print('dmiu, dsigma, and dweight<', eps,', done!')
            break
        miuOld=np.copy(miu)
        sigmaOld=np.copy(sigma)
        weightOld=np.copy(weight)

    return miu, sigma, weight

def provableEM(x):
    l=5
    k=2
    n=x[0].shape[0]
    m=x.shape[0]
    eps=1.0e-6

    #initialization
    miu0=np.zeros((l,2))
    a=np.arange(m)
    np.random.shuffle(a)
    index=a[:l]
    # don't use the follow one, which might generate repeate sampels
    # and lead a sigma to be 0
    #index=np.random.randint(m, size=l)
    print('initial index', index)
    miu0=x[index]
    weight0=np.ones(l)/l
    sigma0=np.sqrt(0.5/n)*np.ones(l)
    for i in range(l):
        dmiu=miu0-miu0[i]
        mask=np.ones(l, dtype=bool)
        mask[i]=False
        dmiu=dmiu[mask]
        dmiu=np.sum(dmiu*dmiu, axis=1)
        sigma0[i]*=np.sqrt(np.min(dmiu))

    #EM
    miu, sigma, weight=EM(miu0, sigma0, weight0, x)
    print('1st round EM done!')

    #Pruning
    #sometimes it can have only 1 hot weight, can't satisfy
    #mask=(weight>=(0.5/l+2./m))
    mask=np.argsort(weight)
    mask=mask[-2:]
    miu=miu[mask]
    sigma=sigma[mask]

    l=miu.shape[0]
    i0=np.random.randint(l)
    # since k=2,
    # we only have 1 center left for choosing here
    dmiu=miu-miu[i0]
    dmiu=np.sqrt(np.sum(dmiu*dmiu, axis=1))
    sigma[sigma==0.]+=0.01*eps
    dmiu/=(sigma+sigma[i0])
    i1=np.argmax(dmiu)
    index=np.array([i0, i1])

    miu=miu[index]
    weight=np.array([0.5, 0.5])
    sigma0=sigma0[mask]
    sigma=sigma0[index]
    print('Pruning done')

    #EM
    miu2, sigma2, weight2=EM(miu, sigma, weight, x)
    print('2nd round EM done!')

    print('miu:',miu2)
    print('sigma:',sigma2)
    print('weight:',weight2)

#x=np.genfromtxt('../data/xeasy.txt', delimiter=',')
x=np.genfromtxt('../data/x1.txt', delimiter=',')
#x=np.genfromtxt('../data/x2.txt', delimiter=',')
#draw(x)
#pi_, miu_, sigma_=kMeans(x) 
EM_only(x)
#provableEM(x)
plt.show()
