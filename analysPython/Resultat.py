import numpy as np
import matplotlib.pyplot as plt
from funktioner import  getSandSigma
from scipy.special import erf

#Gör saker för fina grafer
plt.rcParams['text.usetex'] = True
matlab_cmap = [
    [0, 0.4470, 0.7410],  # Blue
    [0.8500, 0.3250, 0.0980],  # Orange
    [0.9290, 0.6940, 0.1250],  # Yellow
    [0.4940, 0.1840, 0.5560],  # Purple
    [0.4660, 0.6740, 0.1880],  # Green
    [0.3010, 0.7450, 0.9330],  # Light blue
    [0.6350, 0.0780, 0.1840],  # Dark red
]

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=matlab_cmap)


#ifall man vill plotta alla osäkerheter på samma gång
testing = False

#läs in alla mätvärden
measAlts = ["manuell1.txt","manuell2.txt","auto1.txt","auto2.txt"]

#sigma från uppställningsosäkerheter, se rapport
sigmaPolVec = np.array([0.0855, 0.0855, 0.0257, 0.0257])

#Få S och sigma från mätdata för varje mätserie
measurements = np.array([getSandSigma(measAlt) for measAlt in measAlts])
SVec = measurements[:,0]
sigmaNVec = measurements[:,1]
sigmaTotVec = np.sqrt(sigmaPolVec**2 + sigmaNVec**2)


#Intervall att plotta över
numPoints = int(1e5)
start = 1.9
end_ = 2.3
x = np.linspace(start,end_,numPoints)

#Gör gaussanpassning för varje mätning, använder bara första och sista senare
Gaussian = lambda mu, sigma: np.exp( -1/2*( (x - mu)/sigma)**2 )/(np.sqrt(2*np.pi)*sigma)

BellVec = np.array([Gaussian(S,sigma) for S,sigma in zip(SVec,sigmaNVec)])
BellPolVec = np.array([Gaussian(S,sigma) for S,sigma in zip(SVec,sigmaPolVec)])
BellTotVec = np.array([Gaussian(S,sigma) for S,sigma in zip(SVec,sigmaTotVec)])


#Hitta hur många punkter en standardavvikelse är 
#Gör bara för den bra S-mätningen
stepsPerSigma = int(np.round(numPoints*sigmaTotVec[-1]/(end_ - start)))

Sind = 3
numStds = 4
print(x[np.argmax(BellTotVec[Sind])])

startIndex = np.argmax(BellTotVec[Sind]) - numStds*stepsPerSigma
endIndex = np.argmax(BellTotVec[Sind])
devYvals = BellTotVec[-1,endIndex:startIndex:-stepsPerSigma]
devXvals = x[endIndex:startIndex:-stepsPerSigma]

#Räkna ut sannolikheten att S < 2        
def getProbs(S,sigma):
    Phi = lambda x : 1/2*(erf(x/np.sqrt(2)) )
    # P(S<2) = 1/2 - Phi()
    numStds = (S - 2)/sigma
    return 1/2 - Phi(numStds) 

Probs = [getProbs(S,sigma) for S,sigma in zip(SVec,sigmaTotVec)]




#Plotta saker och printa resultat

names = ['manuell 1','manuell 2','auto 1', 'auto 2']
print(*[f'\n {names[i]} \n S: {SVec[i]}\t sigma: {sigmaTotVec[i]} P(s<2): {Probs[i]} \n'  for i in range(len(measAlts))] )

ploting = True

if ploting:
    #Sätter färger och grejer i grafen
    
    styles = ['dashdot','dashdot','solid','solid']

    colorIndicies = [0,3,2,1]
    
    if testing:
        plotMeas = range(4)
        labels = [
            r"Manuell 1 ",
            r"Manuell 2",
            r"Automatisk 1",
            r"Automatisk 2",
            ]
        
    else: 
        labels = [
        r"Manuell mätning",
        r"Automatiserad mätning"
        ]
        plotMeas = [0,3]


    for i in plotMeas:
        plt.plot(x,BellTotVec[i], color = f'C{colorIndicies[i]}', linestyle = styles[i], alpha = 1, linewidth = 2)

    if not testing:
        for j in range(numStds):
            xPrime = devXvals[j]
            yPrime = devYvals[j] 
            #rita streck med standardavvikelsers avstånd
            has = ['center', *(numStds-1)*['right']]
            plt.plot(2*[xPrime], [0, yPrime], color =  f'C{colorIndicies[-1]}', linestyle = 'dashed',alpha = 0.8 if j == 0 else .5)
            text = '$\widetilde{S}$' + (j!=0)*(' - ' + (j!=1)*str(j) + ' $\sigma_S$')
            plt.text(xPrime - (j!=0)*.005, yPrime + .1, text, ha=has[j], va='bottom',fontsize = 10)


    plt.legend(labels)
    plt.xlim([start,end_])
    plt.ylim([-0,12])
    # plt.grid(True)
    plt.xlabel(r'$S$-värde',fontsize=12)
    plt.ylabel(r'Sannolikhetstäthet $\rho(S)$',fontsize=12)
    plt.title(r'Sannolikhetsfördelning för $S$ för manuella och automatiserade mätningar')
    plt.show()