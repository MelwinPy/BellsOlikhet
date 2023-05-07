import numpy as np

#Läs in datan givet från LabVIEW-programmet
def myLoad(file):
    '''
    count(i,j) är N(alpha_i,beta_j) där alpha_i och beta_j ökar med i/j
    vilket i det naturliga koordinatsystemet är 
     alpha = [0,45,90,135] och beta = [-22.5,22.5,67.5,112.5]
    vilket är alpha', alpha, alpha'_perp, alpha_perp   
     beta', beta, beta'_perp, beta_perp
    
     om man bryr sig om ortogonalitet följer matrisen:
       HH  H'H     VH  V'H
       HH' H'H'    VH' V'H'    
       HV  H'V     VV  V'V
       HV' H'V'    VV' V'V'
    '''

    measurement = np.loadtxt(file,delimiter='\t').astype(float)
    countMat = measurement[:,:4]
    sigmaCountMat = measurement[:,4:]
    return countMat,sigmaCountMat

#Räknar osäkerheten i varje E-värde 
def getEsigma(NMat,sigmaMat,tot):
    #matriser för att räkna ut dE/dN, se rapporten för härledning
    HHMat = np.array([
        [1,1],
        [-1,1]
        ])

    VHMat = np.array([
        [-1,-1],
        [1,-1]
        ])
    
    HVMat = np.array([
        [-1,1],
        [-1,-1]
        ])
    
    VVMat = np.array([
        [1,1],
        [1,-1]
        ])

    bigMat = np.zeros((4,2,2))
    bigMat[0,:,:] = HHMat
    bigMat[1,:,:] = VHMat
    bigMat[2,:,:] = HVMat
    bigMat[3,:,:] = VVMat
   
    Esigma = np.zeros(4)
    
    for i in range(4):
        dEdN = bigMat[i,:,:]*NMat/tot**2
        Esigma[i] = np.sum(np.sum(dEdN*sigmaMat))
    
    Esigma = np.sqrt(sum(Esigma**2))

    return Esigma


def getSandSigma(file):
    #Läs in värden
    countMat,sigmaCountMat = myLoad(file)

    #Ordning är AB, ABp, ApB och ApBp
    NMat = np.zeros((4,2,2))
    NMat[0,:,:] = countMat[1::2,1::2]
    NMat[1,:,:] = countMat[::2,1::2]
    NMat[2,:,:] = countMat[1::2,::2]
    NMat[3,:,:] = countMat[::2,::2]
    
    #totala antalet m'tningar f;r varje inst'llning
    tots = np.sum(np.sum(NMat,1),1)

    #E-värde ges av E = (N_{VV} - N_{VH} - N_{HV} + N_{HH})/tot
    ECoeffs = np.array([
        [1,-1],                
        [-1,1]
        ])
    
    EMat = np.sum(np.sum(ECoeffs*NMat,1),1)/tots
    
    #S-värde ges av E_{AB} - E_{AB'} + E_{A'B} + E_{A'B'}
    SCoeffs = [1,-1,1,1]
    S = sum(EMat*SCoeffs)
    
    #räkna sigma
    sigmaMat = np.zeros((4,2,2))
    sigmaMat[0,:,:] = sigmaCountMat[1::2,1::2]
    sigmaMat[1,:,:] = sigmaCountMat[::2,1::2]
    sigmaMat[2,:,:] = sigmaCountMat[1::2,::2]
    sigmaMat[3,:,:] = sigmaCountMat[::2,::2]

    #Räkna osäkerheten 
    EsigmaVec = np.zeros(4)
    
    #Räknar osäkerheten från mätdatan för varje E-värde
    for i in range(4):
        EsigmaVec[i] = getEsigma(NMat[i,:,:],sigmaMat[i,:,:],tots[i])

    #Räknar osäkerheten från mätdata för S
    sigma = np.sqrt(sum(EsigmaVec**2))

    return S, sigma
