import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import time
import random
import Problema as p 

#Borrar
#Cambio
#Revisar


M = [[1, 0, 0, 2, 3, 0, 0, 0, 4], [0, 1, 0, 0, 0, 2, 0, 3, 0], [0, 1, 0, 0, 2, 0, 0, 0, 3], [1, 0, 0, 2, 3, 0, 0, 4, 0], [0, 1, 0, 0, 2, 0, 0, 0, 3], [0, 1, 0, 0, 2, 0, 0, 3, 0], [1, 0, 2, 0, 3, 0, 0, 4, 0], [1, 0, 2, 3, 0, 0, 0, 4, 0], [1, 0, 2, 3, 0, 0, 0, 4, 0], [1, 0, 2, 3, 4, 0, 0, 5, 0], [0, 1, 0, 0, 0, 0, 2, 0, 3], [1, 0, 0, 2, 0, 0, 3, 0, 4], [0, 1, 0, 0, 0, 0, 2, 3, 0], [1, 0, 0, 0, 0, 0, 2, 3, 0], [1, 0, 0, 2, 3, 0, 0, 0, 4], [0, 1, 0, 0, 0, 2, 0, 0, 3], [0, 1, 0, 2, 3, 0, 0, 4, 0], [1, 2, 0, 0, 0, 3, 0, 0, 4], [1, 0, 0, 0, 0, 2, 0, 0, 0], [0, 1, 0, 0, 0, 2, 0, 0, 0]]
Tik = np.array([(5, 0, 0, 3, 4, 0, 0, 0, 5), (0, 5, 0, 0, 0, 5, 0, 4, 0), (0, 5, 0, 0, 4, 0, 0, 0, 5), (4, 0, 0, 4, 3, 0, 0, 4, 0), (0, 5, 0, 0, 3, 0, 0, 0, 3), (0, 5, 0, 0, 3, 0, 0, 4, 0), (4, 0, 3, 0, 4, 0, 0, 3, 0), (4, 0, 3, 3, 0, 0, 0, 3, 0), (5, 0, 3, 5, 0, 0, 0, 3, 0), (5, 0, 3, 5, 4, 0, 0, 4, 0), (0, 3, 0, 0, 0, 0, 5, 0, 5), (4, 0, 0, 4, 0, 0, 5, 0, 5), (0, 5, 0, 0, 0, 0, 5, 5, 0), (4, 0, 0, 0, 0, 0, 5, 5, 0), (5, 0, 0, 4, 3, 0, 0, 0, 5), (0, 3, 0, 0, 0, 4, 0, 0, 5), (0, 3, 0, 4, 3, 0, 0, 5, 0), (5, 3, 0, 0, 0, 4, 0, 0, 5), (4, 0, 0, 0, 0, 3, 0, 0, 0), (0, 4, 0, 0, 0, 4, 0, 0, 0)])
R = [3, 3, 2, 2, 2, 2, 4, 2]
Pi = [75, 130, 110, 145, 110, 105, 140, 115] 
A = [375, 375, 0, 1300, 0, 650, 1110, 0, 0, 1450, 1100, 550, 525, 0, 700, 0, 700, 700, 575, 0]
Bk = [900, 2000, 2000, 1600, 1500, 1800, 1400, 1700, 1500]
MTBFk = [90, 51, 73, 60, 76, 62, 71, 58, 65]
c=2
lb=2
ub=6



instancia = p.Problema(M, Tik, R, Pi, A, Bk, MTBFk, c, lb, ub)


datos=[]
subiter=[]
Ttime=[]
TOP=[]
TOP12=[]
winners=[]
VectorS=[]
VectorF=[]


VECTOR=[]
SOLUCION=[]
PBWO=[]
FFF=[]



begtime=time.time()
begoptime=time.time()

vectorF=[]
vectorS=[]

## Cross-over

def crossover(p1, p2):
    nvar = len(p1)
    
    
    c12=[]
    for w in range(int(nvar/2)):
        masc=[]
        for i in range(len(M)+c*len(Bk)):
            ale=random.randint(0, 1)
            masc.append(ale)

        c1=[]
        c2=[]
        for j in range(len(masc)):
            if masc[0]==1:
                c1.append(p1[j])
                c2.append(p2[j])
            else:
                c1.append(p1[j])
                c2.append(p2[j])
        c12.append(c1)
        c12.append(c2)
    return(c12)


def bw_optimizer(pp, cr, pm, npop, max_iter, g, CdT):
    
    cont_CdT=0
    
    vectorF=[]
    vectorS=[]

    # Inicializar población
    pob = []
    fitness = []
    for i in range(npop):
        pob.append(instancia.generarSolucion())
    
    for i in range(npop):
        fitness.append(instancia.calcularFitness(pob[i]))
    
    full_time=0
    
###############################################################################

    # Evaluar población
    
    orden=np.argsort(fitness)
    cfppop=pob.copy()
    cfpfitness=fitness.copy()
    
    pob = []
    fitness = []
    
    for i in range(npop):
        pob.append(cfppop[orden[i]])
        fitness.append(cfpfitness[orden[i]])
        
    gbest = pob[0].copy()
    fit_gbest = fitness[0].copy()

###############################################################################

    # Inicializar variables
    
    nr = int(npop*pp) # Número de participantes en la reproducción

    T_OpTime=0
    
###############################################################################

    # Iteraciones
    for w in range(0, max_iter):
        
        if cont_CdT<CdT: #Condición de término por repetición
         
            pop1=[]
            for i in range(nr):
                pop1.append(pob[i])
    
            pop2=[] #Cambio: Podría ser igual que decir pop2=pop[npop-1].copy()?????
            for i in range(len(pob[npop-1])):
                pop2.append(pob[npop-1][i])
            
            
            fit=[]
            
            
            for i in range(npop): #Se ordena porque llegan de la iteración previa (No tomar en cuenta la iteración 0)
                fit.append(instancia.calcularFitness(pob[i]))
            
            orden=np.argsort(fit)
            cfppop=pob.copy()
            cfpfit=fit.copy()
            
            pob = []
            fit = []
            
            for i in range(npop):
                pob.append(cfppop[orden[i]])
                fit.append(cfpfit[orden[i]])
            
            best = pob[0].copy()
            fit_best = fit[0].copy()
    
            vectorF.append(fit_best)
            
            if fit_best == fit_gbest: #Contador para condición de término por repectición
                cont_CdT=cont_CdT+1
            else:
                cont_CdT=1 #Significa que el nuevo valor es menor (No puede ser mayor)
            
            if fit_best < fit_gbest:
                
                gbest = best.copy()
                fit_gbest = fit_best.copy()

                #if fit_gbest <= 4671.344198054972: #Condición opcional de optimalidad (Borrar)
                #    optime=time.time()
                #    T_OpTime=optime-begoptime
                #    winners.append(gbest) #Borrar
            
    ###############################################################################
            childsize=0        
    
            ## Reproduciones
            
            for j in range(0, nr):
                
                p1=[]
                p2=[]
                
                i1=random.randint(0, len(pop1)-1)
                p1=pop1[i1].copy()
                pop1.pop(i1) #Se elimina este elemento de la población para evitar seleccionar al mismo padre 2 veces (i1!=i2)
                
                i2=random.randint(0, len(pop1)-1)
                p2=pop1[i2].copy()

#[_____________________________________________PARTE NUEVA___________________________________________________________________
                
                #Podría reordenarce nuevamente el pop1 para que sea como era inicialmente
                
                pop1.append(p1) #Se vuelve a añadir el padre eliminado (p1) a pop1
                
                fitpop1=[]
                for i in range(0, len(pop1)):
                    fitpop1.append(instancia.calcularFitness(pop1[i]))
                
                orden1=np.argsort(fitpop1)
                cfppop1=pop1.copy()
                
                pop1 = []
                
                for i in range(0, len(cfppop1)): #Se ordena de nuevo pop1 de mejor a peor (menor a mayor)
                    pop1.append(cfppop1[orden1[i]])

#_____________________________________________PARTE NUEVA___________________________________________________________________]
                
                child = crossover(p1, p2)
               
                children = []        
                
                for i in range(len(child)): 
                    children.append(instancia.repararSolucion(child[i]))
                    
    ###############################################################################        
                
                ### Canibalismo de padre
                            
                if instancia.calcularFitness(p1) > instancia.calcularFitness(p2): #Solo vuelve el mejor padre
                    pop2=np.vstack((pop2, p2))
                    
                else:
                    pop2=np.vstack((pop2, p1))
    
    ###############################################################################
    
                ### Canibalismo de hermanos
                
                childfit=[]
                for i in range(0, len(children)):
                    childfit.append(instancia.calcularFitness(children[i]))
                
                orden=np.argsort(childfit)
                cfpchildren=children.copy()
                cfpchildfit=childfit.copy()
                
                children = []
                childfit = []
                
                for i in range(0, len(cfpchildren)): #Se ordenan los hijos de mejor a peor (menor a mayor)
                    children.append(cfpchildren[orden[i]])
                    childfit.append(cfpchildfit[orden[i]])
                
                cfpchildren=children.copy()
                cfpchildfit=childfit.copy()
                
                children = []
                childfit = []
                
                for i in range(int(len(cfpchildren)*cr)): #Revisar: Aquí está la prueba de que no se necesita el ns
                    children.append(cfpchildren[i])
                    childfit.append(cfpchildfit[i])
                
                pop2 = np.vstack((pop2, children))
                
                childsize=childsize+len(children) #Cantidad de hijos para calcular la mutación según el pseudo-código

            for i in range(len(pop2)):
                pop2[i]=instancia.repararSolucion(pop2[i]) #Revisar: Volver a reparar?

    ###############################################################################
    
            ## Mutaciones
            
            pop3 = []
            nm = int(childsize*pm)
            
            for i in range(nm):
                index = random.randint(0, len(pop1)-1)
    
                mut = pop1[index].copy() #Selección de widow a mutar
                
                lm=list(range(len(mut))) #Lista de posiciones para poder mutar
    
                cp1=random.choice(lm)
                lm.remove(cp1)
                
                cp2=random.choice(lm)
    
                mut[cp1], mut[cp2] = mut[cp2], mut[cp1]

                pop3.append(instancia.repararSolucion(mut))
            
            pop2 = np.vstack((pop2, pop3))
    
            for i in range(len(pop2)): #Revisar: Volver a reparar?
                pop2[i]=instancia.repararSolucion(pop2[i]) #Reparar las widows infactibles
            
    ###############################################################################
    
            ## Ordenar por fitness
            
            fit2=[]
            for i in range(len(pop2)):
                fit2.append(instancia.calcularFitness(pop2[i]))
            
            orden=np.argsort(fit2)
            cfppop2=pop2.copy()
            cfpfit2=fit2.copy()
            
            pop2 = []
            fit2 = []
            
            for i in range(npop):
                
                pop2.append(cfppop2[orden[i]])
                fit2.append(cfpfit2[orden[i]])
    
    ###############################################################################
    
            ## Selección de mejores resultados
            
            pob = pop2.copy()
            
            
        
        

    return instancia.calcularFitness(gbest), gbest, full_time, T_OpTime, winners, vectorS, vectorF


def main():
    # Lectura de archivo
    
    PP=[0.2, 0.4, 0.6, 0.8]
    CR=[0.2, 0.4, 0.6, 0.8]
    PM=[0.2, 0.4, 0.6, 0.8]
    NPOP=[25, 50, 75, 100]
    MAX_ITER=[25, 50, 75, 100]
    #PP=[0.5]
    #CR=[0.4]
    #PM=[0.8]
    #NPOP=[50]
    #MAX_ITER=[75]
    
    contar=0
    cont=0
    
    for i in range(len(PP)):
        pp=PP[i]
        for j in range(len(CR)):
            cr=CR[j]
            for k in range(len(PM)):
                pm=PM[k]
                for l in range(len(NPOP)):
                    npop=NPOP[l]
                    for m in range(len(MAX_ITER)):
                        max_iter=MAX_ITER[m]
                        contar=contar+1
                        
                        
                        CdT=int(npop*0.2) #max_iter #Condición de término
                        
                        pbwo=[]
                        pbwo.append(pp)
                        pbwo.append(cr)
                        pbwo.append(pm)
                        pbwo.append(npop)
                        pbwo.append(max_iter)
                        PBWO.append(pbwo)
                        
                        winners=[]
                        VectorS=[]
                        VectorF=[]
                        FF=[]
                        
                        for g in range(30):
                            
                            cont=cont+1

                            fit_gbest, gbest, full_time, T_OpTime, winners, vectorS, vectorF = bw_optimizer(pp, cr, pm, npop, max_iter, g, CdT)
                            
                            
                            
                            ftime=time.time()
                            F=ftime-begtime
                            FF.append(F)
                            
                            #print("Resultados de la iteración ", g+1, " de la combinación ", PBWO)
                            print("------------------------------------------------------------------------------------------")
                            print("Solución ", contar, "-", cont, ":", fit_gbest, " de pp=", pp, ";cr=", cr, ";pm=", pm, ";npop=", npop, ";max_iter=", max_iter)
                            print("Tiempo con condición de termino: ", F)
                            print("------------------------------------------------------------------------------------------")
                            
                            VectorS.append(gbest)
                            VectorF.append(vectorF)
                            
                        SOLUCION.append(VectorS)
                        VECTOR.append(VectorF)
                        FFF.append(FF)
                        
                        
                        matriz=[PBWO, FFF, VECTOR] #1.- Valor Óptimo, 2.- Vector solución, 3.- Tiempo para encontrar el óptimo, 4.- Evolución de la widow
                        Mmatriz=np.transpose(matriz)
                        df_rrss=pd.DataFrame(Mmatriz, columns=['pp, cr, pm, npop, max_iter', 'Tiempo', 'Réplicas x Secuencias'])
                        df_rrss.to_excel('BWO & MV-CFP (Datos)_2.xlsx') #______EXCEL______EXCEL______EXCEL______EXCEL______EXCEL______EXCEL______EXCEL______EXCEL______EXCEL

                        
                        

if __name__ == '__main__':
    main()
    
print("***********FINALIZADO***********FINALIZADO***********FINALIZADO***********FINALIZADO")