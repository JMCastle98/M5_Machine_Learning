import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

def separar_datos(x,y, porcentaje_prueba, porcentaje_validation = 0):
    temp_size = porcentaje_validation + porcentaje_prueba
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size = temp_size)
    if(porcentaje_validation > 0):
        test_size = porcentaje_prueba/temp_size
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size = test_size) 
    else: 
        return [x_train, None, x_temp, y_train, None, y_temp]
    return [x_train, x_val, x_test, y_train, y_val, y_test]
    

def separar_datasets(data,K=1, random_seed = 420):
    random = True 
    if K == 1:
        kfold = KFold(data.shape[0], shuffle = random, random_state = random_seed)
    else:
        kfold = KFold(K, shuffle = random, random_state = random_seed)
  
    ciclo = 1
    for indices_train, indices_test in kfold.split(data):
        print("Ciclo: "+str(ciclo))
        print("\t datos para entrenamiento:"+str(data[indices_train]))
        print("\t datos para prueba:"+str(data[indices_test]))
        ciclo+=1

def matriz_confusion(y_expected,y_predicted):
    result = confusion_matrix(y_expected,y_predicted)
    TN,FP,FN,TP = result.ravel()
    print("True positives: "+str(TP))
    print("True negatives: "+str(TN))
    print("False positives: "+str(FP))
    print("False negative: "+str(FN))
    
    return TN,FP,FN,TP 

def scores(TN,FP,FN,TP):
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    accuracy = accuracy * 100

    sensibilidad = TP / (TP + FN)
    sensibilidad = sensibilidad * 100

    especificidad = TN / (TN + FP)
    especificidad = especificidad * 100
    
    return accuracy,sensibilidad,especificidad

def comparar_clasificadores(y_expected_c1,y_predicted_c1,y_expected_c2,y_predicted_c2):
    print("Scores del primer clasificador:")
    TN1,FP1,FN1,TP1 = matriz_confusion(y_expected_c1,y_predicted_c1)
    print("Scores del segundo clasificador:")
    TN2,FP2,FN2,TP2 = matriz_confusion(y_expected_c2,y_predicted_c2)
                                       
    if TN1 > TN2:
        print(f"El clasificador 1 tiene más True Negatives ({TN1})")
    elif TN1 == TN2:
        print(f"Ambos clasificadores tienen el mismo número de True Negatives ({TN1})")
    else: print(f"El clasificador 2 tiene más True Negatives ({TN2})")
                                       
    if FP1 < FP2:
        print(f"El clasificador 1 tiene menos False Positives ({FP1})")
    elif FP1 == FP2:
        print(f"Ambos clasificadores tienen el mismo número de False Positives ({FP1})" )   
    else: print(f"El clasificador 2 tiene más False Positives ({FP2})")
                                       
    if FN1 < FN2:
        print(f"El clasificador 1 tiene menos False Negatives ({FN1})")
    elif FN1 == FN2:
        print(f"Ambos clasificadores tienen el mismo número de False Negatives({FN1})")  
    else: print(f"El clasificador 2 tiene más False Negatives ({FN2})")
                                       
    if TP1 > TP2:
        print(f"El clasificador 1 tiene más True Positives ({TP1})")
    elif TP1 == TP2:
        print(f"Ambos clasificadores tienen el mismo número de True Positives ({TP1})")    
    else: print(f"El clasificador 2 tiene más True Positives({TP2})")
                                       
                                       
    accuracy_c1,sensibilidad_c1,especificidad_c1 = scores(TN1,FP1,FN1,TP1)
    accuracy_c2,sensibilidad_c2,especificidad_c2 = scores(TN2,FP2,FN2,TP2)
                                       
                                                    
    if accuracy_c1 > accuracy_c2:
        print(f"El clasificador 1 tiene mejor accuracy con un valor de: {accuracy_c1}")
    elif accuracy_c1 == accuracy_c2:
        print(f"Ambos clasificadores tienen un accuracy con valor de: {accuracy_c1}")
    else: print(f"El clasificador 2 tiene mejor accuracy con un valor de: {accuracy_c2}")
                                       
    if sensibilidad_c1 > sensibilidad_c2:
        print(f"El clasificador 1 tiene mejor sensibilidad con un valor de: {sensibilidad_c1}")
    elif sensibilidad_c1 == sensibilidad_c2:
        print(f"Ambos clasificadores tienen una sensibilidad con valor de: {sensibilidad_c1}")
    else: print(f"El clasificador 2 tiene mejor sensibilidad con un valor de: {sensibilidad_c2}")     
    
    if especificidad_c1 > especificidad_c2:
        print(f"El clasificador 1 tiene mejor especificidad con un valor de: {especificidad_c1}")
    elif especificidad_c1 == especificidad_c2:
        print(f"Ambos clasificadores tienen una especificidad con valor de: {especificidad_c1}")
    else: print(f"El clasificador 2 tiene mejor especificidad con un valor de: {especificidad_c2}")
                                       
                                       
                                           
        
    
        
        
        

        
        
        
        
        

