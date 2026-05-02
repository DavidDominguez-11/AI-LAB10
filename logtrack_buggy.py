#logtrack_buggy.py
import numpy as np

np.random.seed(42)

CARRILES = 20
K = 10

def transicion(h_prev):
    # El vehiculo se mueve +1, 0 o -1 carril con igual probabilidad
    delta = np.random.choice([-1, 0, 1])
    return int(np.clip(h_prev + delta, 0, CARRILES - 1))

def emision(sensor, h):
    # Sensor reporta el carril real con prob 0.6, adyacente con 0.2, error con 0.2
    dist = abs(sensor - h)
    if dist == 0: 
        return 0.6
    elif dist == 1: 
        return 0.2
    else: 
        return 0.2 / (CARRILES - 2)

def filtrado_particulas(observaciones):
    particulas = np.random.randint(0, CARRILES, K)
    for t, sensor in enumerate(observaciones):
        # PASO 1: Proponer
        propuestas = np.array([transicion(h) for h in particulas])
        
        # PASO 2: Ponderar
        pesos = np.array([emision(sensor, h) for h in propuestas])
        pesos_norm = pesos / pesos.sum()
        
        # PASO 3: Remuestrear <-- REVISEN ESTA LINEA
        #idx = np.argsort(pesos_norm)[-K:]
        idx = np.random.choice(len(propuestas), size=K, p=pesos_norm)
        particulas = propuestas[idx]
        
        print(f"t={t+1} | sensor={sensor} | particulas={sorted(particulas)}")
    return particulas

# Secuencia de sensores simulando movimiento real del vehiculo
observaciones = [5, 6, 7, 8, 6, 5]
filtrado_particulas(observaciones)