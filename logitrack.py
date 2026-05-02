import numpy as np
import matplotlib.pyplot as plt

CARRILES = 20
K = 5


# Dinámica del vehículo (real y partículas)
def transicion(h):
    delta = np.random.choice([-1, 0, 1])
    nuevo = h + delta
    
    # Reflexión en bordes
    if nuevo < 0:
        nuevo = 1
    elif nuevo >= CARRILES:
        nuevo = CARRILES - 2
        
    return nuevo


# Modelo de sensor

def emision(sensor, h):
    dist = abs(sensor - h)
    
    if dist == 0:
        return 0.6
    elif dist == 1:
        return 0.2
    else:
        return 0.2 / (CARRILES - 2)


# Simulación del vehículo real

def simular_vehiculo(pasos):
    trayectoria_real = []
    observaciones = []
    
    h_real = np.random.randint(0, CARRILES)
    
    for _ in range(pasos):
        # Movimiento real
        h_real = transicion(h_real)
        trayectoria_real.append(h_real)
        
        # Generar sensor
        probs = []
        for i in range(CARRILES):
            probs.append(emision(i, h_real))
        
        probs = np.array(probs)
        probs /= probs.sum()
        
        sensor = np.random.choice(range(CARRILES), p=probs)
        observaciones.append(sensor)
    
    return trayectoria_real, observaciones


# Filtrado de partículas

def filtro_particulas(observaciones):
    particulas = np.random.randint(0, CARRILES, K)
    
    estimaciones = []
    historial_particulas = []
    
    for sensor in observaciones:
        # PASO 1: Proponer
        propuestas = np.array([transicion(h) for h in particulas])
        
        # PASO 2: Ponderar
        pesos = np.array([emision(sensor, h) for h in propuestas])
        pesos /= pesos.sum()
        
        # PASO 3: Remuestreo (CORRECTO)
        idx = np.random.choice(range(K), size=K, p=pesos)
        particulas = propuestas[idx]
        
        # PASO 4: Estimación (media)
        estimacion = np.mean(particulas)
        
        estimaciones.append(estimacion)
        historial_particulas.append(particulas.copy())
    
    return estimaciones, historial_particulas


# Visualización

def visualizar(trayectoria_real, estimaciones, historial_particulas):
    pasos = len(trayectoria_real)
    
    plt.figure(figsize=(12,6))
    
    # Línea real
    plt.plot(range(pasos), trayectoria_real, label="Trayectoria real")
    
    # Estimación
    plt.plot(range(pasos), estimaciones, linestyle='--', label="Estimación (media partículas)")
    
    # Partículas
    for t in range(pasos):
        plt.scatter([t]*K, historial_particulas[t], alpha=0.5)
    
    plt.xlabel("Tiempo")
    plt.ylabel("Carril")
    plt.legend()
    plt.title("Seguimiento con Filtro de Partículas (K=5)")
    plt.show()


# Ejecutar ejemplo

np.random.seed(42)

trayectoria_real, observaciones = simular_vehiculo(30)
estimaciones, historial = filtro_particulas(observaciones)

visualizar(trayectoria_real, estimaciones, historial)