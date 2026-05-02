import numpy as np
import matplotlib.pyplot as plt

# Parámetros
CARRILES = 20

# Dinámica (transición)
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
        h_real = transicion(h_real)
        trayectoria_real.append(h_real)

        probs = np.array([emision(i, h_real) for i in range(CARRILES)])
        probs /= probs.sum()

        sensor = np.random.choice(range(CARRILES), p=probs)
        observaciones.append(sensor)

    return trayectoria_real, observaciones

# Filtro de partículas (K variable)
def filtro_particulas_K(observaciones, K):
    particulas = np.random.randint(0, CARRILES, K)

    estimaciones = []
    historial_particulas = []

    for sensor in observaciones:
        propuestas = np.array([transicion(h) for h in particulas])

        pesos = np.array([emision(sensor, h) for h in propuestas])
        pesos /= pesos.sum()

        idx = np.random.choice(range(K), size=K, p=pesos)
        particulas = propuestas[idx]

        estimaciones.append(np.mean(particulas))
        historial_particulas.append(particulas.copy())

    return estimaciones, historial_particulas

# Visualización básica
def visualizar(trayectoria_real, estimaciones, historial_particulas, K):
    pasos = len(trayectoria_real)

    plt.figure(figsize=(12,6))
    plt.plot(range(pasos), trayectoria_real, label="Trayectoria real")
    plt.plot(range(pasos), estimaciones, linestyle='--', label="Estimación")

    for t in range(pasos):
        plt.scatter([t]*K, historial_particulas[t], alpha=0.5)

    plt.xlabel("Tiempo")
    plt.ylabel("Carril")
    plt.title(f"Filtro de Partículas (K={K})")
    plt.legend()
    plt.show()

# Error
def calcular_error(trayectoria_real, estimaciones):
    return np.array([abs(r - e) for r, e in zip(trayectoria_real, estimaciones)])

# Experimento (Task 3.2)
def experimento(K, num_sim=50, pasos=30):
    errores_totales = []
    errores_promedio_sim = []

    for _ in range(num_sim):
        trayectoria_real, observaciones = simular_vehiculo(pasos)
        estimaciones, _ = filtro_particulas_K(observaciones, K)

        errores = calcular_error(trayectoria_real, estimaciones)

        errores_totales.append(errores)
        errores_promedio_sim.append(np.mean(errores))

    errores_totales = np.array(errores_totales)
    error_promedio_tiempo = errores_totales.mean(axis=0)

    return errores_totales, error_promedio_tiempo, np.array(errores_promedio_sim)

# Métricas
def metricas(errores_totales, errores_sim):
    error_promedio = np.mean(errores_sim)
    error_max = np.max(errores_totales)
    porcentaje_error_alto = np.mean(errores_sim > 5) * 100
    return error_promedio, error_max, porcentaje_error_alto

# Diversidad (Task 3.3)
def diversidad_varianza(particulas):
    return np.var(particulas)

def alerta_colapso(particulas, umbral):
    return np.var(particulas) < umbral

# Filtro con monitoreo de colapso
def filtro_particulas_monitoreo(observaciones, K, umbral):
    particulas = np.random.randint(0, CARRILES, K)

    estimaciones = []
    historial_particulas = []
    alertas = []

    for sensor in observaciones:
        propuestas = np.array([transicion(h) for h in particulas])

        pesos = np.array([emision(sensor, h) for h in propuestas])
        pesos /= pesos.sum()

        idx = np.random.choice(range(K), size=K, p=pesos)
        particulas = propuestas[idx]

        estimaciones.append(np.mean(particulas))
        historial_particulas.append(particulas.copy())
        alertas.append(alerta_colapso(particulas, umbral))

    return estimaciones, historial_particulas, alertas

# Visualización con alertas
def visualizar_con_alertas(trayectoria_real, estimaciones, historial, alertas):
    pasos = len(trayectoria_real)

    plt.figure(figsize=(12,6))
    plt.plot(trayectoria_real, label="Real")
    plt.plot(estimaciones, linestyle="--", label="Estimación")

    for t in range(pasos):
        plt.scatter([t]*len(historial[t]), historial[t], alpha=0.5)

        if alertas[t]:
            plt.axvline(x=t, linestyle=':', alpha=0.3)

    plt.legend()
    plt.title("Detección de colapso")
    plt.show()

# MAIN
np.random.seed(42)

"""
# ---------- Task 3.1 ----------
trayectoria_real, observaciones = simular_vehiculo(30)
estimaciones, historial = filtro_particulas_K(observaciones, K=5)
visualizar(trayectoria_real, estimaciones, historial, K=5)

# ---------- Task 3.2 ----------
errores_K5, error_tiempo_K5, errores_sim_K5 = experimento(K=5)

plt.figure(figsize=(10,5))
plt.plot(error_tiempo_K5)
plt.title("Error promedio por paso (K=5)")
plt.xlabel("Tiempo")
plt.ylabel("Error absoluto medio")
plt.show()

peores_idx = np.argsort(errores_sim_K5)[-5:]
print("Peores simulaciones K=5:", peores_idx)
print("Errores:", errores_sim_K5[peores_idx])

errores_K20, _, errores_sim_K20 = experimento(K=20)

m_K5 = metricas(errores_K5, errores_sim_K5)
m_K20 = metricas(errores_K20, errores_sim_K20)

print("\nComparación:")
print("K=5  ->", m_K5)
print("K=20 ->", m_K20)

"""
# ---------- Task 3.3 ----------

umbral = 0.5

# Escenario 1: sin colapso
obs1 = [5,6,7,6,7,8,7,6,7,8]
real1, _ = simular_vehiculo(len(obs1))
est1, hist1, alert1 = filtro_particulas_monitoreo(obs1, 5, umbral)
visualizar_con_alertas(real1, est1, hist1, alert1)

# Escenario 2: colapso recuperable
obs2 = [5,6,7,8,8,8,7,6,5]
real2, _ = simular_vehiculo(len(obs2))
est2, hist2, alert2 = filtro_particulas_monitoreo(obs2, 5, umbral)
visualizar_con_alertas(real2, est2, hist2, alert2)

# Escenario 3: colapso irrecuperable
obs3 = [5,6,7,8,8,8,3,3,3]
real3, _ = simular_vehiculo(len(obs3))
est3, hist3, alert3 = filtro_particulas_monitoreo(obs3, 5, umbral)
visualizar_con_alertas(real3, est3, hist3, alert3)