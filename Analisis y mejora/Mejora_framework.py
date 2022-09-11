'''
  Algoritmo de regresión lineal con SciKit-Learn 
  Enrique Santos Fraire - A01705746
  09/09/2022
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# Carga y lectura del data set
df = pd.read_excel('Raisin_Dataset.xlsx')

print("Modelo inicial\n")

# Creación del modelo de regresión lineal
model = LinearRegression(fit_intercept=True)

# Asignación de "x" y "y" para el modelo
x = df[["Eccentricity", "Perimeter"]]
y = df['Area']

# Separación de los datos en train y en test
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

# Entrenamiento del modelo
model.fit(x_train, y_train)
print ("Coeficientes del modelo: ", model.coef_)
print ("Intersección: ", model.intercept_)

# T R A I N : Predicciones del modelo
yfit_train = model.predict(x_train)
# Mean Square Error de los datos predichos contra los datos reales
print ("MSE de train: ", mean_squared_error(y_train, yfit_train))
# Precisión del modelo
print ("r2 de train: ", r2_score(y_train, yfit_train))

# T E S T : Predicciones del modelo
yfit_test = model.predict(x_test)
# Mean Square Error de los datos predichos contra los datos reales
print ("MSE de test: ", mean_squared_error(y_test, yfit_test))
# Precisión del modelo
print ("r2 de test: ", r2_score(y_test, yfit_test))

# Cross Validation del modelo
CV = abs(cross_val_score(LinearRegression(), x_train, y_train, cv=10, scoring = "r2").mean())
print ("Cross validation: ", CV)

'''
  Mejora del modelo: Incremento y cambio de variables
'''
print("\n--------------------------------\n")
print("Mejora de modelo\n")

correlation = df.corr()["Area"]
print("Correlación con Área")
print(correlation.sort_values(ascending=False), "\n")

# Creación del modelo de regresión lineal
model2 = LinearRegression(fit_intercept=True)

# Asignación de "x" y "y" para el modelo
x2 = df[["ConvexArea", "Perimeter", "MajorAxisLength", "MinorAxisLength"]]
y2 = df['Area']

# Separación de los datos en train y en test
x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y2, random_state=1)

# Entrenamiento del modelo
model2.fit(x_train2, y_train2)
print ("Coeficientes del modelo mejorado: ", model2.coef_)
print ("Intersección: ", model2.intercept_)

# T R A I N : Predicciones del modelo
yfit_train2 = model2.predict(x_train2)
# Mean Square Error de los datos predichos contra los datos reales
print ("MSE de train: ", mean_squared_error(y_train2, yfit_train2))
# Precisión del modelo
print ("r2 de train: ", r2_score(y_train2, yfit_train2))

# T E S T : Predicciones del modelo
yfit_test2 = model2.predict(x_test2)
# Mean Square Error de los datos predichos contra los datos reales
print ("MSE de test: ", mean_squared_error(y_test2, yfit_test2))
# Precisión del modelo
print ("r2 de test: ", r2_score(y_test2, yfit_test2))

# Cross Validation del modelo
CV2 = abs(cross_val_score(LinearRegression(), x_train2, y_train2, cv=10, scoring = "r2").mean())
print ("Cross validation: ", CV2)

'''
  Graficado de los modelos
'''

# Plot modelo inicial
plt.subplot(1, 2, 1)
plt.scatter(range(len(y_test)), y_test, label = 'y_test')
plt.scatter(range(len(y_test)), yfit_test, label = 'yfit_test')
plt.ylabel("y value")
plt.xlabel("Position")
plt.legend()
plt.title("Initial Model: Test data vs predicted data")

# Plot modelo inicial
plt.subplot(1, 2, 2)
plt.scatter(range(len(y_test2)), y_test2, label = 'y_test2')
plt.scatter(range(len(y_test2)), yfit_test2, label = 'yfit_test2')
plt.ylabel("y value")
plt.xlabel("Position")
plt.legend()
plt.title("Upgraded Model: Test data vs predicted data")

plt.show()
