'''
  Algoritmo de regresion lineal com SciKit-Learn 
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

# Creación del modelo de regresión lineal
model = LinearRegression(fit_intercept=True)

# Asignación de "x" y "y" para el modelo
x = df[["Eccentricity", "Perimeter"]]
y = df['Area']

# Separación de los datos en train y en test
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

# Entrenamiento del modelo
model.fit(x_train, y_train)
print ("Coeficiente del modelo: ", model.coef_)
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

# Plot
plt.scatter(range(len(y_test)), y_test, label = 'y_test')
plt.scatter(range(len(y_test)), yfit_test, label = 'yfit_test')
plt.ylabel("y value")
plt.xlabel("Position")
plt.legend()
plt.title("Test data vs predicted data")

plt.show()