# Importation des bibliothèques nécessaires
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Charger les données
df = pd.read_csv('AmesHousing.csv')

# Nettoyage des données
df.columns = df.columns.str.strip()
df.fillna(df.median(numeric_only=True), inplace=True)
df['Year Built'] = pd.to_numeric(df['Year Built'], errors='coerce')

# Calcul de l'âge de la maison
df['Age'] = df['Yr Sold'] - df['Year Built']

# Encodage des variables catégorielles
df_encoded = pd.get_dummies(df, columns=['Neighborhood', 'House Style'])

# Préparation des données pour le modèle de régression
# Sélection des variables
X = df[['Gr Liv Area', 'Total Bsmt SF', '1st Flr SF']]
y = df['SalePrice']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# Recherche des paramètres optimaux pour Ridge et Lasso
ridge = Ridge()
lasso = Lasso()

param_grid = {
    'alpha': [0.1, 1.0, 10.0, 100.0]
}

ridge_cv = GridSearchCV(ridge, param_grid, cv=5)
lasso_cv = GridSearchCV(lasso, param_grid, cv=5)

ridge_cv.fit(X_train, y_train)
lasso_cv.fit(X_train, y_train)

# Prédictions avec les modèles optimisés
y_pred_linear = model.predict(X_test)
y_pred_ridge = ridge_cv.predict(X_test)
y_pred_lasso = lasso_cv.predict(X_test)

# Évaluation du modèle
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print(f'Erreur Quadratique Moyenne (MSE) - Régression Linéaire : {mse_linear}')
print(f'Coefficient de Détermination (R²) - Régression Linéaire : {r2_linear}')

print(f'Erreur Quadratique Moyenne (MSE) - Ridge : {mse_ridge}')
print(f'Coefficient de Détermination (R²) - Ridge : {r2_ridge}')

print(f'Erreur Quadratique Moyenne (MSE) - Lasso : {mse_lasso}')
print(f'Coefficient de Détermination (R²) - Lasso : {r2_lasso}')

# Analyse des résidus pour la régression linéaire
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_linear)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('Valeurs Réelles')
plt.ylabel('Valeurs Prédites')
plt.title('Valeurs Réelles vs Prédites - Régression Linéaire')
plt.show()
