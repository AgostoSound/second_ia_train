import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Importar los datos
ratings = pd.read_csv("u.data", sep="\t", names=["userId", "movieId", "rating", "timestamp"])
movies = pd.read_csv("u.item", sep="\t", names=["movieId", "title", "genres"])

# Eliminar calificaciones no válidas
ratings = ratings[ratings["rating"].between(1, 5)]

# Filtrar películas con pocas calificaciones
min_ratings_per_movie = 5
ratings_count_by_movie = ratings.groupby("movieId").size()
ratings = ratings[ratings["movieId"].isin(ratings_count_by_movie[ratings_count_by_movie >= min_ratings_per_movie].index)]

# Convertir las calificaciones a valores numéricos escalados
scaler = MinMaxScaler(feature_range=(0, 1))
ratings["rating"] = scaler.fit_transform(ratings[["rating"]])

# Crear matrices de datos para entrenamiento y evaluación
from sklearn.model_selection import train_test_split

X = ratings[["userId", "movieId"]]
y = ratings["rating"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Datos de entrenamiento: {X_train.shape}, {y_train.shape}")
print(f"Datos de evaluación: {X_test.shape}, {y_test.shape}")

# ...

# Entrenar el modelo de recomendación
# ...
