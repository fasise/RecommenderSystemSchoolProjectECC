from flask import Flask, render_template, jsonify, request
import json
import pickle
import math

def sanitize_data(data):
    """Remplace les valeurs NaN par None ou une chaîne vide pour les rendre valides en JSON."""
    if isinstance(data, list):
        return [sanitize_data(item) for item in data]
    elif isinstance(data, dict):
        return {key: sanitize_data(value) for key, value in data.items()}
    elif isinstance(data, str) and data.lower() == "nan":
        return None  # Ou "" si vous préférez une chaîne vide
    elif isinstance(data, float) and math.isnan(data):
        return None  # Ou "" si vous préférez une chaîne vide
    return data

# import pickle
from content_based_recommender import content_based_recommender

# Définir la fonction model_final avant de charger le modèle
def model_final(user_id):
    return content_based_recommender(
        dfc,
        tfidf_clean, 
        user_profiles_normalized,
        user_id
    )

# Charger la fonction `model_final` depuis le fichier pickle
with open('./models/model_content_based_recommendation.pkl', 'rb') as f:
    model_final = pickle.load(f)

# Charger les objets nécessaires
with open('./models/data.pkl', 'rb') as f:
    dfc = pickle.load(f)

with open('./models/matrice_tfidf.pkl', 'rb') as f:
    tfidf_clean = pickle.load(f)

with open('./models/Profiles_users.pkl', 'rb') as f:
    user_profiles_normalized = pickle.load(f)

# print(user_profiles_normalized)

user_id =user_profiles_normalized.index[20]
# Définir `user_id` pour les recommandations
# user_id = 'AFNYIBWKJLJQKY4BGK77ZOTVMORA,AFCTNNMP2LZLY5466YJ5AY3JE5ZA,AG3XBWOAL65DJSBHJ7LQ2K54HJKQ'

# Utiliser la fonction chargée pour obtenir les recommandations

# Afficher les résultats
# print(top_recommendations)






app = Flask(__name__)


with open('users.json', 'r') as f:
    loaded_users = json.load(f)

# Fonction pour charger les utilisateurs depuis le fichier JSON (ou utiliser un mock ici)
def load_users():
    # Vous pouvez charger les utilisateurs depuis un fichier JSON ou une base de données.
    return [
        {
            "id": 1,
            "name": "Utilisateur 1",
            "boughtProducts": [
                {"name": "Produit A", "image": "https://via.placeholder.com/100", "similarity": 0.85},
                {"name": "Produit B", "image": "https://via.placeholder.com/100", "similarity": 0.75}
            ],
            "recommendedProducts": [
                {"name": "Produit X", "image": "https://via.placeholder.com/100", "similarity": 0.90},
                {"name": "Produit Y", "image": "https://via.placeholder.com/100", "similarity": 0.80}
            ]
        },
        {
            "id": 2,
            "name": "Utilisateur 2",
            "boughtProducts": [
                {"name": "Produit C", "image": "https://via.placeholder.com/100", "similarity": 0.70},
                {"name": "Produit D", "image": "https://via.placeholder.com/100", "similarity": 0.65}
            ],
            "recommendedProducts": [
                {"name": "Produit Z", "image": "https://via.placeholder.com/100", "similarity": 0.88},
                {"name": "Produit W", "image": "https://via.placeholder.com/100", "similarity": 0.76}
            ]
        },
        {
            "id": 3,
            "name": "Utilisateur 3",
            "boughtProducts": [
                {"name": "Produit E", "image": "https://via.placeholder.com/100", "similarity": 0.92},
                {"name": "Produit F", "image": "https://via.placeholder.com/100", "similarity": 0.79}
            ],
            "recommendedProducts": [
                {"name": "Produit M", "image": "https://via.placeholder.com/100", "similarity": 0.95},
                {"name": "Produit N", "image": "https://via.placeholder.com/100", "similarity": 0.85}
            ]
        }
    ]

# Route pour afficher la page principale avec la liste des utilisateurs
@app.route('/')
def index():
    # users = load_users()  # Charger les utilisateurs depuis le fichier JSON
    users= loaded_users
    return render_template('index.html', users=users)

# Route pour récupérer les produits achetés et recommandés par un utilisateur spécifique
@app.route('/recommender/<user_id>', methods=['GET'])
def recommender(user_id):
    # Appel de la fonction model_final pour obtenir les données de l'utilisateur
    user_data = model_final(user_id)
    print("Model Final Output:", user_data)  # Affiche la sortie du modèle pour débogage

    # Vérifier si le résultat est une liste et contient au moins un utilisateur
    if isinstance(user_data, list) and len(user_data) > 0:
        user = user_data[0]  # On accède au premier utilisateur dans la liste
        
        # Vérifier si l'utilisateur a bien les produits achetés et recommandés
        if 'boughtProducts' in user and 'recommendedProducts' in user:
            # Appliquer la fonction sanitize_data pour nettoyer les valeurs invalides
            sanitized_user_data = {
                'boughtProducts': sanitize_data(user['boughtProducts']),
                'recommendedProducts': sanitize_data(user['recommendedProducts'])
            }
            return jsonify(sanitized_user_data)
        else:
            return jsonify({'error': 'Données utilisateur manquantes'}), 400
    else:
        return jsonify({'error': 'Utilisateur non trouvé ou données incorrectes'}), 404


if __name__ == '__main__':
    app.run(debug=True)
