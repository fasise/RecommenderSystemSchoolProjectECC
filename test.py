import pickle
from content_based_recommender import content_based_recommender
import json

# Définir la fonction model_final avant de charger le modèle
def model_final(user_id):
    return content_based_recommender(
        dfc,
        tfidf_clean, 
        user_profiles_normalized,
        user_id
    )

# Charger la fonction `model_final` depuis le fichier pickle
with open('model_content_based_recommendation.pkl', 'rb') as f:
    model_final = pickle.load(f)

# Charger les objets nécessaires
with open('data.pkl', 'rb') as f:
    dfc = pickle.load(f)

with open('matrice_tfidf.pkl', 'rb') as f:
    tfidf_clean = pickle.load(f)

with open('Profiles_users.pkl', 'rb') as f:
    user_profiles_normalized = pickle.load(f)

# print(user_profiles_normalized)

user_id =user_profiles_normalized.index[20]
# Définir `user_id` pour les recommandations
# user_id = 'AFNYIBWKJLJQKY4BGK77ZOTVMORA,AFCTNNMP2LZLY5466YJ5AY3JE5ZA,AG3XBWOAL65DJSBHJ7LQ2K54HJKQ'

# Utiliser la fonction chargée pour obtenir les recommandations
top_recommendations = model_final(user_id)

# Afficher les résultats
print(top_recommendations)

# Fonction pour créer une liste d'utilisateurs avec un ID unique et un nom généré
# def generate_users(user_profiles_normalized):
#     users = []
    
#     # Si 'user_id' est l'index, nous l'extrayons pour créer des ID uniques.
#     for idx, user_id in enumerate(user_profiles_normalized.index):  # ou si 'user_id' est une colonne : user_profiles_normalized['user_id']
#         user = {
#             "id": user_id,  # L'ID commence à 1
#             "name": f"Utilisateur {idx + 1}"  # Générer un nom, sinon vous pouvez avoir un vrai nom ici
#         }
#         users.append(user)
    
#     return users

# # Créer la liste d'utilisateurs
# all_users_list = generate_users(user_profiles_normalized)

# # Afficher la liste des utilisateurs
# print(all_users_list[:10])  # Afficher les 10 premiers utilisateurs pour tester

# # Enregistrer la liste d'utilisateurs dans un fichier JSON
# with open('users.json', 'w') as f:
#     json.dump(all_users_list, f, ensure_ascii=False, indent=4)

# Charger le fichier JSON pour vérifier que l'enregistrement a bien fonctionné
# with open('users.json', 'r') as f:
#     loaded_users = json.load(f)

# Afficher les utilisateurs chargés depuis le fichier JSON
# print("\nUtilisateurs chargés depuis 'users.json':")