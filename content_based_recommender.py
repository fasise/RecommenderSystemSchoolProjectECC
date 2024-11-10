import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def content_based_recommender(data, tfidf_matrix, user_profiles, user_id, n_recommendations=10):
    """
    Fonction de recommandation basée sur le contenu retournant un dictionnaire formaté.
    
    Parameters:
    - data : DataFrame contenant les informations sur les produits et notes
    - tfidf_matrix : DataFrame des vecteurs TF-IDF pour chaque produit
    - user_profiles : DataFrame des profils utilisateurs
    - user_ids : liste des identifiants utilisateurs
    - n_recommendations : nombre de recommandations par utilisateur
    
    Returns:
    - list : Liste de dictionnaires contenant les recommandations par utilisateur
    """
    result = []
    if user_id not in user_profiles.index:
        print("L'utilisateur n'a pas suffisamment de données d'historique pour établir un profil.")
        return None, None

    # Profil de l'utilisateur cible
    user_profile = user_profiles.loc[user_id].values.reshape(1, -1)
    
    # 1. Filtrer les produits vus par l'utilisateur
    seen_product_ids = data[data['user_id'] == user_id]['product_id'].unique()
    seen_products = tfidf_matrix[tfidf_matrix.index.isin(seen_product_ids)]
    
    if seen_products.isna().any().any():
        seen_products = seen_products.fillna(0)
    
    # Calcul des similarités cosinus pour les produits vus
    seen_similarity_scores = cosine_similarity(user_profile, seen_products.values).flatten()
    
    # Créer un DataFrame avec les produits vus et leurs scores
    seen_products_scores = pd.DataFrame({
        'product_id': seen_products.index,
        'similarity_score': seen_similarity_scores
    }).sort_values(by='similarity_score', ascending=False)
    
    # 2. Filtrer les produits non vus par l'utilisateur
    unseen_products = tfidf_matrix[~tfidf_matrix.index.isin(seen_product_ids)]
    
    if unseen_products.isna().any().any():
        unseen_products = unseen_products.fillna(0)
    
    # Calcul des similarités cosinus pour les produits non vus
    print("Calcul des similarités pour recommander des produits non vus.")
    similarity_scores = cosine_similarity(user_profile, unseen_products.values).flatten()
    
    # Créer un DataFrame avec les produits non vus et leurs scores
    recommendations = pd.DataFrame({
        'product_id': unseen_products.index,
        'similarity_score': similarity_scores
    })
    
    # Joindre les informations des produits
    product_info = data[['product_id']].drop_duplicates()
    recommendations = recommendations.merge(product_info, on='product_id', how='left')
    
    # Trier et sélectionner les n meilleures recommandations
    top_recommendations = recommendations.sort_values(
        by='similarity_score', 
        ascending=False
    ).head(n_recommendations)

    # Ajouter les liens d'images
    img_link_df = data[['product_id', 'img_link']].drop_duplicates()
    top_recommendations = pd.merge(top_recommendations, img_link_df, on='product_id', how='left')
    seen_products_scores = pd.merge(seen_products_scores, img_link_df, on='product_id', how='left')
    
    # # Afficher les recommandations et les produits vus
    # print("\nTop", n_recommendations, "recommandations pour l'utilisateur", user_id, ":")
    # for idx, row in top_recommendations.iterrows():
    #     print(f"\nProduit {row['product_id']}")
    #     print(f"Score de similarité : {row['similarity_score']:.4f}")
    #     print(f"Lien de l'image : {row['img_link']}")
    
    # print("\nProduits déjà vus par l'utilisateur", user_id, "avec leurs scores de similarité :")
    # for idx, row in seen_products_scores.iterrows():
    #     print(f"\nProduit {row['product_id']}")
    #     print(f"Score de similarité : {row['similarity_score']:.4f}")
    #     print(f"Lien de l'image : {row['img_link']}")

        # Création du dictionnaire pour l'utilisateur
    user_dict = {
        "id": user_id,
        "name": f"Utilisateur {user_id}",
        "boughtProducts": [
            {
                "name": row['product_id'],
                "image": row['img_link'],
                "similarity": float(row['similarity_score'])
            }
            for _, row in seen_products_scores.iterrows()
        ],
        "recommendedProducts": [
            {
                "name": row['product_id'],
                "image": row['img_link'],
                "similarity": float(row['similarity_score'])
            }
            for _, row in top_recommendations.iterrows()
        ]
    }
    
    result.append(user_dict)
    
    return result
    #return top_recommendations, seen_products_scores


# top_recommendations, seen_products_scores = content_based_recommender(
#     dfc,
#     tfidf_clean, 
#     user_profiles_normalized, 
#     user_id
# )
