import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
import pandas as pd

def content_based_recommender(
        data, tfidf_matrix, user_profiles, user_id,
        n_recommendations=5, similarity_metric="cosine", similarity_threshold=0.7):
    """
    Fonction de recommandation basée sur le contenu retournant un dictionnaire formaté.

    Parameters:
    - data : DataFrame contenant les informations sur les produits et notes
    - tfidf_matrix : DataFrame des vecteurs TF-IDF pour chaque produit
    - user_profiles : DataFrame des profils utilisateurs
    - user_id : identifiant utilisateur
    - n_recommendations : nombre de recommandations par utilisateur
    - similarity_metric : la métrique de similarité ("cosine", "euclidean", "manhattan")
    - similarity_threshold : seuil de similarité pour retenir les recommandations

    Returns:
    - list : Liste de dictionnaires contenant les recommandations par utilisateur
    """
    result = []
    if user_id not in user_profiles.index:
        print("L'utilisateur n'a pas suffisamment de données d'historique pour établir un profil.")
        return None, None

    # Sélection du profil de l'utilisateur
    user_profile = user_profiles.loc[user_id].values.reshape(1, -1)

    # Choix de la fonction de similarité
    if similarity_metric == "cosine":
        similarity_func = cosine_similarity
    elif similarity_metric == "euclidean":
        similarity_func = lambda x, y: -euclidean_distances(x, y)  # inverser pour que la distance négative simule la similarité
    elif similarity_metric == "manhattan":
        similarity_func = lambda x, y: -manhattan_distances(x, y)
    else:
        raise ValueError("Métrique de similarité non supportée. Utilisez 'cosine', 'euclidean' ou 'manhattan'.")

    # Filtrer les produits vus par l'utilisateur
    seen_product_ids = data[data['user_id'] == user_id]['parent_asin'].unique()
    seen_products = tfidf_matrix[tfidf_matrix.index.isin(seen_product_ids)]

    # Gestion des NaN pour les produits vus
    if seen_products.isna().any().any():
        seen_products = seen_products.fillna(0)

    # Calcul de la similarité pour les produits vus
    seen_similarity_scores = similarity_func(user_profile, seen_products.values).flatten()

    # Créer un DataFrame pour les produits vus et leurs scores
    seen_products_scores = pd.DataFrame({
        'parent_asin': seen_products.index,
        'similarity_score': seen_similarity_scores
    }).sort_values(by='similarity_score', ascending=False)

    # Filtrer les produits non vus par l'utilisateur
    unseen_products = tfidf_matrix[~tfidf_matrix.index.isin(seen_product_ids)]

    # Gestion des NaN pour les produits non vus
    if unseen_products.isna().any().any():
        unseen_products = unseen_products.fillna(0)

    # Calcul des similarités pour les produits non vus
    #print("Calcul des similarités pour recommander des produits non vus.")
    similarity_scores = similarity_func(user_profile, unseen_products.values).flatten()

    # Créer un DataFrame avec les scores de similarité des produits non vus
    recommendations = pd.DataFrame({
        'parent_asin': unseen_products.index,
        'similarity_score': similarity_scores
    })

    # Appliquer le seuil de similarité
    recommendations = recommendations[recommendations['similarity_score'] >= similarity_threshold]

    # Joindre les informations des produits
    product_info = data[['parent_asin']].drop_duplicates()
    recommendations = recommendations.merge(product_info, on='parent_asin', how='left')

    # Trier et sélectionner les n meilleures recommandations
    top_recommendations = recommendations.sort_values(
        by='similarity_score',
        ascending=False
    ).head(n_recommendations)

    # Ajouter les liens d'images
    images_df = data[['parent_asin', 'large_image_url','product_title']].drop_duplicates()
    top_recommendations = pd.merge(top_recommendations, images_df, on='parent_asin', how='left')
    seen_products_scores = pd.merge(seen_products_scores, images_df, on='parent_asin', how='left')


    # Création du dictionnaire pour l'utilisateur
    user_dict = {
        "id": user_id,
        "name": f"Utilisateur {user_id}",
        "boughtProducts": [
            {
                "name": row['product_title'],
                "image": row['large_image_url'],
                "similarity": float(row['similarity_score'])
            }
            for _, row in seen_products_scores.iterrows()
        ],
        "recommendedProducts": [
            {
                "name": row['product_title'],
                "image": row['large_image_url'],
                "similarity": float(row['similarity_score'])
            }
            for _, row in top_recommendations.iterrows()
        ]
    }

    result.append(user_dict)
    return result


