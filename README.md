# Projet d'Analyse de Données sur le Dataset OpenFoodFacts

Ce projet a pour objectif d'explorer et d'analyser les données issues du dataset OpenFoodFacts, une base de données collaborative contenant des informations nutritionnelles et diverses métadonnées sur des milliers de produits alimentaires à travers le monde.

## Participants

- **Matéo**, responsable de la **mise en forme des données (data cleaning)**:
  - Nettoyé et filtré le dataset initial.
  - Supprimé certaines colonnes jugées peu pertinentes ou trop incomplètes.
  - Fragmenté le jeu de données en sous-ensembles plus légers afin d'accélérer les traitements ultérieurs.
  - Effectué les premiers tests sur **HDBSCAN** dans le but de comprendre son fonctionnement.

- **Margot**, en charge de **l'analyse des données** à l'aide de plusieurs approches de machine learning :
  - Application de **K-Means** pour identifier des groupes de produits similaires.
  - Utilisation de **HDBSCAN** pour une segmentation plus fine et plus flexible.
  - Expérimentation avec des **autoencodeurs** pour détecter des motifs complexes ou atypiques dans les données.
  - Assemblage général du code pour créer le notebook et nettoyage global du projet.

## Objectifs

- Comprendre les relations entre différents attributs des produits alimentaires.
- Identifier des clusters représentatifs de types de produits ou de profils nutritionnels.
- Explorer les capacités des modèles non supervisés pour la classification et la détection d'anomalies.
