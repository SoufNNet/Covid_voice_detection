# Détection du COVID par Analyse Audio Multimodale

## Contexte
Ce projet utilise une approche multimodale en intelligence artificielle pour détecter le COVID-19. Le système analyse simultanément plusieurs types d'enregistrements audio d'une même personne pour une détection plus robuste.

## Dataset Coswara
Le projet utilise le dataset Coswara, disponible sur Kaggle :
[Coswara Dataset - COVID-19 Detection](https://www.kaggle.com/datasets/janashreeananthan/coswara)

Ce dataset contient des enregistrements audio collectés via une application web, incluant des échantillons de personnes saines et atteintes du COVID-19.

### Structure des Données
Le dataset contient des enregistrements de participants du monde entier, avec :
- Des métadonnées (âge, sexe, état de santé)
- Des enregistrements audio standardisés

### Approche Multimodale
Le modèle combine trois modalités audio différentes :
- Toux (2 enregistrements)
- Respiration (2 enregistrements)
- Voix (3 enregistrements)

Pour chaque modalité, deux types de caractéristiques sont extraites :
- Spectrogrammes Mel
- Coefficients MFCC (Mel-frequency cepstral coefficients)

Cette fusion multimodale permet une analyse plus complète des signes potentiels du COVID-19.

## Classification
Le modèle classifie les personnes en deux catégories :
- Saines (incluant les personnes guéries)
- Atteintes du COVID-19 (cas légers, modérés et asymptomatiques)

## Contenu du Code

### Traitement des Données
- Extraction des fichiers audio depuis des archives TAR
- Traitement de 100 échantillons par classe pour garantir un équilibre
- Extraction parallèle des caractéristiques de chaque modalité

### Architecture Multimodale
- Réseau de neurones convolutif (CNN) avec branches parallèles
- Pour chaque modalité :
  - Traitement séparé des spectrogrammes Mel et MFCC
  - Couches de convolution 2D
  - Normalisation par lots (BatchNormalization)
  - Pooling
- Fusion des caractéristiques
- Couches denses finales pour la classification
- Entraînement sur 20 époques avec early stopping

### Résultats Générés
Le code produit :
- Matrice de confusion
- Rapport de classification
- Graphiques d'entraînement (accuracy et loss)
- Sauvegarde du modèle entraîné

## Dépendances Principales
- tensorflow
- librosa (pour le traitement audio)
- pandas
- numpy
- scikit-learn
- matplotlib et seaborn (pour les visualisations)

## Auteur
[Votre nom]