# Analyse de séries temporelles — Portfolio Data Science

## Objectif
Ce projet rassemble différents travaux pratiques autour de l’analyse de séries temporelles en Python :  
- manipulation et visualisation de données temporelles,  
- détection de saisonnalité et tendance,  
- analyse spectrale et périodogrammes,  
- modélisation AR/ARIMA,  
- prédiction et évaluation des performances.

## Jeux de données étudiés et les points abordés

### 1. Syndromes grippaux (*Openhealth_S-Grippal.csv*)
- Construction d’une série temporelle sur la variable **IAS_brut**  
- Gestion des valeurs manquantes par imputation  
- Visualisation et transformations (logarithme)  
- Analyse spectrale avec le périodogramme pour la saisonnalité
- Filtrage d’une saisonnalité annuelle  
- Prédiction basée sur une tendance périodique (harmoniques cosinus/sinus)

### 2. Trafic Internet (*lbl-tcp-3.tcp*)
- Construction d’une série temporelle (paquets par intervalle de 10 secondes)  
- Estimation d’un modèle auto-régressif **AR(p)**  
- Sélection de l’ordre p (AIC, BIC, validation croisée)  
- Analyse des résidus et comparaison à une loi normale
- Test et simulation en streaming  

### 3. Southern Oscillation Index (*soi.tsv*)
- Nettoyage et traitement des données manquantes  
- Analyse des fonctions d’autocorrélation (ACF, PACF)  
- Estimation d’un modèle AR(p) et validation via les résidus  
- Comparaison périodogramme vs densité spectrale théorique  

## Remarques
- Les fichiers de données volumineux ou sensibles sont exclus du dépôt via `.gitignore`.  
- Pour exécuter les notebooks et scripts :  
  ```bash
  pip install numpy scipy pandas statsmodels matplotlib scikit-learn

## Auteur

Pety Ialimita RAKOTONIAINA  
[GitHub](https://github.com/MitaDataAI)  
[LinkedIn](https://www.linkedin.com/in/pety-ialimita-rakotoniaina-341583171)
