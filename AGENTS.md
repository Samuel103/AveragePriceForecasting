# Présentation du projet

Ce projet vise à prévoir le prix moyen mensuel de produits agroalimentaires et à
fournir un agent IA capable de répondre aux questions sur les prix historiques
et les prévisions.

## Structure du dépôt

Le code est organisé en deux composants dans le dossier `src/` :

- `src/agent/` : composant destiné à l'agent IA utilisant OpenAI pour répondre
  aux questions sur les prix et les prévisions.
- `src/forecasting/` : composant de prévision utilisant LightGBM pour estimer
  le prix moyen de chaque produit pour les prochains mois. La préparation des
  données, l'entraînement, l'évaluation et la génération des prévisions sont
  dans `src/forecasting/`. Le point d'entrée est `src/forecasting/main.py`,
  accessible via `uv run forecast` depuis la racine. Les deux composants
  partagent le projet uv principal et son environnement `.venv`.

## Données

Le composant de prévision utilise :

- `src/forecasting/data/raw/18100245.csv` : l'historique des prix source.
- `src/forecasting/data/predictions/forecasts.json` : les prévisions exportées,
  avec un enregistrement par produit et par mois (`PRODUCTS`, `DATE`, `VALUE`).
- `src/forecasting/models/forecast_model.pkl` : le modèle final et son vocabulaire
  de produits.

Le dossier `data/` à la racine conserve les anciens fichiers; le nouveau pipeline
n'y exporte plus ses prévisions.

Les chemins ci-dessus sont relatifs à la racine du dépôt.

## Portée des instructions

Ce fichier fournit le contexte général du projet. Avant de modifier un composant,
consulter également son fichier `AGENTS.md` pour prendre en compte les
instructions propres à ce composant.
