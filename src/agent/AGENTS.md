## Architecture

- **`agents/`** : Définit les agents IA, leurs instructions et les outils auxquels ils ont accès.
- **`tools/`** : Contient les fonctions que les agents peuvent appeler pour effectuer des actions.
- **`services/`** : Gère la logique métier, les appels aux API externes et les interactions avec les bases de données.
- **`models/`** : Définit les structures de données avec Pydantic.
- **`workflows/`** : Coordonne les différentes étapes d'un processus et l'exécution de plusieurs agents.
- **`main.py`** : Point d'entrée de l'application qui initialise et exécute les agents.

### Principe de fonctionnement

L'**agent** analyse la demande et choisit un **tool** approprié. Le tool fait appel à un **service** pour exécuter l'opération et retourne le résultat à l'agent.

Cette séparation facilite les tests, la réutilisation du code et l'ajout de nouvelles fonctionnalités sans modifier l'ensemble de l'application.

**Objectif :** garder les agents responsables des décisions et les services responsables de l'exécution de la logique métier.