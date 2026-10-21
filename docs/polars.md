# Mini-cours Polars en Python

Polars est une librairie Python permettant de manipuler des données sous forme de DataFrame, un peu comme Pandas, mais avec un moteur optimisé pour les performances.

On va voir les opérations essentielles : `select()`, `filter()`, `group_by()`, `sum()`, `len()`, `sort()` et quelques autres.

## 1. Créer un DataFrame

```
import polars as pldf = pl.DataFrame({    "produit": ["Pomme", "Banane", "Pomme", "Orange", "Banane"],    "categorie": ["Fruit", "Fruit", "Fruit", "Fruit", "Fruit"],    "quantite": [10, 5, 15, 8, 12],    "prix": [2.0, 1.5, 2.0, 3.0, 1.5]})print(df)
```

| produit | categorie | quantite | prix |
| ------- | --------- | -------- | ---- |
| Pomme   | Fruit     | 10       | 2.0  |
| Banane  | Fruit     | 5        | 1.5  |
| Pomme   | Fruit     | 15       | 2.0  |
| Orange  | Fruit     | 8        | 3.0  |
| Banane  | Fruit     | 12       | 1.5  |

## 2. Sélectionner des colonnes — `select()`

```
# Une colonnedf.select("produit")# Plusieurs colonnesdf.select("produit", "prix")# Avec expressiondf.select(    pl.col("produit"),    pl.col("prix") * 2)
```

`pl.col("prix")` signifie qu'on fait référence à la colonne `prix`.

## 3. Filtrer des données — `filter()`

```
# Quantité supérieure à 10df.filter(pl.col("quantite") > 10)# Plusieurs conditionsdf.filter(    (pl.col("quantite") > 10) &    (pl.col("prix") < 3))
```

Utilise `&` pour ET et `|` pour OU.

## 4. Regrouper des données — `group_by()`

C'est l'équivalent du `GROUP BY` en SQL.

```
df.group_by("produit").agg(    pl.col("quantite").sum())
```

Résultat (l'ordre des groupes peut varier) :

| produit | quantite |
| ------- | -------- |
| Pomme   | 25       |
| Banane  | 17       |
| Orange  | 8        |

Tu peux utiliser plusieurs colonnes :

```
df.group_by("categorie", "produit").agg(    pl.col("quantite").sum())
```

## 5. Compter le nombre de records — `pl.len()`

C'est l'équivalent de `COUNT(*)` en SQL.

```
df.group_by("produit").agg(    pl.len().alias("nombre_records"))
```

| produit | nombre_records |
| ------- | -------------- |
| Pomme   | 2              |
| Banane  | 2              |
| Orange  | 1              |

Pour compter toutes les lignes du DataFrame :

```
df.height# 5
```

Pour compter uniquement les valeurs non nulles d'une colonne :

```
df.select(pl.col("prix").count())
```

## 6. Faire plusieurs calculs dans un `group_by()`

Tu peux combiner plusieurs agrégations dans le même `agg()`.

```
result = df.group_by("produit").agg(    pl.len().alias("nb_ventes"),    pl.col("quantite").sum().alias("total_quantite"),    pl.col("prix").mean().alias("prix_moyen"),    pl.col("prix").max().alias("prix_max"))
```

| produit | nb_ventes | total_quantite | prix_moyen | prix_max |
| ------- | --------- | -------------- | ---------- | -------- |
| Pomme   | 2         | 25             | 2.0        | 2.0      |
| Banane  | 2         | 17             | 1.5        | 1.5      |
| Orange  | 1         | 8              | 3.0        | 3.0      |

## 7. Créer une colonne — `with_columns()`

```
df = df.with_columns(    (pl.col("quantite") * pl.col("prix")).alias("total"))
```

Cela ajoute une colonne `total`, sans supprimer les colonnes existantes.

Pour modifier une colonne :

```
df = df.with_columns(    (pl.col("prix") * 1.10).alias("prix"))
```

## 8. Trier les résultats — `sort()`

```
# Ordre croissantdf.sort("prix")# Ordre décroissantdf.sort("quantite", descending=True)
```

## 9. Autres opérations utiles

| Opération                    | Exemple                            |
| ---------------------------- | ---------------------------------- |
| 5 premières lignes           | `df.head(5)`                       |
| 5 dernières lignes           | `df.tail(5)`                       |
| Colonnes disponibles         | `df.columns`                       |
| Types des colonnes           | `df.schema`                        |
| Nombre de lignes             | `df.height`                        |
| Somme d'une colonne          | `df.select(pl.col("prix").sum())`  |
| Moyenne                      | `df.select(pl.col("prix").mean())` |
| Valeurs uniques              | `df.select("produit").unique()`    |
| Supprimer une colonne        | `df.drop("categorie")`             |
| Renommer une colonne         | `df.rename({"prix": "price"})`     |
| Supprimer les doublons       | `df.unique()`                      |
| Remplacer les valeurs nulles | `df.fill_null(0)`                  |

## 10. Exemple complet : analyser les ventes

Voici un exemple qui combine tout ce qu'on vient de voir.

```
result = (    df    .filter(pl.col("quantite") >= 5)    .with_columns(        (pl.col("quantite") * pl.col("prix")).alias("total")    )    .group_by("produit")    .agg(        pl.len().alias("nb_ventes"),        pl.col("quantite").sum().alias("quantite_totale"),        pl.col("total").sum().alias("revenu")    )    .sort("revenu", descending=True))print(result)
```

Résultat :

| produit | nb_ventes | quantite_totale | revenu |
| ------- | --------- | --------------- | ------ |
| Pomme   | 2         | 25              | 50.0   |
| Banane  | 2         | 17              | 25.5   |
| Orange  | 1         | 8               | 24.0   |

## 11. Bonus : `LazyFrame`

Une fonctionnalité importante de Polars est l'exécution lazy. Tu construis une requête et Polars optimise son exécution avant de calculer les résultats.

```
result = (    pl.scan_csv("ventes.csv")    .filter(pl.col("quantite") > 10)    .group_by("produit")    .agg(        pl.col("quantite").sum()    )    .collect())
```

`collect()` déclenche l'exécution.

## À retenir

- `select()` : choisir des colonnes.
- `filter()` : filtrer des lignes.
- `with_columns()` : créer ou modifier des colonnes.
- `group_by()` : regrouper des lignes.
- `agg()` : appliquer des agrégations.
- `pl.len()` : compter les lignes.
- `sum()` : calculer une somme.
- `alias()` : renommer le résultat d'une expression.
- `sort()` : trier les résultats.

La différence importante avec SQL : dans Polars, tu peux enchaîner les opérations pour construire progressivement ton traitement de données.

Documentation officielle : [Polars – Getting Started](https://docs.pola.rs/user-guide/getting-started/?utm_source=chatgpt.com).&#x20;

[image](https://www.google.com/s2/favicons?domain=https://docs.pola.rs\&sz=32)

Polars user guide

+1
