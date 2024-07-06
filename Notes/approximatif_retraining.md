# Réentraînement Approximatif

Le réentraînement approximatif dans le contexte de l'apprentissage machine, et plus spécifiquement dans le cadre du désapprentissage, consiste à ajuster les paramètres du modèle pour inverser ou atténuer l'effet de certains échantillons de données sur lesquels le modèle a été formé.

## Formulation Mathématique

### 1. Calcul de la Différence des Gradients (Gradient Difference)

Soit $\theta$ les paramètres actuels du modèle. Pour un ensemble de données d'origine $(x, y)$ et un ensemble de données modifié $(x', y')$, la différence des gradients est calculée comme suit :

$
\Delta g = \nabla_\theta \mathcal{L}(x, y; \theta) - \nabla_\theta \mathcal{L}(x', y'; \theta)
$

où $\mathcal{L}$ représente la fonction de perte.

### 2. Mise à Jour de Premier Ordre (First-Order Update)

Pour une mise à jour de premier ordre, la nouvelle estimation des paramètres $\theta_{\text{new}}$ est obtenue directement à partir de la différence des gradients :

$
\theta_{\text{new}} = \theta - \tau \Delta g
$

où $\tau$ est le taux de désapprentissage.

### 3. Mise à Jour de Deuxième Ordre (Second-Order Update)

Pour une mise à jour de deuxième ordre, nous devons approximer l'inverse du produit Hessien-vecteur. L'algorithme LiSSA ou les gradients conjugués peuvent être utilisés pour cela :

$
\Delta \theta \approx H^{-1} \Delta g
$

où $H$ est la matrice Hessienne (la matrice des secondes dérivées de la fonction de perte par rapport aux paramètres).

Ensuite, la mise à jour des paramètres est donnée par :

$
\theta_{\text{new}} = \theta - \tau \Delta \theta
$

### 4. Algorithme LiSSA

L'algorithme LiSSA (Linear-time Subsampled Saddle-point Approximation) est utilisé pour approximer $ H^{-1} \Delta g $ :

$
\Delta \theta = \Delta g + \frac{1}{m} \sum_{i=1}^{m} \left( (I - \alpha H)^{k_i} \Delta g \right)
$

où $m$ est le nombre d'itérations, $\alpha$ est un facteur de régularisation, et $k_i$ est le nombre de sous-échantillonnages.

## Algorithme Complet

1. Calculer la différence des gradients $\Delta g$.
2. Choisir la méthode de mise à jour :
   - Pour le premier ordre : $\Delta \theta = \Delta g$.
   - Pour le deuxième ordre : utiliser LiSSA ou les gradients conjugués pour approximer $ H^{-1} \Delta g $.
3. Mettre à jour les paramètres du modèle :

$
\theta_{\text{new}} = \theta - \tau \Delta \theta
$

4. Répéter les étapes ci-dessus pour chaque itération de réentraînement approximatif.

En résumé, le réentraînement approximatif vise à ajuster les paramètres du modèle en utilisant des approximations des gradients et des produits Hessien-vecteur pour inverser les effets indésirables des échantillons de données modifiés, tout en assurant que le modèle reste efficace et précis.
