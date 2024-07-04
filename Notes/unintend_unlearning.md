## Build the Model
## Inject Canary Sequence
    Explanation:
    Injection Point: The canary is injected in the middle of the sample_text. You can change the injection_point to 0 (beginning) or len(sample_text) (end) or any other position as needed.

    Updating the Training Data: The sample_text_with_canary now contains the injected canary, and the subsequent steps prepare the training sequences from this modified text.

    This approach allows you to include the canary sequence in your training data, after which you can train your model as usual.
## Train model

## Generate Text
    First, we need to generate text sequences from the trained model and evaluate if the canary sequence appears in the generated text

## Evaluate Unlearning Techniques

## Measure the Exposure of Sensitive Data

## Analyze Perplexity Distribution
L'intérêt d'utiliser les fonctions calc_perplexity_distribution et approx_exposure réside dans leur capacité à mesurer et à évaluer la mémorisation non intentionnelle dans les modèles de langage. Voici les raisons principales :

1. Mesure de la Mémorisation Non Intentionnelle :
La mémorisation non intentionnelle se produit lorsqu'un modèle de langage mémorise des informations sensibles ou spécifiques des données d'entraînement, telles que des numéros de téléphone, des adresses, etc., au lieu d'apprendre des structures linguistiques générales. Cela peut entraîner des fuites de données si le modèle est utilisé pour générer du texte.

2. Distribution de Perplexité (calc_perplexity_distribution) :
Cette fonction calcule la distribution des perplexités pour un grand nombre de séquences aléatoires. La perplexité est une mesure de la prévisibilité d'une séquence : plus la perplexité est basse, plus la séquence est prévisible. En calculant cette distribution, vous pouvez :

Évaluer la prévisibilité des séquences générées par le modèle.
Identifier des anomalies dans les séquences mémorisées.
Comparer la perplexité des canaris avec celle des séquences aléatoires pour voir si les canaris sont particulièrement mémorisés.
3. Exposition Approximative (approx_exposure) :
Cette fonction utilise la distribution de perplexité pour calculer l'exposition approximative de séquences spécifiques. L'exposition mesure à quel point une séquence est mémorisée par rapport à toutes les séquences possibles.

Quantification de la Mémorisation : En calculant l'exposition, vous pouvez quantifier à quel point le modèle mémorise les séquences spécifiques (comme les canaris).
Évaluation de la Sécurité des Données : Si une séquence a une exposition élevée, cela signifie qu'elle est plus mémorisée que la plupart des autres séquences, ce qui peut indiquer une fuite de données sensibles.
