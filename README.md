# Classification Produit à partir de Photos et de Descriptions Produit

*Projet 6 de la formation Data Scientist d'OpenClassrooms*

L'entreprise e-commerce Place de Marché souhaite réaliser un moteur de classification pour classer automatiquement les produits proposés sur son site web, à partir de leur description et/ou de leur photo. On démontre dans ce notebook la faisabilité d'une telle classification à partir d'outils d'intelligence artificielle.


Le jeu de données est constitué de 1050 produits de Place de Marché, avec un nom, une description et une photo associées. C'est à partir de ces données non structurées qu'est effectuée la classification des produits en catégories.

note : les photos produit ne sont pas accessibles sur ce projet GitHub, faute d'espace libre.

## Vectorisation des Données Non Structurées


Le projet contient deux fichiers dans lesquels est réalisée la vectorisation des données non structurées :

- [DataPreparationText.ipynb](https://nbviewer.jupyter.org/github/EloiLQ/product-NLP-CP/blob/main/DataPreparationText.ipynb), pour les données textuelles (nom et description produit). Les modèles NLP utilisés sont : [TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), [Word2Vec](https://code.google.com/archive/p/word2vec/) et [USE]((https://tfhub.dev/google/universal-sentence-encoder/1))

- [DataPreparationPictures.ipynb](https://nbviewer.jupyter.org/github/EloiLQ/product-NLP-CP/blob/main/DataPreparationPictures.ipynb), pour les données visuelles (photo produit). Les modèles CP utilisés sont SIFT et le réseau de neurones convolutif VGG16.


note : ce projet Git-hub ne contient pas les fichiers des encoder NLP [Word2Vec](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g) et [USE](https://tfhub.dev/google/universal-sentence-encoder/1) utilisés dans ce projet, faute de place.


## Classification des Produits


Une fois vectorisées, les descriptions et photos produit sont utilisées en entrée d'une classification. Une première classification non-supervisée permet de tester la faisabilité du projet. Une réduction dimensionnelle avec UMAP suivie d'un clustering avec K-means est appliqué. Ce premier résultat est finalement comparé à une classification supervisée, dans laquelle on utilise un classifieur SVM. Les trois notebooks suivant réalisent une classification :

- [ClassificationDesc.ipynb](https://nbviewer.jupyter.org/github/EloiLQ/product-NLP-CP/blob/main/ClassificationDesc.ipynb) : classification des produits selon leur description

- [ClassificationNames.ipynb](https://nbviewer.jupyter.org/github/EloiLQ/product-NLP-CP/blob/main/ClassificationNames.ipynb) : classification des produits selon leur nom

- [ClassificationImages.ipynb](https://nbviewer.jupyter.org/github/EloiLQ/product-NLP-CP/blob/main/ClassificationImages.ipynb) : classification des produits selon leur photo


## Résumé des Résultats

L'accuracy est utilisée comme métrique pour quantifier la qualité de la classification. Elle est calculée à partir des catégories estimées (en non-supervisé et en supervisé) et des catégories 'vraies' produit. Les catégories 'vraies' sont attribuées par une intelligence humaine, et sont au nombre de 7. Le jeu de donnée contient 150 produits par catégorie.


En non-supervisé, les meilleurs résultats sont obtenus à partir des noms produit, vectorisés avec le modèle NLP USE. L'accuracy atteint les 77 %. À partir des descriptions produit, les meilleurs résultats sont obtenus avec le modèle TF-IDF, avec une accuracy de 67 %. Avec les images, c'est avec le CNN VGG16 que l'on obtient les meilleurs résultats, avec une accuracy d'environ 48 %.


Enfin, en classification supervisée, les modèles NLP considérés obtiennent des performances similaires, légèrement au-dessus de 90 % (91 % pour les noms produits et 93 % pour les descriptions produit). Le modèle VGG16 obtient une accuracy de 81 % à partir des photos produit.
