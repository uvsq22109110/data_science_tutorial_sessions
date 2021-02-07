1. Word embedding est un ensemble de technique permettant de présenter les mots avec des nombres. Plus précisement ces méthodes permettent de relier (faire correspondre) des mots à des vecteurs dans un espace de dimension K. Ces techniques sont utilisées pour la génération des varaiables, le clustering des documents, la classification des textes... les Word emebedding permettent de : 

    - calculer les similarités entre les mots : Grâce a la projection d'un mot dans un espace de dimension K on peut calculer des distances de similarités pour proposer des synonymes ou des antonymes.  
    - Créer des groupes de mots similaires :  retrouver des mots qui se trouvent dans un contexte similaire
    - Génération de variables pour la classification : Comme les modèles ne peuvent pas s'e,traîner sur du texte brute, un mapping vers un espace vectoriel est nécessaire.
    - Clustering des documents : Retrouver des documents qui portent sur les mêmes topics et le regrouper
    - F.Y.I : Une technique de word embedding qui est très simple et naive c'est one hot encoding. L'inconvinient de cette technique c'est quelle ne permet pas de calculer des similarités bien comme il le faut et il contient beaucoup d'informations dans les vecteurs qui en sert à rien (valeur à 0)
        
2. Word2Vec est une technique de Word embedding. Il s'agit d'un réseau de neurones non profond, avec une seule couche cachée entre l'entrée et la sortie. Cette technique permet d'obtenir un mapping entre un vocabulaire et un espace vectoriel de dimension K. Il s'agit d'un problème de classification supervisé. Cette méthode est apparue en 2013 chez Google (papier de recherche Mikolov et al.).
Il existe 2 implémentations du Word2Vec
  
    - Continuous Bag of words (CBOW)
    - skip gram

<div style="text-align:center;">
    <img style="height:500px;width:700px" src="http://mccormickml.com/assets/word2vec/skip_gram_net_arch.png" />
</div>

3. Différence CBOW et Skip-gram
    - **CBOW (continuous Bag of words)**, le mot actuel est prédit en utilisant une fenêtre qui l'entoure (contexte). Par example, si wi-1,wi-2,wi+1,wi+2 sont les mots d'une fenêtre de taille 2 (contexte), CBOW va prédire le mot wi.
<div style="text-align:center;">
    <img style="height:500px;width:700px" src="https://www.researchgate.net/profile/Meenakshi_Tripathi4/publication/283161683/figure/fig4/AS:486973749108738@1493114997121/Architecture-of-continuous-bag-of-words-model-CBOW.png" />
</div>

    - **Skip-Gram** réalise l'opération invese de CBOW, c'est à dire il va prédire un contexte à partir d'un mot. On peut inveser l'exemple précédent pour mieux comprendre. Pour un mot wi, Skip-gram va prédire le contexte c'est à dire wi-1,wi-2,wi+1,wi+2.

<div style="text-align:center;">
        <img style="height:500px;width:700px" src="https://paperswithcode.com/media/methods/Screen_Shot_2020-05-26_at_2.04.55_PM.png" />
</div>


CBOW is plus rapide que skip gram et donne de meilleures fréquence pour les mots fréquents. Par contre; skip gram ne demande pas beaucoup d'exemple pour l'entrainement et est capable de présenter même les mots rares. Pour **Skip-gram**, il faut noter que des accélération du temps de calcul sont possible en utilisant **Ngative Sampling ou hierarchical softmax**

4. **Le stemming** et **la lemmatization** sont deux techniques de normalisation de text : 

    - **Le Stemming** réduit la variation dans les mots en supprimant le début ou la fin d'un mot. Il prend en considération une liste prédéfinie de préfixes et suffixes communément utilisés. Le **Stemmer** le plus répandu **Porter stemmer**.  
    - De l'autre côté, **la lemmatization** prend en considération l'analyse morphologique des mots. Pour ce faire, elle a besoin de **dictionnaires détaillés** où l'algorithme peut chercher dedans pour trouver une base pour le mot en question **(Lemma)**.

<div style="text-align:center;">
        <img style="height:500px;width:700px" src="https://miro.medium.com/max/2050/1*2K4VxxRtewNw4iP-Kh5Z7Q.png" />
</div>


5. See code below
6. See code below
7. Gensim Word2Vec prend en paramètres :  
    - size: (default 100) la dimesion de l'espace d'Encoding. (Longueur du vecteur correspondant à un mot du vocabulaire)
    - window: (default 5) le nombre de mot maximal à prendre en cosidération dans le contexte
    - min_count: (default 5) Le nombre mininmum de fréquence d'apparition d'un mot pour le prendre dans la phase de train (Permet d'élaguer certains mots très spécifiques)
    - workers: (default 3) le nombre de threads à utiliser pendant l'entrainement.
    - sg: (default 0 or CBOW) L'implémentation à utiliser, soit CBOW (0) ou skip gram (1).
8. See code below
9. We could see that all information returned by the model have meaning. Either we have clubs, or players..
``` Python
[('lebron', 0.6455210447311401)]
[('neymar', 0.6294288039207458), ('dud', 0.6268420815467834), ('hazard', 0.5691757798194885), ('fast', 0.5573602914810181), ('suarez', 0.5519919991493225), ('rate', 0.5231587290763855), ('stuck', 0.5230600833892822), ('zlatan would', 0.5185644030570984), ('pleas ronaldo', 0.5180506110191345), ('immedi', 0.5177865624427795)]
``` 
    
10. See code below
11. See code below
12. See code below
13. See code below