# Smart Attendance System

J’ai développé ce projet de vision par ordinateur pour automatiser la prise de présence dans une classe virtuelle. À partir d’une image contenant plusieurs étudiants, mon application détecte les visages, reconnaît les identités suffisamment fiables et prépare les listes des présents, des absents et des personnes à vérifier.

Mon jeu de démonstration contient **25 étudiants fictifs**. Les portraits sont synthétiques et le projet fonctionne sur **CPU**, avec une interface de bureau et des exports CSV.

## Fonctionnalités

- Analyse du dataset : images illisibles, flou, exposition, taille des visages et quasi-doublons.
- Alignement des visages et redimensionnement sans déformation.
- Séparation entraînement / validation / test avant augmentation.
- Détection avec **YuNet**, traitement par zones des grandes images et suppression des détections en double.
- Reconnaissance **FaceNet512**, avec plusieurs références par étudiant, seuils calibrés et rejet des correspondances incertaines.
- Génération de classes différentes : présents, positions, portraits, éclairage et orientation.
- Gestion des présences et absences avec identifiants stables, date et heure.
- Interface Tkinter : inspection, navigation entre les séances et affichage des images en grand.
- Comparaison avec **LBPH** et génération de rapports d’évaluation.

## Technologies utilisées

| Composant | Technologie |
|---|---|
| Langage | Python 3.10 |
| Traitement d’images | OpenCV, NumPy |
| Détection | YuNet via DeepFace |
| Reconnaissance | FaceNet512 via DeepFace |
| Exécution CPU | ONNX Runtime |
| Modèle de comparaison | LBPH |
| Données et exports | Pandas |
| Graphiques | Matplotlib |
| Interface | Tkinter, Pillow |

J’utilise FaceNet512 préentraîné pour représenter chaque visage par un vecteur de **512 valeurs**. La reconnaissance compare ce vecteur aux références enregistrées avec une distance cosinus. Une petite distance indique une plus grande similarité ; cette distance n’est pas un pourcentage de confiance.

## Organisation du projet

```text
Smart_Attendance_System/
├── app.py                      # Interface et commandes principales
├── students.json               # Registre des 25 identités fictives
├── requirements.txt
├── src/
│   ├── dataset_manager.py       # Qualité, doublons, préparation, augmentation
│   ├── face_detector.py         # Détection, alignement et filtrage
│   ├── face_embeddings.py       # Extraction et cache FaceNet512
│   ├── face_recognizer.py       # Galerie, comparaison et rejet
│   ├── calibration.py           # Calibration sur la validation
│   ├── classroom_generator.py   # Simulation des séances
│   ├── session_runner.py        # Reconnaissance et comparaison facultative
│   ├── attendance_manager.py    # Présences, absences et exports
│   ├── evaluation.py            # Métriques et comparaison avec LBPH
│   └── ui.py                    # Interface de bureau
├── tests/
└── README.md
```

L’application crée aussi des dossiers locaux : `processed_dataset/`, `test_dataset/`, `embeddings/`, `sessions/`, `attendance_reports/` et `reports/`. Les scripts de mes premières expérimentations OpenCV/LBPH sont conservés à la racine ; le point d’entrée principal est `app.py`.

## Installation sous Windows

```powershell
git clone https://github.com/oumaymazerrik/Smart_Attendance_System.git
Set-Location Smart_Attendance_System
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Tkinter doit être inclus dans l’installation de Python. Aucun GPU ni Docker n’est nécessaire. DeepFace télécharge les poids nécessaires au premier usage, puis les réutilise depuis son cache local.

Il faut installer une seule distribution d’OpenCV. Si un ancien environnement contient plusieurs variantes incompatibles, je le normalise ainsi :

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python-headless opencv-contrib-python
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Préparer les portraits

**Les portraits, les embeddings, les poids et les rapports de présence ne sont pas publiés dans ce dépôt.** Après clonage, il faut fournir un dataset synthétique local avant de préparer la galerie.

Je range les portraits dans des dossiers correspondant aux noms du registre `students.json` :

```text
dataset/
├── Adam/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── Amine/
├── Lina/
├── Sara/
├── Yassine/
└── ...
```

Chaque image doit contenir un seul visage exploitable. Je recommande au moins cinq portraits distincts par identité. Après filtrage, la reconnaissance exige au minimum deux références d’entraînement, une image de validation et une image de test par étudiant.

Les originaux restent dans `dataset/`. Les augmentations sont réservées à l’entraînement. Les versions augmentées d’un même portrait ne sont pas réparties entre entraînement et test. Quand j’ajoute de nouveaux étudiants, les répartitions des anciens sont conservées si leurs sources et le seed de préparation n’ont pas changé.

```powershell
.\.venv\Scripts\python.exe app.py prepare --seed 42
.\.venv\Scripts\python.exe app.py build
```

La première commande produit les contrôles qualité et les visages préparés. La seconde construit la galerie d’embeddings, calibre les seuils et recherche les identités trop similaires.

## Utilisation

```powershell
.\.venv\Scripts\python.exe app.py gui
```

Dans l’interface, je peux choisir le nombre d’étudiants et le **seed**, cliquer sur **Generate**, puis **Recognize**, et consulter les onglets **Tous**, **Présents**, **Absents confirmés** et **À vérifier**. **Inspect** explique les décisions ; un clic sur l’image ouvre la pleine résolution ; **Save CSV** enregistre les listes. **Compare truth** compare les résultats à la simulation, et **Previous / Next** permettent de revoir les séances précédentes.

Le seed sert à reproduire une simulation : à paramètres et données identiques, un même seed reproduit la même classe. Il ne modifie pas le modèle.

Une séance peut aussi être exécutée en ligne de commande :

```powershell
$session = .\.venv\Scripts\python.exe app.py generate --count 20 --seed 42 | ConvertFrom-Json
.\.venv\Scripts\python.exe app.py recognize $session --compare
```

`--count 0` crée une classe vide. Sans `--count`, le nombre de présents est tiré au hasard. Le mode par défaut utilise les portraits de test ; `--split train` sert aux démonstrations avec des portraits d’entraînement et ne constitue pas une évaluation indépendante. `--keep-open` conserve une séance provisoire.

## Présences et absences

Je distingue le nombre d’absences prévu par la simulation des identités confirmées par le modèle.

**Exemple : 25 inscrits − 20 personnes générées = 5 absences prévues.** Ce compteur apparaît directement dans l’interface. Les noms des absents confirmés proviennent des prédictions, pas des étiquettes de simulation.

Une image de classe générée est automatiquement finalisée lorsque les comptes de visages attendus et détectés correspondent, avec le même registre et sans erreur d’inférence. Ce contrôle de couverture utilise uniquement les comptes de `simulation.json`. Les autres cas demandent une vérification manuelle. Les visages inconnus ou ambigus restent à vérifier.

Les portraits actuels de **Sara et Lina** sont trop proches pour une identification fiable. Une séance peut ainsi annoncer cinq absences prévues, mais seulement quatre noms d’absents confirmés et deux étudiants à vérifier : l’un de ces deux peut être présent. Je conserve cette incertitude au lieu de forcer un nom.

Chaque séance produit ses propres fichiers dans `attendance_reports/` :

| Fichier | Contenu |
|---|---|
| `<session>.csv` | Registre complet |
| `<session>_absents.csv` | Absents confirmés |
| `<session>_a_verifier.csv` | Étudiants à vérifier |
| `<session>_non_reconnus.csv` | Tous les étudiants non reconnus présents, y compris les cas incertains |
| `<session>.json` | Compteurs et conditions de finalisation |

Les CSV contiennent `Student_ID`, `Name`, `Date`, `Time`, `Status` et `Distance`. Une nouvelle sauvegarde de la même séance met à jour ses fichiers sans dupliquer les étudiants. Les horaires utilisent le fuseau `Africa/Casablanca`.

## Évaluation

```powershell
.\.venv\Scripts\python.exe app.py evaluate
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.absence_validation
```

Ces vérifications nécessitent le dataset local et les sorties de préparation/calibration. L’évaluation compare FaceNet512 et un nouveau modèle LBPH sur les mêmes images réservées au test. Les métriques, prédictions et matrices de confusion sont enregistrées dans `reports/`.

### Résultats de ma démonstration synthétique

Mon jeu local contient 145 portraits pour 25 identités. Après filtrage, il reste 132 groupes de sources : 82 pour l’entraînement, 25 pour la validation et 25 pour le test. L’augmentation produit 410 images d’entraînement au total.

| Mesure sur 25 portraits de test | FaceNet512 | LBPH |
|---|---:|---:|
| Identités correctement reconnues | 23/25 | 10/25 |
| Exactitude | 92 % | 40 % |
| Précision macro | 92 % | 40 % |
| Rappel macro | 92 % | 40 % |
| F1-score macro | 92 % | 40 % |

Dans la classe complète testée, les 25 visages ont été détectés : 23 étudiants ont été correctement reconnus, tandis que Sara et Lina sont restées à vérifier. Les exports contenaient 25 identifiants uniques. Les 12 tests locaux passent, dont un test de l’interface pour le cas « 20 présents identifiés, 5 absents ».

### Limites

Ces chiffres décrivent un petit jeu de démonstration synthétique. Plusieurs vues d’un même nouvel étudiant ont été produites ensemble : elles sont corrélées et ne représentent pas des prises de vue indépendantes. Ces résultats ne suffisent donc pas à établir une précision de 92 % sur une vraie classe.

Le taux de fausse acceptation de personnes inconnues reste non mesuré, faute d’identités inconnues indépendantes dans le jeu actuel. Le programme accepte un dossier local `unknown_dataset/` pour cette évaluation. Les profils prononcés, l’occlusion et les petits visages restent également des difficultés.

Pour améliorer la fiabilité, je dois compléter les références, remplacer les portraits trop similaires et évaluer le système sur davantage d’images indépendantes.

## Auteur

[oumaymazerrik](https://github.com/oumaymazerrik)
