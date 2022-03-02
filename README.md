# mask-detector-covid

* [Github Pages](https://visiont3lab.github.io/mask-detector-covid/) https://visiont3lab.github.io/mask-detector-covid/


Parte 1)  Setup and Training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/visiont3lab/mask-detector-covid/blob/main/notebooks/Project_Covid_Mask_Classifier_Part1.ipynb)

Parte 2)  Face Detector [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/visiont3lab/mask-detector-covid/blob/main/notebooks/Project_Covid_Mask_Classifier_Part2.ipynb)

Parte 3)  Mask Classifier (SVM) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/visiont3lab/mask-detector-covid/blob/main/notebooks/Project_Covid_Mask_Classifier_Part3.ipynb)

Parte 4)  Cross validation + Fine tuning [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/visiont3lab/mask-detector-covid/blob/main/notebooks/Project_Covid_Mask_Classifier_Part4.ipynb)

Parte 5)  Comple Classification Project Pipeline [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/visiont3lab/mask-detector-covid/blob/main/notebooks/Classification_Project.ipynb)

[Website demo streamlit](https://mask-detector-covid.herokuapp.com/)

## Configuration  Environment 

```
# Crea un ambiente virtuale
virtualenv --python=python3.8 env
# or python3.8 -m venv  env

# Entra dentro l'ambiente virtuale
source env/bin/activate

# Install le dipendenze (solo una volta)
pip install -r requirements.txt

# Run the training with standard datatest
python main_full.py

# Run training with augmentation
# To train with augmented dataset
# 1. Generate augmented dataset 
cd utils
python test_keras_augmentation.py
# 2. set  self.TRAIN_WITH_AUGMENTATION  = True
python main_full.py

# Run detection + model
python main_simple.py

# Streamlit app
streamlit run app_streamlit.py

# ---- Extra
# Compare haarcascade detector with mobilnet
python test_face_detectors.py

# pipeline haarcascade + svm
python test_haarcascade_detectors.py

# model to find person
python test_person_detector_mobilessd.py

```

```
# Salva tutti i pacchetti python contenuti nell'ambiente virtuale
pip freeze > requirements.txt

# Install versione specifica di una libreria
pip install scikit-learn==1.0.1
```
