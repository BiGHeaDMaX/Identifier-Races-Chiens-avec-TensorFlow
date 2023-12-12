#######################################################
#        APP DE PREDICTION DE RACES DE CHIEN          #
#     Fonctionne avec Streamlit version: 1.29.0       #
# A lancer avec : streamlit run "04 - Application.py" #
#######################################################

# On commence par importer streamlit
import streamlit as st

# Titre de la page (onglet)
st.set_page_config(page_title="Dog detector")
# Titre
st.title("Détecteur de race de chien")

################################
# CHARGEMENT DES BIBLIOTHEQUES #
################################

# st.spinner : pour afficher un message durant l'exécution d'un bloc de code
with st.spinner('⏳ Chargement des bibliothèques, veuillez patienter...'):
    import numpy as np  # Manipulation d'arrays et utilisation de np.argmax
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Rescaling
    # Modèle préentraîné InceptionV3
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

###########
# CLASSES #
###########

# Classes sur lesquelles avait été entraîné notre modèle
class_names = [
"n02085620-Chihuahua",
"n02085782-Japanese Spaniel",
"n02085936-Maltese Dog",
"n02086079-Pekinese",
"n02086240-Shih-Tzu",
"n02086646-Blenheim Spaniel",
"n02086910-Papillon",
"n02087046-Toy Terrier",
"n02087394-Rhodesian Ridgeback",
"n02088094-Afghan Hound",
]
# Création d'un string pour affichier
# les classes sur la page de l'app
race_connues_string = "Les races connues par le modèle sont les suivantes :\n\n"
for i in class_names:
    race_connues_string += f"- {i[10:]}\n"  # [10:] pour masquer le code de classe

######################
# CREATION DU MODELE #
######################

@st.cache_resource(show_spinner=False)  # Mettre en cache cette fonction pour réexécution plus rapide
def model_creator():

    txt_creation = "🛠️ Initialisation du modèle, veuillez patienter..."
    model_bar = st.progress(0, text=txt_creation)  # Création d'une barre de progression
    # Création et compilation du modèle
    # sans GPU, pour un fonctionnement universel
    # Récupération du modèle pré-entraîné
    model_base = InceptionV3(include_top=False,  # On ne prend pas les dernières couches,
                                                 # que l'on va adapter à notre problématique
                                weights="imagenet",
                                input_shape=(299, 299, 3)  # Format d'entrée par défaut pour ce modèle
                    )
    model_bar.progress(20, text=txt_creation)  # Avancée de la barre de progression
    # On fige les couches du modèle préentraîné
    # les poids ne seront pas modifiés
    for layer in model_base.layers:
        layer.trainable = False
    model_bar.progress(30, text=txt_creation)  # Avancée de la barre de progression
    # Définition du nouveau modèle
    model_incep = Sequential([Rescaling(1./127.5, offset=-1, input_shape=(299, 299, 3)),  # Pour ce modèle, rescaler les images augmente énormément
                                                                                          # les performances. Changement : [0, 255] vers [-1, 1]
                                                                                          # Ce layer est appliqué lors du training et de l'inference
                              model_base,  # Modèle préentraîné sans les top layers
                              GlobalAveragePooling2D(),
                              Dense(256, activation='relu'),
                              Dropout(0.5),
                              Dense(len(class_names), activation='softmax')  # len(class_names) : nombre de classes
                  ])
    model_bar.progress(60, text=txt_creation)  # Avancée de la barre de progression
    # Compilation du modèle 
    model_incep.compile(loss="categorical_crossentropy", optimizer='Adagrad', metrics=["accuracy"])
    model_bar.progress(90, text=txt_creation)  # Avancée de la barre de progression
    # Chargement des meilleurs poids des couches spécifiques
    # aux races de chiens obtenus lors de l'entraînement
    model_incep.load_weights("model_incep_best_weights.h5")
    model_bar.progress(100, text=txt_creation)  # Avancée de la barre de progression
    model_bar.empty()  # Suppression de la barre de progression

    return model_incep

# Création et compilation du modèle
model_incep = model_creator()

##################################
# FONCTION DE CHARGEMENT D'IMAGE #
#    ET CONVERSION EN ARRAY      #
##################################

def chargeur_image(image=None):

    # Chargement du fichier
    img = load_img(image, target_size=(299, 299))
    # Conversion en array
    img = img_to_array(img)
    # Rechape de l'array
    img = img.reshape(1,299,299,3)
    return img

##########################
# FONCTION DE PREDICTION #
##########################

# fait un model.predict sur une image (array)
def predicteur(img=None, class_names=None, model=None):

    prediction = model.predict(img, verbose=False)
    prediction = np.argmax(prediction)
    return class_names[prediction]

###################
# FONCTION MAIN() #
###################

def main():

    # On affiche les races de chien que
    # le modèle est capable de prédire
    st.code(race_connues_string)

    fichier = st.file_uploader("Chargez la photo d'un chien dont vous souhaitez prédire la race : ",
                               # Un seul fichier à la fois
                               accept_multiple_files=False,
                               # Formats acceptés
                               type=['jpg', 'png', 'jpeg', 'bmp', 'gif', 'tiff']
              )
    empl_image = st.empty()
    empl_resultat = st.empty()
    
    # Si un fichier a été chargé
    if fichier is not None:
        with st.spinner("Chargement de l'image..."):
            # Affichage de l'image chargée
            empl_image.image(fichier, width=299)
        with st.spinner('Prédiction en cours...'):
            # Chargement et convertion de l'image en array
            img = chargeur_image(image=fichier)
            # Predict du modèle sur l'array
            resultat = predicteur(img=img, class_names=class_names, model=model_incep)
            # Message dans encart vert
            empl_resultat.success(f"Race de chien trouvée : {resultat[10:]}", icon="🐶")  # [10:] pour masquer le code de classe

# Si le script est lancé en propre, pas importé
# dans un autre script, alors lancer main()
if __name__ == "__main__":
    main()