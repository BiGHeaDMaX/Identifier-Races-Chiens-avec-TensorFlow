import streamlit as st

# Titre de la page
st.set_page_config(page_title="Dog detector")

# Liste des classes
class_names = [
"n02085620-Chihuahua",
"n02085782-Japanese_spaniel",
"n02085936-Maltese_dog",
"n02086079-Pekinese",
"n02086240-Shih-Tzu",
"n02086646-Blenheim_spaniel",
"n02086910-papillon",
"n02087046-toy_terrier",
"n02087394-Rhodesian_ridgeback",
"n02088094-Afghan_hound",
]
# Création d'un string pour affichier les classes
# sur la page de l'app
race_connues_string = "Les races connues par le modèle sont les suivantes :\n\n"
for i in class_names:
    race_connues_string += f"- {i[10:]}\n"

# Fonction pour charger une image
# et la convertir en array
def chargeur_image(image=None, load_img=None, img_to_array=None):
    # Chargement du fichier
    img = load_img(image, target_size=(299, 299))
    # Conversion en array
    img = img_to_array(img)
    # Rechape de l'array
    img = img.reshape(1,299,299,3)
    return img

# Fonction de prédiction
# fait un model.predict sur une image
def predicteur(img=None, class_names=None, model=None, np=None):
    prediction = model.predict(img, verbose=False)
    prediction = np.argmax(prediction)
    return class_names[prediction]

# Fonction principale de notre app
def main():

    # Titre
    st.title("Détecteur de race de chien")

    ################################
    # CHARGEMENT DES BIBLIOTHEQUES #
    ################################

    # Je vais inclure les imports dans la fonction main()
    # pour pouvoir indiquer que les imports sont en cours
    txt_chargement_modules = "⏳ Chargement des modules, veuillez patienter..."
    import_bar = st.progress(0, text=txt_chargement_modules)  # Création d'une barre de chargement
    import numpy as np  # Manipulation d'arrays et utilisation de np.argmax
    import_bar.progress(20, text=txt_chargement_modules)
    from tensorflow.keras.models import Sequential
    import_bar.progress(40, text=txt_chargement_modules)
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Rescaling
    import_bar.progress(60, text=txt_chargement_modules)
    # Modèle préentraîné InceptionV3
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    import_bar.progress(80, text=txt_chargement_modules)
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    import_bar.progress(100, text=txt_chargement_modules)
    import_bar.empty()

    ######################
    # CREATION DU MODELE #
    ######################

    txt_creation = "🛠️ Création du modèle, veuillez patienter..."
    model_bar = st.progress(0, text=txt_creation)  # Création d'une barre de chargement
    # Création et compilation du modèle
    # sans GPU, pour un fonctionnement universel
    # Récupération du modèle pré-entraîné
    model_base = InceptionV3(include_top=False,  # On ne prend pas les dernières couches,
                                                 # que l'on va adapter à notre problématique
                             weights="imagenet",
                             input_shape=(299, 299, 3)  # Format d'entrée par défaut pour ce modèle
                 )
    model_bar.progress(20, text=txt_creation)
    # On fige les couches du modèle préentraîné
    # les poids ne seront pas modifiés
    for layer in model_base.layers:
        layer.trainable = False
    model_bar.progress(30, text=txt_creation)
    # Définition du nouveau modèle
    model_incep = Sequential([
                        Rescaling(1./127.5, offset=-1, input_shape=(299, 299, 3)),  # Pour ce modèle, rescaler les images augmente énormément
                                                                                    # les performances. Changement : [0, 255] vers [-1, 1]
                                                                                    # Ce layer est appliqué lors du training et de l'inference
                        model_base,  # Modèle préentraîné sans les top layers
                        GlobalAveragePooling2D(),
                        Dense(256, activation='relu'),
                        Dropout(0.5),
                        Dense(len(class_names), activation='softmax')  # len(class_names) : nombre de classes
            ])
    model_bar.progress(60, text=txt_creation)
    # compilation du modèle 
    model_incep.compile(loss="categorical_crossentropy", optimizer='Adagrad', metrics=["accuracy"])
    # Chargement des meilleurs poids des couches spécifiques
    # aux races de chiens obtenus lors de l'entraînement
    model_bar.progress(90, text=txt_creation)
    model_incep.load_weights("model_incep_best_weights.h5")
    model_bar.progress(100, text=txt_creation)
    model_bar.empty()

    # On affiche les races de chien que
    # le modèle est capable de prédire
    races_connues = st.code(race_connues_string)

    fichier = st.file_uploader("Chargez la photo d'un chien dont vous souhaitez prédire la race : ")
    empl_image = st.empty()
    empl_bouton = st.empty()
    empl_resultat = st.empty()
    clic_bouton=False
    
    # Si un fichier a été chargé
    if fichier is not None:
        # st.spinner : pour afficher un message
        # durant l'exécution d'un bloc de code
        with st.spinner("Chargement de l'image..."):
            # Affichage de l'image chargée
            empl_image.image(fichier, width=299)
            # Bouton pour lancer la prédiction
            clic_bouton = empl_bouton.button("Trouver la race du chien")

    # Si le bouton de empl_bouton est cliqué
    if clic_bouton:
        with st.spinner('Prédiction en cours...'):
            # On efface le bouton
            empl_bouton.empty()
            # Chargement et convertion de l'image en array
            img = chargeur_image(image=fichier, load_img=load_img, img_to_array=img_to_array)
            # Predict du modèle sur l'array
            resultat = predicteur(img=img, class_names=class_names, model=model_incep, np=np)
            # Message dans encart vert
            empl_resultat.success(f"Race de chien trouvée : {resultat[10:]}", icon="🐶")

if __name__ == "__main__":
    main()