import streamlit as st

from strokePred import rf,knn,dtc, svm
from strokePred import x_test, y_test

from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://cdn.pixabay.com/photo/2019/04/24/11/27/flowers-4151900_960_720.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

def main():
    if st.button('ROC görbe megjelenítése'):
       # Tesztadatok előrejelzése
        y_pred = rf.predict_proba(x_test)[:, 1]  # Első oszlopban a pozitív osztály előrejelzéseinek valószínűségeit tároljuk

        # ROC görbe számítása
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)

        # Streamlit alkalmazás
        st.title("ROC görbe")

        # ROC görbe megjelenítése
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label='ROC görbe (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Véletlenszerű előrejelzés (AUC = 0.50)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Hamis pozitív arány')
        plt.ylabel('Eredeti pozitív arány')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        st.pyplot(plt)

    if st.button('Modellek összevetése'):
        rf_accuracy = rf.score(x_test, y_test)
        knn_accuracy = knn.score(x_test, y_test)
        svm_accuracy = svm.score(x_test, y_test)
        # Kiíratás
        st.write('RandomForest pontossága: {}%'.format(round((rf_accuracy*100),2)))
        st.write('KNN pontossága: {}%'.format(round((knn_accuracy*100),2)))
        st.write('SVM pontossága: {}%'.format(round((svm_accuracy*100),2)))

if __name__ == '__main__':
   main()
   add_bg_from_url()
