import streamlit as st

from strokePred import rf,knn,dtc
from strokePred import x_test, y_test

from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def main():
    st.title('Stroke előrejelző app')

    if st.button('Konfúziós mátrix megjelenítése'):
        # Valószínűségek és címkék generálása
        np.random.seed(0)
        n_samples = 1000
        y_true = np.random.randint(2, size=n_samples)
        y_scores = np.random.rand(n_samples)

        # ROC görbe kiszámítása
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # Streamlit alkalmazás létrehozása
        st.title("ROC görbe")
        st.write("Ez egy egyszerű példa a ROC görbe megjelenítésére a Streamlit segítségével.")

        # ROC görbe kirajzolása
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'ROC görbe (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('Hamis pozitív arány')
        ax.set_ylabel('Valós pozitív arány')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        st.pyplot(fig)

    if st.button('Modellek összevetése'):
        rf_accuracy = rf.score(x_test, y_test)
        knn_accuracy = knn.score(x_test, y_test)
        dtc_accuracy = dtc.score(x_test, y_test)
        # Kiíratás
        st.write('RandomForest pontossága: ', rf_accuracy) #{}%'.format(rf_accuracy*100))
        st.write('KNN pontossága: ', knn_accuracy) #{}%'.format(knn_accuracy*100))
        st.write('SVM pontossága: ', dtc_accuracy) #{}%'.format(dtc_accuracy*100))

if __name__ == '__main__':
    main()
