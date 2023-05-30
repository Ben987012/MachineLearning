import streamlit as st

from strokePred import rf,knn,dtc, svm
from strokePred import x_test, y_test

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def add_bg_from_url(url):
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url(url);
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

def main():
    st.title("STROKE ELŐREJELZŐ APP")
    
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
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('True Positive arány')
        plt.ylabel('False Positive arány')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        st.pyplot(plt)
        #add_bg_from_url("https://freerangestock.com/sample/145397/artificial-intelligence-background--abstract-ai-background-with.jpg")

    if st.button('Modellek összevetése'):
        rf_accuracy = rf.score(x_test, y_test)
        knn_accuracy = knn.score(x_test, y_test)
        svm_accuracy = svm.score(x_test, y_test)
        # Kiíratás
        st.write('RandomForest pontossága: {}%'.format(round((rf_accuracy*100),2)))
        st.write('KNN pontossága: {}%'.format(round((knn_accuracy*100),2)))
        st.write('SVM pontossága: {}%'.format(round((svm_accuracy*100),2)))
        #add_bg_from_url("https://png.pngtree.com/thumb_back/fh260/back_our/20190621/ourmid/pngtree-blue-artificial-intelligence-technology-ai-robot-banner-image_196890.jpg")
        
        # Streamlit alkalmazás
        st.title("Háttér visszaállítás")
            #add_bg_from_url("https://static.toiimg.com/photo/msid-87343087/87343087.jpg")

if __name__ == '__main__':
   main()
   add_bg_from_url("https://static.toiimg.com/photo/msid-87343087/87343087.jpg")
