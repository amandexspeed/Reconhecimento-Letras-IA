import pandas as pd
import numpy as np
from sklearn import preprocessing

url = "https://raw.githubusercontent.com/amandexspeed/Reconhecimento-Letras-IA/e57e5c480be18545438f4446eae4b801a54ceb7a/dados.csv"

base_Treinamento = pd.read_csv(url,sep=',', encoding = 'latin1').values

print("Pré-processamento dos dados")

lb = preprocessing.LabelBinarizer()

lb.fit([1,-1])

resultados = lb.fit_transform(base_Treinamento[:, 0])
resultados = resultados * 2 - 1

atributos_treinamento = np.column_stack((base_Treinamento[:, 1], base_Treinamento[:, 2], base_Treinamento[:, 3], base_Treinamento[:, 4], base_Treinamento[:, 5], base_Treinamento[:, 6], base_Treinamento[:, 7], base_Treinamento[:, 8], base_Treinamento[:, 9]))

resultados_treinamento = np.hstack(resultados)

if(__name__ == "__main__"):
   
   print("Pré-processamento das classes")
   print(resultados)

   print("normalização dos atributos")
   print(atributos_treinamento)

   print("normalização dos resultados")
   print(resultados_treinamento)