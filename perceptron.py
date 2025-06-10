from sklearn.linear_model import Perceptron
from preprocessamento import atributos_treinamento, resultados_treinamento,lb

neuronio = Perceptron()
neuronio.fit(atributos_treinamento, resultados_treinamento)

print("Acurácia do modelo:")
print(neuronio.score(atributos_treinamento, resultados_treinamento))

print("Pesos do neurônio:")
print(neuronio.coef_)

print("Bias do neurônio:")
print(neuronio.intercept_)

print("Predição para um novo exemplo:")
novo_exemplo = [[1,1,1,1,1,0,0,1,0]]
print("A predição é: ", "T" if neuronio.predict(novo_exemplo)[0] == 1 else "H")
