#CRIANDO ARVORE DE DECISAO
import pandas as pd
# da biblioteca SKLEARN importar TREE
from sklearn import tree
# da biblioteca sklearn.metrics importar accuracy_score
from sklearn.metrics import accuracy_score
# da biblioteca sklearn.model_selection importar train_test_split
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# leitura dos dados
df = pd.read_csv('/content/vendas_rio_tuor_2023.csv')




# definição dos dados de entrada X, de saída y, de treinamento e de teste
X = df.drop('Produto', axis=1)
y = df['Produto']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state = 1)

# aprendizado
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

# # mostrar desempenho
# y_prediction = clf.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_prediction))

#  mostrar classificação
# entrada1 = float(input('Digite sepal length: '))
# entrada2 = float(input('Digite sepal width: '))
# entrada3 = float(input('Digite petal length: '))
# entrada4 = float(input('Digite petal width: '))
# df_para_classificar = pd.DataFrame(
#       [[entrada1, entrada2, entrada3, entrada4]], columns=
#        ['sepal length', 'sepal width', 'petal length', 'petal width'])
# y_prediction = clf.predict(df_para_classificar)
# print("Prediction for Decision Tree: ", y_prediction)
