import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_decision_tree(file_path):
    # Carregar os dados tratados
    df = pd.read_csv(file_path)
    
    # Separar as features (X) e o target (y)
    features = df.drop(columns=['Walc'])  # 'Walc' será o alvo
    target = df['Walc']
    
    # Dividir em conjunto de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42)
    
    # Treinar o modelo de Árvore de Decisão
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Fazer previsões
    y_pred = model.predict(X_test)
    
    # Avaliar o modelo
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Acurácia do Modelo: {accuracy:.2f}")
    print("Relatório de Classificação:")
    print(report)
    
    return model

if __name__ == "__main__":
    # Caminho ajustado para acessar o dataset
    file_path = os.path.join("..", "dataset", "student-mat-cleaned.csv")
    
    # Treinar e avaliar o modelo
    model = train_decision_tree(file_path)
    print("Modelo de classificação treinado com sucesso!")
