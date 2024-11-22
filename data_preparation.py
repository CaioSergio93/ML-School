import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

def train_regression_model(file_path):
    # Carregar o dataset
    df = pd.read_csv(file_path)

    # Preparar os dados
    X = df.drop(columns=['G3'])
    y = df['G3']

    # Normalizar os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Treinar o modelo de regressão linear
    model = LinearRegression()
    model.fit(X_scaled, y)

    # Salvar o modelo treinado, o scaler e as colunas
    joblib.dump(model, 'linear_regression_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')  # Salvar o scaler também
    joblib.dump(X.columns, 'columns.pkl')  # Salvar as colunas

    print("Modelo de regressão treinado e salvo com sucesso!")

if __name__ == "__main__":
    file_path = "C:\\Users\\caios\\Desktop\\ML\\dataset\\student-mat-cleaned.csv"
    train_regression_model(file_path)
