import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Criando dados fictícios para treinamento
# Substitua esses dados por seu dataset real
data = {
    "idade": np.random.randint(15, 22, 100),
    "sexo": np.random.choice([0, 1], 100),  # 0 = F, 1 = M
    "endereco": np.random.choice([0, 1], 100),  # 0 = Rural, 1 = Urbano
    "tamanho_familia": np.random.choice([0, 1], 100),  # 0 = GT3, 1 = LE3
    "status_pais": np.random.choice([0, 1], 100),  # 0 = A, 1 = T
    "mae_anos_escolaridade": np.random.randint(0, 5, 100),
    "pai_anos_escolaridade": np.random.randint(0, 5, 100),
    "profissao_mae": np.random.choice([0, 1], 100),
    "profissao_pai": np.random.choice([0, 1], 100),
    "razao_curso": np.random.choice([0, 1], 100),
    "responsavel": np.random.choice([0, 1], 100),
    "tempo_viagem": np.random.randint(1, 5, 100),
    "tempo_estudo": np.random.randint(1, 5, 100),
    "falhas_academicas": np.random.randint(0, 5, 100),
    "apoio_educacional": np.random.choice([0, 1], 100),
    "apoio_familiar": np.random.choice([0, 1], 100),
    "aulas_particulares": np.random.choice([0, 1], 100),
    "atividades_extracurriculares": np.random.choice([0, 1], 100),
    "pre_escola": np.random.choice([0, 1], 100),
    "ensino_superior": np.random.choice([0, 1], 100),
    "acesso_internet": np.random.choice([0, 1], 100),
    "relacionamento_romantico": np.random.choice([0, 1], 100),
    "relacionamento_familiar": np.random.randint(1, 6, 100),
    "tempo_livre": np.random.randint(1, 6, 100),
    "atividades_sociais": np.random.randint(1, 6, 100),
    "consumo_alcool_semana": np.random.randint(1, 6, 100),
    "consumo_alcool_fim_semana": np.random.randint(1, 6, 100),
    "saude": np.random.randint(1, 6, 100),
    "ausencias": np.random.randint(0, 20, 100),
    "nota_final": np.random.randint(0, 20, 100)  # Nota final como target
}

# Criando o DataFrame
df = pd.DataFrame(data)

# Separando os recursos (X) e o alvo (y)
X = df.drop("nota_final", axis=1)
y = df["nota_final"]

# Normalizando os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Treinando o modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Avaliando o modelo
score = model.score(X_test, y_test)
print(f"Acurácia do modelo: {score * 100:.2f}%")

# Salvando o modelo, scaler e colunas
joblib.dump(model, "modelo_predicao.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "columns.pkl")

print("Modelo, scaler e colunas salvos com sucesso!")
