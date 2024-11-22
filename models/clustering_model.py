import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def train_kmeans(file_path, n_clusters=3):
    # Carregar os dados tratados
    df = pd.read_csv(file_path)
    
    # Selecionar features (excluímos a coluna target 'Walc' porque não é usada para clusterização)
    features = df.drop(columns=['Walc'])
    
    # Normalizar os dados para evitar que escalas diferentes influenciem os clusters
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Treinar o modelo K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_features)
    
    # Adicionar os rótulos dos clusters ao dataframe
    df['Cluster'] = kmeans.labels_
    
    # Salvar os dados com clusters em um novo arquivo
    output_file = os.path.join("..", "dataset", "student-mat-clusters.csv")
    df.to_csv(output_file, index=False)
    print(f"Clusters adicionados ao arquivo: {output_file}")
    
    # Visualizar a inércia (métrica para avaliar o desempenho do K-Means)
    print(f"Inércia (Soma dos Erros Quadráticos): {kmeans.inertia_}")
    
    return kmeans, df

def plot_clusters(df, feature_x, feature_y, cluster_column='Cluster'):
    # Plotar os clusters em relação a duas features
    plt.figure(figsize=(10, 6))
    for cluster in df[cluster_column].unique():
        cluster_data = df[df[cluster_column] == cluster]
        plt.scatter(cluster_data[feature_x], cluster_data[feature_y], label=f"Cluster {cluster}")
    
    plt.title(f"Clusters baseados nas features '{feature_x}' e '{feature_y}'")
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Caminho para o dataset tratado
    file_path = os.path.join("..", "dataset", "student-mat-cleaned.csv")
    
    # Treinar o modelo de K-Means
    kmeans, clustered_data = train_kmeans(file_path, n_clusters=3)
    
    # Exibir os clusters em um gráfico (escolha as colunas desejadas para o eixo X e Y)
    plot_clusters(clustered_data, feature_x='G1', feature_y='G2')
    print("Modelo de clusterização treinado com sucesso!")
