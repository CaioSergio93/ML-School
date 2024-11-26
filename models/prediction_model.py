import pandas as pd
import numpy as np
import joblib

# Função para fazer predições com novos dados
def predizer_nota_interativa():
    # Carregar o modelo, scaler e colunas salvos
    modelo = joblib.load("modelo_predicao.pkl")
    scaler = joblib.load("scaler.pkl")
    colunas = joblib.load("columns.pkl")

    # Definindo as opções para algumas variáveis categóricas
    opcoes = {
        "sexo": {0: "Feminino", 1: "Masculino"},
        "endereco": {0: "Rural", 1: "Urbano"},
        "tamanho_familia": {0: "Mais de 3 membros", 1: "3 ou menos membros"},
        "status_pais": {0: "Separados", 1: "Juntos"},
        "profissao_mae": {0: "Não trabalha", 1: "Trabalha"},
        "profissao_pai": {0: "Não trabalha", 1: "Trabalha"},
        "razao_curso": {0: "Não estuda", 1: "Estuda"},
        "responsavel": {0: "Pai/Mãe", 1: "Outro responsável"},
        "apoio_educacional": {0: "Não recebe", 1: "Recebe"},
        "apoio_familiar": {0: "Não recebe", 1: "Recebe"},
        "aulas_particulares": {0: "Não", 1: "Sim"},
        "atividades_extracurriculares": {0: "Não", 1: "Sim"},
        "pre_escola": {0: "Não", 1: "Sim"},
        "ensino_superior": {0: "Não", 1: "Sim"},
        "acesso_internet": {0: "Não", 1: "Sim"},
        "relacionamento_romantico": {0: "Não tem", 1: "Tem"},
    }

    # Solicitar os valores do usuário
    print("Por favor, insira os dados solicitados:")

    entrada = {}
    for coluna in colunas:
        if coluna in opcoes:
            # Para colunas com opções definidas
            print(f"\nEscolha uma opção para {coluna}:")
            for key, value in opcoes[coluna].items():
                print(f"{key}: {value}")
            while True:
                try:
                    valor = int(input(f"{coluna} (digite o número da opção): "))
                    if valor not in opcoes[coluna]:
                        print(f"Valor inválido. Digite uma opção entre {list(opcoes[coluna].keys())}.")
                    else:
                        entrada[coluna] = valor
                        break
                except ValueError:
                    print(f"Entrada inválida para {coluna}. Tente novamente.")
        else:
            # Para colunas numéricas
            while True:
                try:
                    valor = input(f"{coluna}: ")
                    valor = float(valor) if '.' in valor else int(valor)
                    entrada[coluna] = valor
                    break
                except ValueError:
                    print(f"Entrada inválida para {coluna}. Tente novamente.")

    # Criar o DataFrame com os dados fornecidos
    entrada_df = pd.DataFrame([entrada])

    # Garantir que as colunas estão na mesma ordem
    entrada_df = entrada_df[colunas]

    # Escalar os novos dados
    entrada_escalada = scaler.transform(entrada_df)

    # Fazer a predição
    predicao = modelo.predict(entrada_escalada)

    return predicao[0]

# Executar o processo de predição interativo
if __name__ == "__main__":
    resultado = predizer_nota_interativa()
    print(f"\nA predição da nota final é: {resultado:.2f}")
