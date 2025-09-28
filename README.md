# machine_learning
Modelo de Machine Learning que classifica espécies de flores Iris com base em características como comprimento e largura das sépalas e pétalas.
# Lógica Implementada:
1.	Carregamento de Dados: Foi utilizado o dataset Iris que contém 150 amostras com 4 características cada, classificadas em 3 espécies.

2.	Pré-processamento:
	Divisão em treino/teste (80/20) para avaliar generalização
	Normalização dos dados para melhor convergência do modelo

3.	Arquitetura da Rede Neural:
	Camada de entrada: 4 neurônios (features)
	Camada oculta: 10 neurônios com ReLU
	Camada de saída: 3 neurônios com Softmax (probabilidades)

4.	Treinamento:
	100 épocas para aprendizado
	Otimizador Adam para ajuste de pesos
	Função de perda adequada para classificação multiclasse

5.	Avaliação: Teste em dados nunca vistos para medir performance real

6.	Previsões: Conversão de probabilidades em classes e verificação de acurácia
O modelo aprende padrões nas características das flores para classificar corretamente as espécies Iris, alcançando alta acurácia geralmente acima de 90%.
