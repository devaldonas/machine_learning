# Passo 1: Importar Bibliotecas e Carregar Dados
import tensorflow as tf
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carregar o dataset Iris do scikit-learn
# O dataset Iris contém 150 amostras de 3 espécies de flores (setosa, versicolor, virginica)
# Cada amostra tem 4 características: comprimento e largura da sépala e pétala
iris = load_iris()
X = iris.data  # Características (features)
y = iris.target  # Rótulos (labels)

# Passo 2: Pré-processamento dos Dados
# Dividir os dados em conjuntos de treinamento (80%) e teste (20%)
# random_state=42 garante reproduibilidade dos resultados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados para melhor performance do modelo
# A normalização coloca todas as características na mesma escala
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Aprender e aplicar transformação nos dados de treino
X_test = scaler.transform(X_test)  # Aplicar a mesma transformação nos dados de teste

# Passo 3: Construir o Modelo
# Criar uma rede neural sequencial com TensorFlow/Keras
model = tf.keras.Sequential([
    # Camada de entrada com 4 neurônios (um para cada característica)
    tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
    # Camada oculta com 10 neurônios e função de ativação ReLU
    tf.keras.layers.Dense(10, activation='relu'),
    # Camada de saída com 3 neurônios (um para cada classe) e softmax para probabilidades
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compilar o modelo configurando o otimizador, função de perda e métricas
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Adequada para classificação multiclasse
              metrics=['accuracy'])

# Passo 4: Treinar o Modelo
# Treinar o modelo por 100 épocas (passagens por todo o dataset)
# validation_split=0.1 usa 10% dos dados de treino para validação durante o treinamento
history = model.fit(X_train, y_train, 
                   epochs=100, 
                   validation_split=0.1,
                   verbose=1)  # Mostra progresso do treinamento

# Passo 5: Avaliar o Modelo
# Avaliar o desempenho do modelo nos dados de teste nunca vistos
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nAcurácia no conjunto de teste: {test_accuracy:.4f}")

# Passo 6: Fazer Previsões
# Fazer previsões com o modelo treinado
predictions = model.predict(X_test)

# Converter probabilidades em classes previstas (0, 1 ou 2)
predicted_classes = tf.argmax(predictions, axis=1)

# Mostrar algumas previsões comparadas com os valores reais
print("\nPrimeiras 10 previsões:")
print("Real:    ", y_test[:10])
print("Previsto:", predicted_classes.numpy()[:10])

# Calcular acurácia manualmente para verificação
correct_predictions = tf.equal(predicted_classes, y_test)
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
print(f"\nAcurácia verificada: {accuracy.numpy():.4f}")