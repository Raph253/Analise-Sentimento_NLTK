import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import string
import matplotlib.pyplot as plt

# Baixar recusos adicionais do NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Texto de exemplo
texto = """
O cachorro correu pelo parque. O gato dormiu na poltrona. O cachorro latiu para o gato.
"""

# Tokenização
tokens = word_tokenize(texto.lower(), language='portuguese') # Convertendo para minúsculo

# Remoção de stopwords
pontuacao = set(string.punctuation)
stop_words = set(stopwords.words('portuguese'))
tokens_filtrados = [word for word in tokens if word not in pontuacao and word not in stop_words]

# Contagem de frequencia
frequencia = Counter(tokens_filtrados)

# Exibição dos resultados
print("Frequência das palavras:")
for palavra, freq in frequencia.items():
    print(f"{palavra}: {freq}")

# Plotagem do gáfocp de barras
plt.figure(figsize=(12,6))
plt.bar(frequencia.keys(), frequencia.values())
plt.xticks(rotation=45)
plt.xlabel('Palavra')
plt.ylabel('Frequencia')
plt.title('Frequencia das palavras no texto')
plt.show()

# Plotagem do gráfico de pizza
plt.figure(figsize=(8,8))
plt.pie(frequencia.values(), labels=frequencia.keys(), autopct='%1.1f%%')
plt.title('Distribuição das Palavras no Texto')
plt.show()