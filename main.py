import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import string

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