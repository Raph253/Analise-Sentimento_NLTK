import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy

# Baixar recursos adicionais do NLTK
nltk.download('movie_reviews')

# Obter avaliações de filmes e dividir em conkuntos de treinamento e teste
neg_reviews = [(movie_reviews.words(file_id), 'negative') for file_id in movie_reviews.fileids('neg')]
pos_reviews = [(movie_reviews.words(file_id), 'positive') for file_id in movie_reviews.fileids('pos')]
reviews = neg_reviews + pos_reviews
split_index = int(len(reviews) * 0.8)
train_set = reviews[:split_index]
test_set = reviews[split_index:]

# Extrair características dos textos (frequência de palavras)
def extract_features(words):
    return dict([(word, True) for word in words])

# Preparar os dados de treinamento e teste
train_features = [(extract_features(words), category) for (words, category) in train_set]
test_features = [(extract_features(words), category) for (words, category) in test_set]

# Treinar o classificador Naive Bayes
classifier = NaiveBayesClassifier.train(train_features)

# Avaliar a preisão do classificador
accuracy = nltk_accuracy(classifier, test_features)
print(f'Acurácia do classificador: {accuracy:.2f}')

# Exemplo de uso do classificador
new_review = "The movie was not good. I would not recommend it"
new_features = extract_features(new_review.split())
print(f'Classificação da nova avaliação: {classifier.classify(new_features)}')