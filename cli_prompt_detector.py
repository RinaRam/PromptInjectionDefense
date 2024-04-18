from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from datasets import load_dataset
import re
import numpy as np
from sklearn.model_selection import train_test_split
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import spacy
import argparse
from keras.callbacks import History
import tensorflow as tf
import logging
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class PromptInjectionDetector:
    def __init__(self):
        pass

    def detect_code(self, text):
        code_keywords = ["os.", "subprocess.", "exec(", "eval(", "system(", "shell(", "`"]
        code_count = sum(1 for keyword in code_keywords if keyword in text)
        return code_count >= 2, code_count

    def detect_regex(self, text):
        regex_patterns = [
            r'\b(re\.\w+\()', r'\b(import\sre|from\sre)', r'\b(pattern|re.compile|re.match|re.search)'
        ]
        regex_count = sum(1 for pattern in regex_patterns if re.search(pattern, text))
        return regex_count > 1, regex_count

    def __init__(self, special_characters):
        self.special_characters = special_characters

    def detect_special_characters(self, text):
        # Perform detection logic
        special_char_count = sum(1 for char in text if char in self.special_characters)
        return special_char_count > 0, special_char_count

    def levenshtein_distance(self, s1, s2):
        len_s1, len_s2 = len(s1), len(s2)

        # Early exit optimization: If the difference in lengths is too large, return a high distance value
        if abs(len_s1 - len_s2) > 3:
            return 10  # Choose a suitable high value

        # Initializing the matrix for dynamic programming
        dp = [[0] * (len_s2 + 1) for _ in range(len_s1 + 1)]

        # Initialization of the first row and column
        for i in range(len_s1 + 1):
            dp[i][0] = i
        for j in range(len_s2 + 1):
            dp[0][j] = j

        # Populate the matrix using dynamic programming
        for i in range(1, len_s1 + 1):
            for j in range(1, len_s2 + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

        return dp[-1][-1]

    def detect_typo_levenshtein(self, text):
        reference_word = "prompt"
        distance = self.levenshtein_distance(reference_word, text.lower())
        threshold = 2  # Lowering the threshold for better accuracy

        # Enhancing accuracy with additional conditions
        if distance > threshold:
            # If the distance is greater than the threshold, consider additional checks
            # Check for transpositions and common mistypings
            if abs(len(reference_word) - len(text)) == 1:
                # Consider transpositions
                for i in range(min(len(reference_word), len(text)) - 1):
                    if reference_word[i] == text[i + 1] and reference_word[i + 1] == text[i]:
                        return (True, distance)  # Return True for transpositions
                # Check for single character additions/deletions
                for i in range(len(text)):
                    edited_text = text[:i] + text[i+1:]
                    if self.levenshtein_distance(reference_word, edited_text) == 1:
                        return (True, distance)  # Return True for single character edits

            return (False, distance)  # Otherwise, return False
        return (False, distance)

    def weighted_combination(self, text):
        weight_code = 0.4
        weight_regex = 0.3
        weight_special_chars = 0.2

        # Get the results from the tuple
        code_detected, _ = self.detect_code(text)
        regex_detected, _ = self.detect_regex(text)
        special_chars_detected, _ = self.detect_special_characters(text)
        typo_detected, _ = self.detect_typo_levenshtein(text)

        # Calculate the weighted sum based on Boolean results
        weighted_sum = (weight_code * code_detected) + (weight_regex * regex_detected) + (weight_special_chars * special_chars_detected) + (weight_typo * typo_detected)

        threshold = 0.5  # Threshold value for determining the presence of injection
        return weighted_sum >= threshold

    def segmented_check(self, text):
        segment_1 = text[:len(text) // 2]  # Первая половина текста
        segment_2 = text[len(text) // 2:]  # Вторая половина текста

        # Применяем разные методы к разным сегментам текста
        check_1 = self.detect_code(segment_1)
        check_2 = self.detect_regex(segment_2)

        # Принимаем решение на основе результатов обоих сегментов
        return check_1 or check_2  # Возвращаем True, если хотя бы один сегмент подозрителен

    def iterative_refinement(self, text):
        # Итеративная проверка с уточнением
        for _ in range(3):  # Проходим несколько раз по тексту для уточнения результата
            code_detected = self.detect_code(text)
            regex_detected = self.detect_regex(text)
            special_chars_detected = self.detect_special_characters(text)
            typo_detected = self.detect_typo_levenshtein(text)

            # Если обнаружено что-то подозрительное, уточняем текст для дальнейшей проверки
            if any([code_detected, regex_detected, special_chars_detected, typo_detected]):
                # Например, убираем часть текста, которая могла вызвать подозрение
                text = text.replace("dangerous_part", "")

        # Возвращаем результат после итераций
        return any([code_detected, regex_detected, special_chars_detected, typo_detected])

    def sequential_deepening(self, text):
        # Последовательная проверка с углублением
        code_detected = self.detect_code(text)
        regex_detected = self.detect_regex(text)
        special_chars_detected = self.detect_special_characters(text)
        typo_detected = self.detect_typo_levenshtein(text)

        # Проверяем наличие подозрительных фрагментов
        suspicious_detected = any([code_detected, regex_detected, special_chars_detected, typo_detected])

        # Если что-то подозрительное обнаружено, углубляем анализ текста
        if suspicious_detected:
            # Выполняем итеративное уточнение и получаем новый результат
            refined_result = self.iterative_refinement(text)

            # Возвращаем результат итеративного уточнения
            return refined_result

        # Возвращаем исходный результат, если ничего подозрительного не обнаружено
        return False

    def combination_1(self, text):
        code_detected = self.detect_code(text)
        special_chars_detected = self.detect_special_characters(text)
        return code_detected and special_chars_detected

    def combination_2(self, text):
        typo_detected = self.detect_typo_levenshtein(text)
        #typo_detected, _ = self.detect_typo_levenshtein(text)
        regex_detected = self.detect_regex(text)
        return typo_detected and regex_detected

    def combination_3(self, text):
        weighted_result = self.weighted_combination(text)
        typo_detected = self.detect_typo_levenshtein(text)
        #typo_detected, _ = self.detect_typo_levenshtein(text)
        return weighted_result or typo_detected

    def combination_4(self, text):
        segmented_result = self.segmented_check(text)
        iterative_result = self.iterative_refinement(text)
        return segmented_result or iterative_result

    def combination_5(self, text):
        sequential_result = self.sequential_deepening(text)
        special_chars_detected = self.detect_special_characters(text)
        return sequential_result and special_chars_detected

    def preprocess_and_train(self, method, texts):
        processed_texts = [str(method(text)) for text in texts]
        return processed_texts

    def vectorize_text(self, texts):
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

        # N-Gram Vectorizer
        ngram_vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=1000)
        ngram_vectors = ngram_vectorizer.fit_transform(texts)

        # Part-of-Speech Tagging
        pos_tags = []
        for text in texts:
            doc = nlp(text)
            pos_tags.append(" ".join([token.pos_ for token in doc]))

        # Combine N-Gram and POS features
        combined_vectors = np.concatenate((ngram_vectors.toarray(), TfidfVectorizer().fit_transform(pos_tags).toarray()), axis=1)

        return combined_vectors, ngram_vectorizer

    def train_neural_network(self, method, X_train, X_test, y_train, y_test):
        X_train_processed = self.preprocess_and_train(method, X_train)
        X_test_processed = self.preprocess_and_train(method, X_test)

        vectorizer = TfidfVectorizer(max_features=1000)
        X_train_vectors = vectorizer.fit_transform(X_train_processed).toarray()
        X_test_vectors = vectorizer.transform(X_test_processed).toarray()

        model = self.build_neural_network(X_train_vectors.shape[1])  # Create a new model instance

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        history = model.fit(X_train_vectors, y_train, epochs=180, batch_size=64, validation_split=0.2, verbose=0)
        
        accuracy = history.history['accuracy'][-1]
        print("Accuracy for {}: {}".format(method.__name__, accuracy))

    def build_neural_network(self, input_shape):
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_shape,)),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        return model

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        history = model.fit(X_train_vectors, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
        accuracy = history.history['accuracy'][-1]
        print("Accuracy for {}: {}".format(method, accuracy))

    def fit_model_with_neural_network(self):
        dataset = load_dataset("deepset/prompt-injections")
        X = dataset['train']['text']
        y = dataset['train']['label']

        if not isinstance(X, list) or not isinstance(y, list):
            X = list(X)
            y = list(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        if not isinstance(X_train, list) or not isinstance(X_test, list) or not isinstance(y_train, list) or not isinstance(y_test, list):
            X_train = list(X_train)
            X_test = list(X_test)
            y_train = list(y_train)
            y_test = list(y_test)

        methods = [
            self.detect_code,
            self.detect_regex,
            self.detect_special_characters,
            self.detect_typo_levenshtein,
            self.weighted_combination,
            self.segmented_check,
            self.iterative_refinement,
            self.sequential_deepening,
            self.combination_1,
            self.combination_2,
            self.combination_3,
            self.combination_4,
            self.combination_5
        ]

        for method in methods:
            # Convert the data to NumPy arrays
            X_train_np = np.array(X_train)
            X_test_np = np.array(X_test)
            y_train_np = np.array(y_train)
            y_test_np = np.array(y_test)

            self.train_neural_network(method, X_train_np, X_test_np, y_train_np, y_test_np)

special_characters = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')']
weight_typo = 0.2


detector = PromptInjectionDetector(special_characters)

# Fit the model with the neural network
detector.fit_model_with_neural_network()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CLI for Prompt Injection Detector')
    parser.add_argument('-t', '--text', type=str, help='Text to analyze')
    parser.add_argument('-m', '--method', type=str, help='Method for analysis')

    args = parser.parse_args()

    special_characters = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')']
    weight_typo = 0.2

    detector = PromptInjectionDetector(special_characters)

    if args.method == 'detect_code':
        result = detector.detect_code(args.text)
        print(f"Result of 'detect_code' method for text '{args.text}': {result}")

tf.get_logger().setLevel(logging.ERROR)
