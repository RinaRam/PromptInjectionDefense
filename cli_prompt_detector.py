import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
import tensorflow as tf
from sklearn.metrics import accuracy_score
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.optimizers import Adam
from transformers import TFBertModel
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from datasets import load_dataset
import numpy as np
import re
import spacy
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from keras.callbacks import History
import logging
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class PromptInjectionDetector:
    def __init__(self, special_characters):
        self.special_characters = special_characters
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = TFBertModel.from_pretrained('bert-base-uncased', trainable=False)
        self.method_accuracies = {}
        self.nlp = spacy.load("en_core_web_sm")

    def preprocess_text(self, text):
        encoded_text = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=64, pad_to_max_length=True, return_tensors='tf')
        return encoded_text['input_ids'][0].numpy()

    def build_enhanced_neural_network(self, input_shape):
        input_layer = Input(shape=(input_shape,), dtype=tf.int32, name="input_layer")
        bert_output = self.bert_model(input_layer)[0]
        flattened_bert_output = Flatten()(bert_output)
        dense_1 = Dense(256, activation='relu')(flattened_bert_output)
        dropout_1 = Dropout(0.5)(dense_1)
        dense_2 = Dense(128, activation='relu')(dropout_1)
        dropout_2 = Dropout(0.3)(dense_2)
        output_layer = Dense(1, activation='sigmoid')(dropout_2)

        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        return model

    def train_enhanced_neural_network(self, X_train, X_test, y_train, y_test, method):
        X_train_processed = np.array([self.preprocess_text(text) for text in X_train])
        X_test_processed = np.array([self.preprocess_text(text) for text in X_test])

        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)

        model = self.build_enhanced_neural_network(X_train_processed.shape[1])

        optimizer = Adam(learning_rate=5e-5)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        learning_rate_scheduler = LearningRateScheduler(lambda epoch: 5e-5 * (0.8 ** epoch))

        checkpoint = ModelCheckpoint("best_initial_model.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', period=1, save_weights_only=True)
        # model_history = model.fit(X_train, y_train,batch_size=64, epochs=180, validation_data=(X_test, y_test),callbacks=[checkpoint])

        history = model.fit(
            X_train_processed, y_train_encoded,
            epochs=180, batch_size=64,
            validation_split=0.2,
            callbacks=[checkpoint],
            verbose=1
        )

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('Initial_Model_Accuracy.png')
        plt.show()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('Initial_Model_loss.png')
        plt.show()

        model.load_weights("best_initial_model.hdf5")

        print("Loss of the model is - " , model.evaluate(X_test_processed,y_test_encoded)[0])
        print("Accuracy of the model is - " , model.evaluate(X_test_processed,y_test_encoded)[1]*100 , "%")

        predictions = model.predict(X_test_processed)

        finaldf = np.array([y_test_encoded, np.concatenate(predictions)]).T
        finaldf = pd.DataFrame(data=finaldf, columns=["Predicted Values", "Actual Values"])

        cm = confusion_matrix(y_test_encoded, [np.round(i) for i in predictions])
        plt.figure(figsize = (12, 10))
        cm = pd.DataFrame(cm , index = [0, 1] , columns = [i for i in [0, 1]])
        ax = sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        plt.title('Confusion Matrix', size=20)
        plt.xlabel('Predicted Labels', size=14)
        plt.ylabel('Actual Labels', size=14)
        plt.savefig('Initial_Model_Confusion_Matrix.png')
        plt.show()


        self.method_accuracies[method] = model.evaluate(X_test_processed,y_test_encoded)[1]
        print("Accuracy for {}: {}".format(method, model.evaluate(X_test_processed,y_test_encoded)[1]))


    def fit_model_with_enhancements(self):
        dataset = load_dataset("deepset/prompt-injections")
        X = dataset['train']['text']
        y = dataset['train']['label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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
            X_train_np = np.array(X_train)
            X_test_np = np.array(X_test)
            y_train_np = np.array(y_train)
            y_test_np = np.array(y_test)

            self.train_enhanced_neural_network(X_train_np, X_test_np, y_train_np, y_test_np, method)

        # Print the accuracies for all methods
        for method, accuracy in self.method_accuracies.items():
            print("Accuracy for {}: {}".format(method, accuracy))

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

    def detect_syntax_features(self, text):
        doc = self.nlp(text)
        # Пример обнаружения основных признаков синтаксиса
        num_sentences = len(list(doc.sents))
        num_tokens = len(doc)
        num_nouns = len([token for token in doc if token.pos_ == "NOUN"])
        num_verbs = len([token for token in doc if token.pos_ == "VERB"])
        num_adjectives = len([token for token in doc if token.pos_ == "ADJ"])
        num_adverbs = len([token for token in doc if token.pos_ == "ADV"])

        return {
            "num_sentences": num_sentences,
            "num_tokens": num_tokens,
            "num_nouns": num_nouns,
            "num_verbs": num_verbs,
            "num_adjectives": num_adjectives,
            "num_adverbs": num_adverbs
        }

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
