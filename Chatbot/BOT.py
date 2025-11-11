import os
import json
import random
from pathlib import Path

import nltk
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Ensure tokenizers are available (quiet)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)   
nltk.download('wordnet', quiet=True)

class ChatbotModelKeras:
    @staticmethod
    def build(input_size: int, output_size: int) -> Sequential:
        model = Sequential()
        model.add(Dense(128, input_shape=(input_size,), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(output_size, activation='softmax'))

        sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return model


 
class ChatbotAssistantKeras:
    def __init__(self, intents_path: Path, function_mappings=None, out_dir: Path | None = None):
        self.model: Sequential | None = None
        self.intents_path = Path(intents_path)

        self.documents: list[tuple[list[str], str]] = []
        self.vocabulary: list[str] = []
        self.intents: list[str] = []
        self.intents_responses: dict[str, list[str]] = {}

        self.function_mappings = function_mappings or {}

        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None

        self.lemmatizer = WordNetLemmatizer()
        self.ignore_chars = {'?', '!', '.', ',', ':', ';', '…', '—', '-'}

        # where to save artifacts (model + dims)
        self.out_dir = Path(out_dir) if out_dir else self.intents_path.parent
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def tokenize_and_lemmatize(text: str) -> list[str]:
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text, preserve_line=True)
        return [lemmatizer.lemmatize(t.lower()) for t in tokens]


    def bag_of_words(self, tokens: list[str]) -> list[int]:
        # vector of 0/1 indicating presence of known vocab tokens
        return [1 if w in tokens else 0 for w in self.vocabulary]


    def parse_intents(self):
        if not self.intents_path.exists():
            raise FileNotFoundError(f"Intents file not found: {self.intents_path}")

        with open(self.intents_path, 'r', encoding='utf-8') as f:
            intents_data = json.load(f)

        # Build intents, responses, vocabulary, and (tokens, tag) documents
        for intent in intents_data['intents']:
            tag = intent['tag']
            if tag not in self.intents:
                self.intents.append(tag)
                self.intents_responses[tag] = intent.get('responses', [])

            for pattern in intent['patterns']:
                pattern_tokens = self.tokenize_and_lemmatize(pattern)
                # keep punctuation out of vocabulary
                pattern_tokens = [w for w in pattern_tokens if w not in self.ignore_chars]
                self.vocabulary.extend(pattern_tokens)
                self.documents.append((pattern_tokens, tag))

        self.vocabulary = sorted(set(self.vocabulary))
        self.intents = sorted(set(self.intents))

    def prepare_data(self):
        if not self.documents:
            raise RuntimeError("No documents. Call parse_intents() first.")

        bags: list[list[int]] = []
        indices: list[int] = []

        for tokens, tag in self.documents:
            bow = self.bag_of_words(tokens)
            bags.append(bow)
            indices.append(self.intents.index(tag))

        X = np.array(bags, dtype=np.float32)
        y_idx = np.array(indices, dtype=np.int64)

        # one-hot targets for Keras softmax
        num_classes = len(self.intents)
        y = np.zeros((y_idx.shape[0], num_classes), dtype=np.float32)
        y[np.arange(y_idx.shape[0]), y_idx] = 1.0

        self.X, self.y = X, y

    def train_model(self, epochs=200, batch_size=5, verbose=0):
        if self.X is None or self.y is None:
            raise RuntimeError("Data not prepared. Call prepare_data() first.")

        input_size = self.X.shape[1]
        output_size = self.y.shape[1]

        self.model = ChatbotModelKeras.build(input_size, output_size)
        self.model.fit(self.X, self.y, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def save_model(self, model_path: Path, dims_path: Path):
        if self.model is None:
            raise RuntimeError("No model to save. Train or load a model first.")
        self.model.save(model_path)
        with open(dims_path, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'input_size': int(self.X.shape[1]) if self.X is not None else None,
                    'output_size': int(len(self.intents)),
                    'intents': self.intents,
                    'vocabulary': self.vocabulary,
                },
                f,
                ensure_ascii=False,
                indent=2
            )

    def load_model(self, model_path: Path, dims_path: Path):
        with open(dims_path, 'r', encoding='utf-8') as f:
            dims = json.load(f)

        self.intents = dims['intents']
        self.vocabulary = dims['vocabulary']
        self.model = load_model(model_path)

    # ---------------------------- Inference ---------------------------------
    def _clean_input(self, sentence: str) -> list[str]:
        tokens = word_tokenize(sentence, preserve_line=True)
        tokens = [self.lemmatizer.lemmatize(w.lower()) for w in tokens]
        tokens = [w for w in tokens if w not in self.ignore_chars]
        return tokens

    def _predict_distribution(self, sentence: str) -> np.ndarray:
        tokens = self._clean_input(sentence)
        bow = np.array([self.bag_of_words(tokens)], dtype=np.float32)
        preds = self.model.predict(bow, verbose=0)[0]  # shape: (num_classes,)
        return preds

    def process_message(self, input_message: str, threshold: float = 0.25) -> str | None:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() or train_model() first.")

        probs = self._predict_distribution(input_message)
        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        top_tag = self.intents[top_idx]

        # Optional: invoke a function mapping if the predicted intent has one
        if top_tag in self.function_mappings:
            try:
                self.function_mappings[top_tag]()  # side effect (e.g., print stocks)
            except Exception:
                pass

        # Thresholded fallback
        if top_prob < threshold:
            # try explicit fallback intents if present
            for fallback_tag in ("no-response", "neutral-response", "fallback_unknown"):
                if fallback_tag in self.intents_responses and self.intents_responses[fallback_tag]:
                    return random.choice(self.intents_responses[fallback_tag])
            return "I'm not sure I understood that. Could you rephrase?"

        responses = self.intents_responses.get(top_tag, [])
        return random.choice(responses) if responses else "…"  # safe default


# ------------------------------- Example usage -------------------------------
def get_stocks():
    stocks = ['AAPL', 'META', 'NVDA', 'GS', 'MSFT']
    print(random.sample(stocks, 3))


if __name__ == "__main__":
    # Choose domain interactively (same feel as your previous script)
    intent_type = input("Enter the intent type (e.g. 'mental', 'stocks'): ").strip() or "mental"

    base = Path(__file__).resolve().parent
    intents_path = base / f"intents_{intent_type}.json"
    
    # if i want to save artifacts in a specific dir
    # artifacts_dir = base / "Chatbot"
    # artifacts_dir.mkdir(exist_ok=True)

    model_path = base / f"chatbot_model_{intent_type}.h5"
    dims_path = base / f"dimensions_{intent_type}.json"

    assistant = ChatbotAssistantKeras(
        intents_path=intents_path,
        function_mappings={'stocks': get_stocks},
        out_dir=base
    )

    # --- TRAIN BLOCK (uncomment to train) ---
    assistant.parse_intents()
    assistant.prepare_data()
    assistant.train_model(epochs=200, batch_size=5, verbose=0)
    assistant.save_model(model_path, dims_path)
    print(f"Trained and saved → {model_path.name}, {dims_path.name}")

    # --- INFER BLOCK (load and chat) ---
    assistant.parse_intents()  # to fill intents_responses even when loading
    assistant.load_model(model_path, dims_path)
    print(f"Go! Bot is running (domain='{intent_type}')")

    try:
        while True:
            msg = input("> ")
            if msg.strip().lower() in ("/quit", "/exit"):
                print("Bye!")
                break
            print(assistant.process_message(msg, threshold=0.25))
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")