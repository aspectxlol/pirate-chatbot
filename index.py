import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Baca masing-masing file, tambahkan kolom POS'
adjectives = pd.read_csv('adjectives.csv',
                         header=None,
                         names=['Word'],
                         on_bad_lines='skip')
adjectives['POS'] = 'ADJ'

adverbs = pd.read_csv('adverbs.csv',
                      header=None,
                      names=['Word'],
                      on_bad_lines='skip')
adverbs['POS'] = 'ADV'

nouns = pd.read_csv('nouns.csv',
                    header=None,
                    names=['Word'],
                    on_bad_lines='skip')
nouns['POS'] = 'NOUN'

plural_nouns = pd.read_csv('plural-nouns.csv',
                           header=None,
                           names=['Word'],
                           on_bad_lines='skip')
plural_nouns['POS'] = 'PLURAL_NOUN'

verbs = pd.read_csv('verbs.csv',
                    header=None,
                    names=['Word'],
                    on_bad_lines='skip')
verbs['POS'] = 'VERB'

# Gabungkan semua jadi satu dataset
dataset = pd.concat([adjectives, adverbs, nouns, plural_nouns, verbs],
                    ignore_index=True)

#untuk menghapus duplikat (pembaharuan)
dataset = dataset.drop_duplicates()

# Hapus baris yang kosong (NaN)
dataset = dataset.dropna()

# X adalah fitur (kata), y adalah target (POS)
X = dataset['Word']
y = dataset['POS']

# Bagi data 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat vektorisasi kata (pembaharuan)
vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 5))
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Visualisasi training dengan SGDClassifier
epochs = 20
model = SGDClassifier(max_iter=1, warm_start=True, random_state=42)
train_accuracies = []
test_accuracies = []

for epoch in tqdm(range(epochs), desc="Training Progress"):
    model.fit(X_train_vect, y_train)
    train_pred = model.predict(X_train_vect)
    test_pred = model.predict(X_test_vect)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

print(f"Final Training Accuracy: {train_accuracies[-1]*100:.2f}%")
print(f"Final Testing Accuracy: {test_accuracies[-1]*100:.2f}%")

# ========================
# UJI COBA: INPUT USER
# ========================

while True:
    user_input = input("\nMasukkan kata (atau ketik 'exit' untuk keluar): ")
    if user_input.lower() == 'exit':
        print("Terima kasih! Program selesai.")
        break
    user_vect = vectorizer.transform([user_input])
    predicted_pos = model.predict(user_vect)
    print(f"Kata: {user_input} -> POS: {predicted_pos[0]}")
