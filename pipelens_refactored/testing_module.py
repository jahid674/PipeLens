import pandas as pd
from pipeline_component.lowercase_handler import LowercaserHandler
from pipeline_component.punctuation_handler import PunctuationRemoverHandler
from pipeline_component.whitespace_handler import WhitespaceCleanerHandler

'''df = pd.DataFrame({
    'text': [
        " Hello, World!  ",
        "This    is   a TEST...   ",
        "Pre-processing    is Important!!!"
    ]
})
y_dummy = [0, 1, 1]
sensitive_dummy = pd.Series(["A", "B", "A"])

config = {'text_column': 'text'}

lowercase_handler = LowercaserHandler(config)
df, y_dummy, sensitive_dummy = lowercase_handler.apply(df, y_dummy, sensitive_dummy)

punct_handler = PunctuationRemoverHandler(config)
df, y_dummy, sensitive_dummy = punct_handler.apply(df, y_dummy, sensitive_dummy)

whitespace_handler = WhitespaceCleanerHandler(config)
df, y_dummy, sensitive_dummy = whitespace_handler.apply(df, y_dummy, sensitive_dummy)

print(df)'''


'''import pandas as pd
from pipeline_component.language_detector_handler import LanguageDetectorHandler
from pipeline_component.language_translator_handler import LanguageTranslatorHandler

# Sample DataFrame with multilingual text
df = pd.DataFrame({
    'text': [
        "লন্ডনে মানবেতর জীবনযাপন!",
        "Este es un texto en español.",
        "Das ist ein deutscher Text.",
        "This is an English sentence.",
    ]
})
y_dummy = [1, 0, 1, 0]
sensitive_dummy = pd.Series(["Group1", "Group2", "Group1", "Group2"])
print(df)
# ---- Step 1: Language Detection ----
config_detect = {
    'text_column': 'text',
    'result_column': 'language'
}
detector = LanguageDetectorHandler(config_detect)
df, y_dummy, sensitive_dummy = detector.apply(df, y_dummy, sensitive_dummy)

# ---- Step 2: Language Translation ----
config_translate = {
    'text_column': 'text',
    'source_lang': 'auto',
    'target_lang': 'en'
}
translator = LanguageTranslatorHandler(config_translate)
df, y_dummy, sensitive_dummy = translator.apply(df, y_dummy, sensitive_dummy)
# ---- Output the result ----
print(df)'''

import pandas as pd
from pipeline_component.tokenizer_handler import TokenizerHandler
from pipeline_component.stopword_handler import StopwordRemoverHandler
from pipeline_component.deduplication_handler import DeduplicatorHandler

# --------------------------
# 1. Sample DataFrame
# --------------------------
data = {
    'text': [
        'The quick brown fox jumps over the lazy dog',
        'The quick brown fox jumps over the lazy dog',  # duplicate row
        'And in the end it does not even matter',
        'This is a test with stopwords in the sentence'
    ],
    'label': [1, 1, 0, 1]
}
df = pd.DataFrame(data)
y_dummy = pd.Series(data['label'])
sensitive_dummy = None

# --------------------------
# 2. Define Config
# --------------------------
config = {
    'text_column': 'text'
}

# --------------------------
# 3. Apply Tokenizer
# --------------------------
token_handler = TokenizerHandler(config)
df, y_dummy, sensitive_dummy = token_handler.apply(df, y_dummy, sensitive_dummy)

# --------------------------
# 4. Apply Stopword Remover
# --------------------------
stopword_handler = StopwordRemoverHandler(config)
df, y_dummy, sensitive_dummy = stopword_handler.apply(df, y_dummy, sensitive_dummy)

config = {
    "subset": None,       # or specify columns like ['id', 'text']
    "verbose": True
}

# --------------------------
# 5. Apply Deduplicator
# --------------------------
dedup_handler = DeduplicatorHandler(config={})
df, y_dummy, sensitive_dummy = dedup_handler.apply(df, y_dummy, sensitive_dummy)

# --------------------------
# 6. Print Final Output
# --------------------------
print("Final cleaned DataFrame:")
print(df)



# 1. Sample mixed DataFrame
data = {
    'age': [25, 30, 25, 30],
    'income': [50000, 60000, 50000, 60000],
    'note': ["High risk", "This is a good customer", "High risk", "High risk"]
}
X = pd.DataFrame(data)
y = pd.Series(["A", "A", "B", "B"])
sensitive = pd.Series(["M", "M", "F", "F"])

print("🔹 Before Processing:")
print(X)

# 2. Tokenization Handler
token_handler = TokenizerHandler(config={'text_column': 'note'})
X, y, sensitive = token_handler.apply(X, y, sensitive)

# 3. Stopword Removal Handler
stop_handler = StopwordRemoverHandler(config={'text_column': 'note'})
X, y, sensitive = stop_handler.apply(X, y, sensitive)

# 4. Deduplication Handler
dedup_handler = DeduplicatorHandler(config={'subset': None, 'verbose': True})
X, y, sensitive = dedup_handler.apply(X, y, sensitive)

print("\n🔹 After Tokenization, Stopword Removal, and Deduplication:")
print(X)
print("Target:", list(y))
print("Sensitive:", list(sensitive))


