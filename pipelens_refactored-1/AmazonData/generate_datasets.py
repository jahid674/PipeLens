from transformers import T5ForConditionalGeneration, T5Tokenizer
import argostranslate.package
import argostranslate.translate
from nltk.corpus import wordnet
import torch
import pandas as pd
import random
import string
from tqdm import tqdm
import concurrent.futures
import os

class AventTranslator:
  def __init__(self):
    device = "cpu"
    model_name = "AventIQ-AI/t5-language-translation"
    self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    self.tokenizer = T5Tokenizer.from_pretrained(model_name)

  def translate_text(self, input_text, target_language="German"):
      device = "cpu"
      formatted_text = f"translate English to {target_language}: {input_text}"
      input_ids = self.tokenizer(formatted_text, return_tensors="pt").input_ids.to(device)
      
      with torch.no_grad():
          output_ids = self.model.generate(input_ids, max_length=50)
      
      return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
  
class ArgosTranslator:
  def __init__(self):
     pass
  def translate_to_french(self, text):
    """Translate a single text to French using Argos Translate."""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    try:
        return argostranslate.translate.translate(text, "en", "fr")
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original if translation fails

class Noise:
    # alpha represents percent of values to add noise to
    def __init__(self, alpha = 0.1):
        print("generating noisy dataset")
        self.percent = alpha
    
    def replace_spaces(self,text):
        result = text.replace(" ", "-")
        return result
    
    def get_word_synonyms(self, word):
      word = word.lower()
      synonyms = []
      synsets = wordnet.synsets(word)
      if (len(synsets) == 0):
          return []
      synset = synsets[0]
      lemma_names = synset.lemma_names()
      for lemma_name in lemma_names:
          lemma_name = lemma_name.lower().replace('_', ' ')
          if (lemma_name != word and lemma_name not in synonyms and " " not in lemma_name):
              synonyms.append(lemma_name)
      return synonyms

        
    def generate_typo(self,text):
        if random.random() <= self.percent:
            print(text)
            text = list(text)
            #num typos
            n_chars_to_flip = round(len(text) * 0.05)

            #characters to add typos to
            pos_error = []
            for i in range(n_chars_to_flip):
                pos_error.append(random.randint(0, len(text) - 1))

            # insert typos
            for pos in pos_error:
                # try-except in case of special characters
                try:
                    # typo error
                    if random.random() <= 0.7:
                      text[pos] = random.choice(string.ascii_lowercase)
                    else:
                      # swap error
                      if pos != 0:
                          text[pos], text[pos-1] = text[pos - 1], text[pos]
                except:
                    break
            # recombine the message into a strng
            text = ''.join(text)
            print(text)
            return text
        else:
            return text
        
def generate_new_asin(first_char):
    # First character is always first_char
    # Remaining 9 characters are random digits or uppercase letters
    chars = string.digits + string.ascii_uppercase
    return first_char + ''.join(random.choice(chars) for _ in range(9))

def generate_historical():
  df = pd.read_csv('AmazonData/products_processed_0.csv')
  original_sample = df.sample(n=2000, random_state=42)
  typo_prob = 0.9
  synonym_prob = 0.1
  
  noisy_sample = original_sample.copy()
  
  # Create mapping dataframe
  asin_mapping = pd.DataFrame(columns=['english_asin', 'noisy_asin'])
  
  # Generate new ASINs
  new_asins = []
  
  for idx, row in noisy_sample.iterrows():
      new_asin = generate_new_asin('E')
      new_asins.append(new_asin)
      
      # Add to mapping
      asin_mapping = pd.concat([asin_mapping, pd.DataFrame({
          'english_asin': [row['asin']], 
          'noisy_asin': [new_asin]
      })], ignore_index=True)
  
  # Replace ASINs in noisy dataset
  noisy_sample['asin'] = new_asins
  
  # Apply noise to titles
  noisy_sample['title'] = noisy_sample['title'].apply(
      lambda text: apply_noise(text, typo_prob, synonym_prob)
  )
  
  # Save the three datasets
  original_sample.to_csv('AmazonData/historical/original_sample.csv', index=False)
  noisy_sample.to_csv('AmazonData/historical/noisy_sample.csv', index=False)
  asin_mapping.to_csv('AmazonData/historical/asin_mapping.csv', index=False)
  
  print(f"Created three CSV files:")
  print(f"1. original_sample.csv - {len(original_sample)} original sampled products")
  print(f"2. noisy_sample.csv - {len(noisy_sample)} noisy products with new ASINs")
  print(f"3. asin_mapping.csv - Mapping between original and noisy ASINs")
  
  return original_sample, noisy_sample, asin_mapping

def generate_failing():
    germanTranslator = AventTranslator()
    frenchTranslator = ArgosTranslator()
    df = pd.read_csv('AmazonData/products_processed_0.csv')
    typo_prob = 0
    synonym_prob = 0.1
    
    multilingual = df.copy()
    
    # Create mapping dataframe
    asin_mapping = pd.DataFrame(columns=['english_asin', 'multilingual_asin'])
    
    # Generate multilingual ASINs and titles
    new_asins = []
    new_titles = []
    
    for idx, row in multilingual.iterrows():
        r = random.random()
        original_text = row['title']
        if r < 0.1:
            new_asin = generate_new_asin('E')
            new_title = apply_noise(original_text, 0.9, 0)
        elif r < 0.5:
            new_asin = generate_new_asin('G')
            new_title = germanTranslator.translate_text(original_text)
        else:
            new_asin = generate_new_asin('F')
            new_title = frenchTranslator.translate_to_french(original_text)
        
        new_asins.append(new_asin)
        new_titles.append(new_title)
        # Add to mapping
        asin_mapping = pd.concat([asin_mapping, pd.DataFrame({
            'english_asin': [row['asin']], 
            'multilingual_asin': [new_asin]
        })], ignore_index=True)
    
    multilingual['asin'] = new_asins
    multilingual['title'] = new_titles
    
    # Synonyms in english products
    df['title'] = df['title'].apply(
        lambda text: apply_noise(text, typo_prob, synonym_prob)
    )
    
    # Save the three datasets
    df.to_csv('AmazonData/failing/english_products.csv', index=False)
    multilingual.to_csv('AmazonData/failing/multilingual_products.csv', index=False)
    asin_mapping.to_csv('AmazonData/failing/asin_mapping.csv', index=False)
    
    print(f"Created three CSV files:")
    print(f"1. {len(df)} english products")
    print(f"2. {len(multilingual)} multilingual products")
    print(f"3. asin_mapping.csv - Mapping between products")
    
    return df, multilingual, asin_mapping

def apply_noise(text, typo_prob, synonym_prob):   
    words = text.split()
    noisy_words = []
    noise = Noise(typo_prob)
    
    for word in words:
        r = random.random()
        if r < synonym_prob:
            synonyms = noise.get_word_synonyms(word)
            if len(synonyms) > 0:
                noisy_words.append(random.choice(synonyms))
            else:
                noisy_words.append(word)
        else:
            # Keep the original word
            noisy_words.append(word)
    
    return noise.generate_typo(' '.join(noisy_words))
    
if __name__ == "__main__":
   generate_failing()