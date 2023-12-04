import re
import emoji
import numpy as np
from dist_list import (
    emoji_vietnamese_dist,
    emoticon_dist,
    spell_correct_dist,
    vnese_stop_words_list,
    emotion_dist,
)
from datasets import load_dataset, Dataset

# This is the preprocessing class (preproc_class)
# Input a dataframe using df and header or a list using lst
# The step and order of data preprocessing can be changed using a list (i.e [1, 3, 4, 6])
# 1. Standardize words (remove all words contains numbers, punctuations, duplicated words)
# 2. Convert emojis and emoticons into text (in english)
# 3. Convert converted emojis and emoticons into vietnamese
# 4. Fix misspelling, abbreviations 
# 5. Remove some particular words (just pronouns)
# 6. Remove one-length words (word contains one character)

class preproc_class:
    def __init__(self, df=None, header=None, lst=None):
        self.lst = lst
        self.df = df
        self.header = header
        self.punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~₫"""

        if self.df is not None:
            self.sentenceList = df[header].to_numpy()
        elif self.lst is not None:
            self.sentenceList = lst
        else:
            raise Exception("No dataframe or list specified")

    def standardize_words(self, sentenceList):
        punc_set = set(self.punctuation)

        # Remove all numbers contain words
        standardize_words = [
            " ".join(
                [
                    word.lower()
                    for word in str(sentence).split()
                    if not any(char.isdigit() for char in word)
                ]
            )
            for sentence in sentenceList
        ]

        # Remove all duplicated characters (besides emojis)
        remove_duplicated = [
            " ".join(
                [
                    re.sub(r"(.)\1+", r"\1", word)
                    if emoji.emoji_count(word) == 0
                    else word
                    for word in sentence.split()
                ]
            )
            for sentence in standardize_words
        ]

        remove_punctuation = []

        # Remove all punctuation symbols using the puctuation set
        for sentence in remove_duplicated:
            sentence_placeholder = sentence.split()
            words_list = []

            for word in sentence_placeholder:
                if punc_set.isdisjoint(word):
                    words_list.append(word)
                elif word in emoticon_dist:
                    words_list.append(word)

            remove_punctuation.append(" ".join(words_list))

        return remove_punctuation

    def emoji_to_text(self, sentenceList):
        # Using the emoji library to convert all emojis into text
        sentence_emo_to_text = [
            emoji.demojize(sentence, delimiters=(":", ": "))
            for sentence in sentenceList
        ]
        sentence_emoticon_to_text = [
            " ".join(
                [
                    word if word not in emoticon_dist else f"{emoticon_dist[word]}"
                    for word in sentence.split()
                ]
            )
            for sentence in sentence_emo_to_text
        ]

        return sentence_emoticon_to_text

    def emoji_text_to_vietnamese(self, sentenceList):
        # Using a handmade dictionary to convert all emoji texts into vietnamese
        sentence_emoji_to_vnese = [
            " ".join(
                [
                    word
                    if word not in emoji_vietnamese_dist
                    else emoji_vietnamese_dist[word]
                    for word in sentence.split()
                ]
            )
            for sentence in sentenceList
        ]

        return sentence_emoji_to_vnese

    def spelling_check(self, sentenceList):
        # Using a handmade dictionary to check words and correct them
        spelling_fix = [
            " ".join(
                [
                    word if word not in spell_correct_dist else spell_correct_dist[word]
                    for word in sentence.split()
                ]
            )
            for sentence in sentenceList
        ]

        return spelling_fix

    def stop_word_removal(self, sentenceList):
        # Using a premade dictionary to remove all stop words
        remove_stop_word = [
            " ".join(
                [word for word in sentence.split() if word not in vnese_stop_words_list]
            )
            for sentence in sentenceList
        ]

        return remove_stop_word

    def remove_length_one(self, sentenceList):
        # Remove words with one character
        one_length_gone = [
            " ".join(
                [
                    word
                    for word in sentence.split()
                    if len(word) >= 2 or emoji.emoji_count(word) > 0
                ]
            )
            for sentence in sentenceList
        ]

        return one_length_gone

    def __str__(self):
        return f"List length: {len(self.sentenceList)}"

    def preprocessing(self, num_list=[1,2,3,4,5,6]):
        # Change steps into a list for easy customtization
        list_placeholder = self.sentenceList

        for step in num_list:
            if step == 1:
                list_placeholder = self.standardize_words(list_placeholder)
            if step == 2:
                list_placeholder = self.emoji_to_text(list_placeholder)
            if step == 3:
                list_placeholder = self.emoji_text_to_vietnamese(list_placeholder)
            if step == 4:
                list_placeholder = self.spelling_check(list_placeholder)
            if step == 5:
                list_placeholder = self.stop_word_removal(list_placeholder)
            if step == 6:
                list_placeholder = self.remove_length_one(list_placeholder)

        return list_placeholder
    
    # def preprocessing(self):
    #     return self.spelling_check(self.emoji_text_to_vietnamese(self.emoji_to_text(self.standardize_words(self.sentenceList))))

def proc_df(df, label, content, num_list=[1,2,3,4,5,6]):
    df_ph = df.copy(deep=True)

    df_ph[label] = [emotion_dist[emotion] for emotion in df_ph[label]]
    df_ph[content] = preproc_class(df_ph, "Sentence").preprocessing(num_list)

    df_ph = df_ph[[content, label]]
    df_ph.columns = ["text", "label"]

    return df_ph
