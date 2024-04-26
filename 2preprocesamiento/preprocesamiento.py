#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import music21 as m21
import numpy as np
import tensorflow.keras as keras
logger="win10Davids"
DATASET_PATH = "c:/Users/{}/Documents/AplicaTesis/1TESTCUECA/".format(logger)
ITEM_PATH = "c:/Users/{}/Documents/AplicaTesis/1CODEDITEM/".format(logger)
#FILE_DATASET = "c:/Users/{}/Documents/AplicaTesis/1DATASET/aplicatesis".format(logger)
#DICTIO_PATH = "c:/Users/{}/Documents/AplicaTesis/1DICCIONARIO/dictiocueca.json".format(logger)
FILE_DATASET = "c:/Users/{}/Documents/AplicaTesis/1DATASET/cueca".format(logger)
DICTIO_PATH = "c:/Users/{}/Documents/AplicaTesis/1DICCIONARIO/cueca.json".format(logger)
SQ_LENGTH = 64


TIME_DURATIONS = [0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2, 3, 4 ]


# In[2]:


def load_krn(dataset_path):
    """Carga todas las piezas en formato krn  music21.
       :return songs Lista que contiene todas las piezas
    """
    songs = []

    # Recorre todos los archivos en el conjunto de datos y los carga con music21
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:

            # Considera solo archivos kern
            if file[-3:] == "krn":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
                
    # Imprime la lista de canciones
    #print("Lista de canciones cargadas:")
    #for song in songs:
     #   print(song)  # Imprime cada canción

    return songs


# In[3]:


def durations_of_time(song, time_durations):
    for note in song.flatten().notesAndRests:
        if note.duration.quarterLength not in time_durations:
            return False
    return True


# In[4]:


def tonality_transpose(song):
  

    # get key from the song
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]

    # estimate key using music21
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    # get interval for transposition. E.g., Bmaj -> Cmaj
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    # transpose song by calculated interval
    tranposed_song = song.transpose(interval)
    return tranposed_song



# In[5]:


def encode_topicsong(song, time_step=0.125):
  

    encoded_song = []

    for event in song.flatten().notesAndRests:

        # handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi # 60
        # handle rests
        elif isinstance(event, m21.note.Rest):
            symbol = "r"

        # convert the note/rest into time series notation
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):

            # if it's the first time we see a note/rest, let's encode it. Otherwise, it means we're carrying the same
            # symbol in a new time step
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # cast encoded song to str
    encoded_song = " ".join(map(str, encoded_song))

    return encoded_song


# In[6]:


def preprocess(dataset_path):

    # Carga las canciones folklóricas
    print("Cargando canciones...")
    songs = load_krn(dataset_path)
    print(f"Se cargaron {len(songs)} canciones.")
    
    for i, song in enumerate(songs):
        # Filtra las canciones que tienen duraciones no aceptables
        if not durations_of_time(song, TIME_DURATIONS):
            continue

        # Transpone las canciones a Do mayor/La menor
        song = tonality_transpose(song)

        # Codifica las canciones con representación de serie temporal musical
        encoded_song = encode_topicsong(song)  # Esto no se usa en el código actual

        # Guarda las canciones en un archivo de texto
        save_path = os.path.join(ITEM_PATH, str(i))
        with open(save_path, "w") as fp:
             fp.write(encoded_song)  # Esto no se usa en el código actual


# In[7]:


def load(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()
    return song


# In[8]:


def onlyfile_dataset(dataset_path, file_dataset_path, sequence_length):
  

    new_song_delimiter = "/ " * sequence_length
    songs = ""

    # load encoded songs and add delimiters
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter

    # remove empty space from last character of string
    songs = songs[:-1]

    # save string that contains all the dataset
    with open(file_dataset_path, "w") as fp:
        fp.write(songs)

    return songs


# In[9]:


def create_dictio(songs, mapping_path):
   
    mappings = {}

    # identify the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))

    # create mappings
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i

    # save voabulary to a json file
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)


# In[10]:


def convert_songs_to_int(songs):
    int_songs = []

    # load mappings
    with open(DICTIO_PATH, "r") as fp:
        mappings = json.load(fp)

    # transform songs string to list
    songs = songs.split()

    # map songs to int
    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs


# In[11]:


def gtraining_sequences(sequence_length):
    """entrada y salida para entrenamiento,64 notas
    :return inputs (ndarray): datos de entrada,return targets (ndarray): datos desalida
    """
    # carga ataset
    songs = load(FILE_DATASET)
    int_songs = convert_songs_to_int(songs)

    inputs = []
    targets = []

    # genera sequencias
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])

    # one-hot 
    vocabulary_size = len(set(int_songs))
    # inputs size: (# of sequences, sequence length, vocabulary size)
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)

    return inputs, targets


# In[12]:


def main():
    preprocess(DATASET_PATH)
    songs = onlyfile_dataset(ITEM_PATH, FILE_DATASET, SQ_LENGTH)
    create_dictio(songs, DICTIO_PATH)
    inputs, targets = gtraining_sequences(SQ_LENGTH)


if __name__ == "__main__":
    main()



# In[14]:


#!jupyter nbconvert --to script preprocesamiento.ipynb



# In[ ]:





# In[ ]:




