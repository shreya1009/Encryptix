import os
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import pickle

# Load the InceptionV3 model
inception_model = InceptionV3(weights='imagenet')
model_new = Model(inception_model.input, inception_model.layers[-2].output)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img

def encode_image(img):
    img = preprocess_image(img)
    feature_vector = model_new.predict(img)
    feature_vector = np.reshape(feature_vector, feature_vector.shape[1])
    return feature_vector

# Load and preprocess the captions
def load_captions(file_path):
    captions = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip().split('\t')
            image_id, caption = line[0], line[1]
            image_id = image_id.split('.')[0]
            if image_id not in captions:
                captions[image_id] = []
            captions[image_id].append(caption)
    return captions

def preprocess_captions(captions):
    all_captions = []
    table = str.maketrans('', '', string.punctuation)
    for key, caps in captions.items():
        for cap in caps:
            cap = cap.lower().translate(table)
            cap = 'startseq ' + cap + ' endseq'
            all_captions.append(cap)
    return all_captions

# Specify paths
images_path = 'Flickr8k_Dataset/Flicker8k_Dataset'
captions_path = 'Flickr8k_Dataset/Flickr8k_text/Flickr8k.token.txt'

# Load the dataset
image_captions = load_captions(captions_path)
all_captions = preprocess_captions(image_captions)

# Tokenize the captions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1

max_length = max(len(cap.split()) for cap in all_captions)

def data_generator(image_captions, tokenizer, max_length, num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n = 0
    while True:
        for key, desc_list in image_captions.items():
            n += 1
            photo = encode_image(os.path.join(images_path, key + '.jpg'))
            for desc in desc_list:
                seq = tokenizer.texts_to_sequences([desc])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == num_photos_per_batch:
                yield [np.array(X1), np.array(X2)], np.array(y)
                X1, X2, y = list(), list(), list()
                n = 0

# Define the model
inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

decoder1 = tf.keras.layers.add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
epochs = 20
steps = len(image_captions)
for i in range(epochs):
    generator = data_generator(image_captions, tokenizer, max_length, 1)
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)

# Save the model and tokenizer
model.save('model.h5')
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Function to generate captions for new images
def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word[yhat]
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    final_caption = in_text.split()
    final_caption = final_caption[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption

# Load the model and tokenizer for inference
model = load_model('model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Test the model on a new image
img_path = os.path.join(images_path, '1000268201_693b08cb0e.jpg')  # replace with the path to your test image
photo = encode_image(img_path)
photo = photo.reshape((1, 2048))
caption = generate_caption(model, tokenizer, photo, max_length)
print("Caption:", caption)

# Display the image
img = Image.open(img_path)
plt.imshow(img)
plt.axis('off')
plt.show()
