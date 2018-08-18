"""
Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
"""

from __future__ import print_function

import io
import random

import numpy as np
from keras.callbacks import LambdaCallback
from keras.layers import Dense, Activation
from keras.layers import LSTM, Embedding
from keras.models import Sequential
from keras.optimizers import RMSprop

with io.open('sample_data.txt', encoding='utf-8') as f:
    text = f.read().lower()

output_file = io.open('output_file.txt', encoding='utf8', mode='w')

print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen), dtype=np.uint8)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t] = char_indices[char]
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(Embedding(input_dim=len(chars)+1, output_dim=25, input_shape=(maxlen,)))
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    print(probas)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        output_file.write('----- diversity:' + str(diversity))

        generated = ''
        sample_sentence = text[start_index: start_index + maxlen]
        generated += sample_sentence
        output_file.write('\n----- Generating with seed: "' + sample_sentence + '"\n')
        output_file.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen))
            for t, char in enumerate(sample_sentence):
                x_pred[0, t] = char_indices[char]

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sample_sentence = sample_sentence[1:] + next_char

            output_file.write(next_char)

        output_file.write(str('\n'))

    output_file.flush()


print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y,
          batch_size=128,
          epochs=10,
          callbacks=[print_callback])
