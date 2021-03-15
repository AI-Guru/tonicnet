import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import pickle
import music21
import itertools
import os
import shutil
import note_seq
from sklearn.manifold import TSNE


experiment_name = "tonicnet"

# Download the dataset.
os.system('rm Jsb16thSeparated*')
os.system('wget https://github.com/czhuang/JSB-Chorales-dataset/raw/master/Jsb16thSeparated.npz')


# Get highest and lowest pitches.

# Load the data into memory.
data = np.load("Jsb16thSeparated.npz", allow_pickle=True, encoding="bytes")

# Go through all subsets and gather pitches.
pitches = []
for dataset_name, songs in data.items():
    for song in songs:
        pitches.extend(song.flatten())

# Remove NaNs and turn into sorted set.
pitches = [pitch for pitch in pitches if str(pitch) != "nan"]
pitches = sorted(list(set(pitches)))

print("Pitches:", pitches)
min_pitch = int(np.min(pitches))
max_pitch = int(np.max(pitches))
    
print("Lowest pitch: ", min_pitch)
print("Highest pitch:", max_pitch)


# Create vocabulary.
vocabulary = []

# Notes and chord qualities.
notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
chord_qualities = ["major", "minor", "diminished", "augmented"]

# Song start and song end.
vocabulary += ["song_start"]
vocabulary += ["song_end"]

# Add chords.
for pitch, quality in itertools.product(range(0, 12), chord_qualities):
    root_pitch = notes[pitch]
    vocabulary += [f"chord_{root_pitch}_{quality}"]

# Add other chord as placeholder for unknown chords and add chord rest as placeholder for silence.
vocabulary += ["chord_other"]
vocabulary += ["chord_rest"]


# Maps pitch to token.
def pitch_to_token(pitch):
    if str(pitch) != "nan":
        pitch = music21.pitch.Pitch(midi=pitch)
        pitch_token = f"pitch_{pitch.nameWithOctave}"
    else:
        pitch_token = "pitch_rest"
    return pitch_token


# Maps token to pitch.
def token_to_pitch(token):
    assert token.startswith("pitch_"), token
    if token == "pitch_rest":
        return float('NaN')
    else:
        token = token.replace("pitch_", "")
        return music21.pitch.Pitch(token).midi


# Add all pitches.
for pitch in range(min_pitch, max_pitch + 1):
    pitch_token = pitch_to_token(pitch)
    vocabulary += [pitch_token]

# Add pitch rest as placeholder for silence.
vocabulary += ["pitch_rest"]

# Print the vocabulary.
print("Vocabulary:")
print(vocabulary, len(vocabulary))


# Maps chords to tokens.
def chord_to_token(chord):
    chord = music21.chord.Chord([int(pitch) for pitch in chord if str(pitch) != "nan"])
    if len(chord) == 0:
        return "chord_rest"
    else:
        pitch_class = music21.pitch.Pitch(chord.root()).pitchClass
        root_pitch = notes[pitch_class]
        quality = chord.quality
        if quality == "other":
            return "chord_other"
        else:
            return f"chord_{root_pitch}_{quality}"

# Loads a dataset of given name. Creates the dataset if it does not exist yet.
def load_dataset(dataset_name):

    assert dataset_name in ["train", "valid", "test"], dataset_name

    # Get preprocessed dataset from file. If file does not exist, create it.
    dataset_filename = f"dataset_{dataset_name}.p"
    if not os.path.exists(dataset_filename):
        print("Preprocessed data file does not exist. Creating now...")

        # The file that contains songs to be encoded.
        dataset_filename_open = f"dataset_{dataset_name}_open.p"

        # The file that contains encoded songs.
        dataset_filename_temp = f"dataset_{dataset_name}_temp.p"

        # Determine the songs that should be encoded. These go into a specific file for later encoding.
        if not os.path.exists(dataset_filename_open):
            print("Initializing convert job file.")
            songs_to_encode = np.load("Jsb16thSeparated.npz", allow_pickle=True, encoding="bytes")[dataset_name]

            # Doing data augmentation.
            if dataset_name == "train":
                print("Augmenting data set...")
                songs_to_encode = augment_songs(songs_to_encode)

            # Store all songs.
            pickle.dump(songs_to_encode, open(dataset_filename_open, "wb"))
            pickle.dump([], open(dataset_filename_temp, "wb"))
        else:
            print("Convert job file already exists.")

        # Get the songs that should be encoded and the already encoded ones.
        songs_to_encode = pickle.load(open(dataset_filename_open, "rb"))
        encoded_songs = pickle.load(open(dataset_filename_temp, "rb"))

        # Encode the remaining songs.
        while len(songs_to_encode) != 0:
            # Get the first song and encode it.
            song = songs_to_encode[0]
            encoded_song = encode_song(song)

            # Update encoded songs file.
            encoded_songs += [encoded_song]
            pickle.dump(encoded_songs, open(dataset_filename_temp, "wb"))

            # Update open songs file.
            songs_to_encode = songs_to_encode[1:]
            pickle.dump(songs_to_encode, open(dataset_filename_open, "wb"))

            # Statistics.
            print(f"Open: {len(songs_to_encode)} Encoded: {len(encoded_songs)}")

        # Nothing more to convert. Finish here.
        print("Done encoding.")
        shutil.move(dataset_filename_temp, dataset_filename)
        os.remove(dataset_filename_open)

    # File should exist now.
    encoded_songs = pickle.load(open(dataset_filename, "rb"))
    print(f"Got {len(encoded_songs)} encoded songs.")

    # Turn into inputs and outputs.
    print("Going supervised...")
    x_data, r_data, p_data, y_data = to_supervised(encoded_songs)

    # Go tf.data.
    x_dataset = tf.data.Dataset.from_tensor_slices(x_data)
    r_dataset = tf.data.Dataset.from_tensor_slices(r_data)
    p_dataset = tf.data.Dataset.from_tensor_slices(p_data)
    y_dataset = tf.data.Dataset.from_tensor_slices(y_data)
    dataset = tf.data.Dataset.zip((x_dataset, r_dataset, p_dataset, y_dataset))
    return dataset


# Data augmentation.
def augment_songs(songs):

    # Pitches that are allowed. Not allowed pitches will be ignored.
    valid_pitches = list(range(36, 82))

    # Ranges of the individual voices.
    voice_ranges = [list(range(60, 82)), list(range(52, 78)), list(range(45, 73)), list(range(36, 67))]

    # Pitch shift range. One octave down, one octave up, all in between.
    augmentations = list(range(-12, 13))

    # Determine if a pitch is valid.
    def is_pitch_valid(pitch, voice):
        if str(pitch) == "nan":
            return True
        elif pitch in valid_pitches and pitch in voice_ranges[voice]:
            return True
        else:
            return False


    # Do the data augmentation.
    augmented_songs = []
    index = 0
    combinations = len(augmentations) * len(songs)
    for song, augmentation in itertools.product(songs, augmentations):
        index += 1
        print(f"\r{len(augmented_songs)} {index}/{combinations}", end="")
        is_valid = True
        augmented_song = []
        for step in song:
            augmented_step = [pitch + augmentation for pitch in step]
            is_valid_array = [is_pitch_valid(pitch, voice) for voice, pitch in enumerate(augmented_step)]
            is_valid = False not in is_valid_array
            if is_valid:
                augmented_song += [augmented_step]
            else:
                break

        if augmentation == 0:
            assert is_valid
        if is_valid:
            assert len(song) == len(augmented_song)
            augmented_songs += [augmented_song]
    print("")

    # Done with data augmentation.
    print(f"Songs before augmentation {len(songs)} and after {len(augmented_songs)}.")
    return augmented_songs


# Encodes a song.
def encode_song(song):
    encoded_song = []
    encoded_song += [vocabulary.index("song_start")]
    for start_index, end_index in zip(range(0, len(song), 16), range(16, len(song), 16)):
        for chord in song[start_index:end_index]:
            encoded_song += [vocabulary.index(chord_to_token(chord))]
            encoded_song += [vocabulary.index(pitch_to_token(chord[0]))]
            encoded_song += [vocabulary.index(pitch_to_token(chord[1]))]
            encoded_song += [vocabulary.index(pitch_to_token(chord[2]))]
            encoded_song += [vocabulary.index(pitch_to_token(chord[3]))]
    encoded_song += [vocabulary.index("song_end")]
    return np.array(encoded_song).astype("uint8")


# Decodes a sequence to make it human readable.
def decode(sequence):
    return [vocabulary[index] for index in sequence]


# Go "supervised". Prepares datasets for training.
def to_supervised(songs):
    x = [np.array(song[:-1]) for song in songs]
    r = [np.array(compute_repetitions(song[:-1])) for song in songs]
    p = [np.array(compute_positions(song[:-1])) for song in songs]
    y = [np.array(song[1:]) for song in songs]
    x = tf.ragged.constant(x)
    r = tf.ragged.constant(r)
    p = tf.ragged.constant(p)
    y = tf.ragged.constant(y)
    return x, r, p, y


# Computes repetition tokens.
def compute_repetitions(sequence):
    repetitions = [0] * len(sequence)
    for index, element in enumerate(sequence):
        past_index = index - 5
        if past_index >= 0:
            past_element = sequence[past_index]
            token = vocabulary[element]
            past_token = vocabulary[past_element]
            if past_element == vocabulary.index("song_start"):
                repetitions[index] = 0
            elif element == vocabulary.index("song_end"):
                repetitions[index] = 0
            elif past_element == element:
                repetitions[index] = repetitions[past_index] + 1
            else:
                assert past_token.startswith("pitch") == token.startswith("pitch"), (past_token, token, decode(sequence)[:index + 1])
                assert past_token.startswith("chord") == token.startswith("chord"), (past_token, token, decode(sequence)[:index + 1])
                repetitions[index] = 0
            if repetitions[index] > 79:
                repetitions[index] = 79
    return repetitions


# Computes naive positional encodings.
def compute_positions(sequence):
    positions = [0] + [index // 5 % 16 for index in range(len(sequence) - 1)]
    return positions


# Preprocess all datasets.
preprocessing_start_time = time.time()

# Training set.
print("Loading training set...")
dataset_train = load_dataset("train")
print("")

# Validation set.
print("Loading validation set...")
dataset_validate = load_dataset("valid")
print("")

# Testing set.
print("Loading test set...")
dataset_test = load_dataset("test")
print("")

# Done processing. Show time.
preprocessing_end_time = time.time()
preprocessing_duration = str(datetime.timedelta(seconds=preprocessing_end_time - preprocessing_start_time))
print(f"Preprocessing Duration {preprocessing_duration}")


# Get an example and show it.
for sample in dataset_train.take(1):
    x, r, p, y = sample
    x, r, p, y = x[:200], r[:200], p[:200], y[:200]
    plt.figure(figsize=(16, 4))
    plt.plot(x, label="x")
    plt.plot(y, label="y")
    plt.plot(r, label="r")
    plt.plot(p, label="p")
    plt.legend()
    plt.savefig("example_sample.png")
    plt.close()
    print(x.numpy())
    print(r.numpy())
    print(p.numpy())
    print(y.numpy())
    print(decode(x))
    print(decode(y))


# # Find out maximum repetition.

# In[10]:


#for dataset in [dataset_train, dataset_validate, dataset_test]:
#    repetition_max = 0
#    for sample in dataset:
#        _, r, _, _ = sample
#        song_max = np.max(r)
#        if song_max > repetition_max:
#            repetition_max = song_max
#    print(repetition_max)


# Now comes the Deep Learning part.

# Our model.
class TonicNet(tf.keras.Model):

    def __init__(self, vocabulary, repetition_tokens, position_tokens, use_repetition_encoding=True, use_position_encoding=True):
        super().__init__()

        self.vocabulary = vocabulary
        self.use_repetition_encoding = use_repetition_encoding
        self.use_position_encoding = use_position_encoding 

        self.embedding_x = tf.keras.layers.Embedding(input_dim=len(self.vocabulary), output_dim=100)
        if self.use_repetition_encoding:
            self.embedding_r = tf.keras.layers.Embedding(input_dim=repetition_tokens, output_dim=32)
        if self.use_position_encoding:
            self.embedding_p = tf.keras.layers.Embedding(input_dim=position_tokens, output_dim=8)

        self.gru_1 = tf.keras.layers.GRU(100, return_state=True, return_sequences=True)
        self.dropout_1 = tf.keras.layers.Dropout(0.3)
        self.gru_2 = tf.keras.layers.GRU(100, return_state=True, return_sequences=True)
        self.dropout_2 = tf.keras.layers.Dropout(0.3)
        self.gru_3 = tf.keras.layers.GRU(100, return_state=True, return_sequences=True)
        self.dropout_3 = tf.keras.layers.Dropout(0.3)
        self.dense = tf.keras.layers.Dense(len(self.vocabulary), activation=None)

    def call(self, inputs):
        x_batch, r_batch, p_batch = inputs
        assert len(x_batch.shape) == 2, x_batch.shape
        assert len(r_batch.shape) == 2, r_batch.shape
        assert len(p_batch.shape) == 2, p_batch.shape

        # Embed inputs.
        x_embedded = self.embedding_x(x_batch)
        if self.use_repetition_encoding:
            r_embedded = self.embedding_r(r_batch)
        if self.use_position_encoding:
            p_embedded = self.embedding_p(p_batch)

        # Concatenate.
        embedding_concat_array = [x_embedded]
        if self.use_repetition_encoding:
            embedding_concat_array += [r_embedded]
        if self.use_position_encoding:
            embedding_concat_array += [p_embedded]
        y = tf.concat(embedding_concat_array, axis=-1)

        # Go recurrent.
        y, _ = self.gru_1(y)
        y = self.dropout_1(y)
        y, _ = self.gru_2(y)
        y = self.dropout_2(y)
        y, _ = self.gru_3(y)

        # Skip connections.
        skip_concat_array = [y]
        if self.use_repetition_encoding:
            skip_concat_array += [r_embedded]
        if self.use_position_encoding:
            skip_concat_array += [p_embedded]
        y = tf.concat(skip_concat_array, axis=-1)

        # Some dropout.
        y = self.dropout_3(y)

        # Fully connected.
        y = self.dense(y)

        # Done.
        return y

    def train(self, dataset_train, dataset_validate, dataset_test, epochs, weights_name, overwrite=False):

        model_filename = f"{weights_name}-weights.h5"
        if not overwrite and os.path.exists(model_filename):
            print("Model already trained.")
            return {}

        # Create the optimizer.
        optimizer = tf.keras.optimizers.Adam()

        # Create history.
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "train_loss_debug": [],
            "validate_loss": [],
            "validate_accuracy": []
        }
        best_validate_loss = 100000.0

        # Prepare cached and shuffled batches for training set.
        dataset_train_batch = dataset_train.cache().shuffle(4096).batch(1)

        # Train.
        training_start_time = time.time()
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()

            # Train per batch.
            mean_train_loss = tf.keras.metrics.Mean()
            mean_train_accuracy = tf.keras.metrics.Mean()
            batch = 0
            for x, r, p, y in dataset_train_batch:
                batch += 1

                # Convert ragged tensors.
                x = x.to_tensor()
                r = r.to_tensor()
                p = p.to_tensor()
                y = y.to_tensor()

                # Do a training step.
                loss_value, accuracy_value = train_step(self, x, r, p, y, optimizer)

                # Update loss and accuracy.
                mean_train_loss(loss_value)
                mean_train_accuracy(accuracy_value)

                # Finish up batch.
                end = "\r"
                print(f"\rEpoch {epoch} Batch {batch}", end="")

            # Update training loss and accuracy.
            train_loss, train_accuracy = mean_train_loss.result(), mean_train_accuracy.result()
            history["train_loss"] += [train_loss]
            history["train_accuracy"] += [train_accuracy]

            # Update validation loss and accuracy.
            validate_loss, validate_accuracy = evaluate(self, dataset_validate)
            history["validate_loss"] += [validate_loss]
            history["validate_accuracy"] += [validate_accuracy]

            # Finish epoch.
            epoch_end_time = time.time()
            epoch_duration = str(datetime.timedelta(seconds=epoch_end_time - epoch_start_time))
            print("\r")
            print(f"Epoch {epoch} Acc {100.0 * train_accuracy:.4f}% Val-Acc {100.0 * validate_accuracy:.4f}% Duration {epoch_duration}")

            # Save best model.
            if weights_name is not None and validate_loss < best_validate_loss:
                self.save_weights(model_filename)
                best_validate_loss = validate_loss
                print(f"Got a new model with validation loss {validate_loss:.4f} and validation accuracy {100.0 * validate_accuracy:.4f}%.")

        # Finish training.
        training_end_time = time.time()
        training_duration = str(datetime.timedelta(seconds=training_end_time - training_start_time))
        print(f"Training Duration {training_duration}")
        return history

    def generate(self, max_steps, stop_on_end, temperature):

        x = self.vocabulary.index("song_start")
        r = 0
        x_sequence = [x]
        r_sequence = [r]
        p_sequence = []
        initial_state_1 = None
        initial_state_2 = None
        initial_state_3 = None
        for index in range(max_steps):

            if index == 0:
                p = 0
            else:
                p = (index - 1) // 5 % 16
            p_sequence += [p]

            # Make it a batch.
            x_batch = np.array([[x]])
            r_batch = np.array([[r]])
            p_batch = np.array([[p]])
            assert len(x_batch.shape) == 2, x_batch.shape
            assert len(r_batch.shape) == 2, r_batch.shape
            assert len(p_batch.shape) == 2, p_batch.shape

            # Embed inputs.
            x_embedded = self.embedding_x(x_batch)
            assert len(x_embedded.shape) == 3, x_embedded.shape
            if self.use_repetition_encoding:
                r_embedded = self.embedding_r(r_batch)
                assert len(r_embedded.shape) == 3, v.shape
            if self.use_position_encoding:
                p_embedded = self.embedding_p(p_batch)
                assert len(p_embedded.shape) == 3, p_embedded.shape

            # Concatenate.
            embedding_concat_array = [x_embedded]
            if self.use_repetition_encoding:
                embedding_concat_array += [r_embedded]
            if self.use_position_encoding:
                embedding_concat_array += [p_embedded]
            y = tf.concat(embedding_concat_array, axis=-1)

            # Do the RNN part.
            y, initial_state_1 = self.gru_1(y, training=False, initial_state=initial_state_1)
            y, initial_state_2 = self.gru_2(y, training=False, initial_state=initial_state_2)
            y, initial_state_3 = self.gru_3(y, training=False, initial_state=initial_state_3)

            # Skip connections.
            skip_concat_array = [y]
            if self.use_repetition_encoding:
                skip_concat_array += [r_embedded]
            if self.use_position_encoding:
                skip_concat_array += [p_embedded]
            y = tf.concat(skip_concat_array, axis=-1)

            # Do fully connected.
            y = self.dense(y)

            # Sample x.
            y = tf.squeeze(y, 0)
            y = y / temperature
            x = tf.random.categorical(y, num_samples=1)[-1,0].numpy()
            x_sequence += [x]

            # Compute r.
            if len(x_sequence) > 5 and x_sequence[-6] == x:
                new_r = min(79, r + 1)
                r = r_sequence[-5]
            else:
                new_r = 0
                r = 0
            assert isinstance(new_r, int)
            r_sequence += [new_r]

            # Double check.
            assert len(x_sequence) == len(r_sequence)

            # Done.
            if stop_on_end and x == self.vocabulary.index("song_start"):
                break

        # Done generating.
        return x_sequence, r_sequence, p_sequence


@tf.function
def train_step(model, x, r, p, y, optimizer):
    with tf.GradientTape() as tape:
        y_pred = model((x, r, p), training=True)
        loss = compute_loss(y, y_pred)
        accuracy = compute_accuracy(y, y_pred)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, accuracy


def evaluate(model, dataset):
    mean_loss = tf.keras.metrics.Mean()
    mean_accuracy = tf.keras.metrics.Mean()

    for x, r, p, y in dataset.batch(1):
        x = x.to_tensor()
        r = r.to_tensor()
        p = p.to_tensor()
        y = y.to_tensor()
        y_pred = model((x, r, p), training=False)
        loss_value = compute_loss(y_true=y, y_pred=y_pred)
        accuracy_value = compute_accuracy(y_true=y, y_pred=y_pred)
        mean_loss(loss_value)
        mean_accuracy(accuracy_value)

    return float(mean_loss.result()), float(mean_accuracy.result())


@tf.function
def compute_loss(y_true, y_pred):
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=True
    )
    loss = tf.reduce_mean(loss)
    return loss


@tf.function
def compute_accuracy(y_true, y_pred):
    y_pred = tf.nn.softmax(y_pred)
    accuracy = tf.keras.metrics.sparse_categorical_accuracy(
        y_true, y_pred
    )
    accuracy = tf.reduce_mean(accuracy)
    return accuracy


# Method for model creation.
def create_model(use_repetition_encoding=True):
    model = TonicNet(vocabulary, repetition_tokens=80, position_tokens=16, use_repetition_encoding=use_repetition_encoding)
    return model


# Create the model.
model = create_model()

# Run it through a couple of examples.
x = np.array([[5, 5, 5, 5]])
r = np.array([[0, 1, 2, 3]])
p = np.array([[0, 0, 0, 0]])
y = model((x, r, p))
print(y)
x = np.array([[5]])
r = np.array([[0]])
p = np.array([[0]])
y = model((x, r, p))
print(y)

# Generate and render sequence.
generated_sequence, repetitions, positions = model.generate(max_steps=128, stop_on_end=False, temperature=0.5)
plt.plot(generated_sequence, label="sequence")
plt.plot(repetitions, label="repetitions")
plt.plot(positions, label="positions")
plt.legend()
plt.savefig("sequence_before_training.png")
plt.close()


# Train the model.
history = model.train(
    dataset_train,
    dataset_validate,
    dataset_test,
    epochs=75,
    weights_name=experiment_name
)


# Visualize results and save everything.
def render_and_save_history(history):
    if history == {}:
        print("History is empty.")
        return
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["validate_loss"], label="validate_loss")
    plt.legend()
    plt.savefig(f"{experiment_name}-history_loss.png")
    plt.close()

    plt.plot(history["train_accuracy"], label="train_accuracy")
    plt.plot(history["validate_accuracy"], label="validate_accuracy")
    plt.legend()
    plt.savefig(f"{experiment_name}-history_accuracy.png")
    plt.close()

    pickle.dump(history, open(f"{experiment_name}-history.p", "wb"))


# Do the rendering.
render_and_save_history(history)
print("Done.")


# Generate a sample after training.
generated_sequence, repetitions, posititions = model.generate(max_steps=128, stop_on_end=False, temperature=0.7)
plt.plot(generated_sequence, label="sequence")
plt.plot(repetitions, label="repetitions")
plt.plot(positions, label="positions")
plt.legend()
plt.savefig("sequence_after_training.png")
plt.close()


# Some configuration for rendering note sequences.
synth = note_seq.midi_synth.fluidsynth
LENGTH_16TH_120BPM = 0.25 * 60 / 120


# Turn into note sequence.
def to_note_sequence(sequence):
    note_sequence = create_empty_note_sequence()
    notes = [None, None, None, None]
    time = 0.0
    current_voice = 0
    for index in sequence:
        assert len(notes) == 4
        token = vocabulary[index]
        #print(token, time, current_voice, len(note_sequence.notes))
        if token == "song_start":
            pass
        elif token.startswith("chord"):
            current_voice = 0
            note = [None, None, None, None]
        elif token == "song_end":
            print("Reached song end.")
            break
        elif token.startswith("pitch"):
            pitch = token_to_pitch(token)
            velocity = 70
            if str(pitch) == "nan":
                pitch = 0
                velocity = 0

            note = notes[current_voice]
            if note is None or note.pitch != pitch:
                note = note_sequence.notes.add()
                note.start_time = time
                note.end_time = time
                note.pitch = pitch
                note.velocity = velocity
                note.program = 19
                note.instrument = current_voice
                notes[current_voice] = note

            current_voice += 1
            if current_voice == 4:
                for note in notes:
                    if note is not None:
                        note.end_time += LENGTH_16TH_120BPM
                note_sequence.total_time += LENGTH_16TH_120BPM
                time += LENGTH_16TH_120BPM
                current_voice = 0

        else:
            assert False, token
    # Done.
    return note_sequence


# Creates an empty note sequence.
def create_empty_note_sequence(qpm=120.0, total_time=0.0):
    note_sequence = note_seq.protobuf.music_pb2.NoteSequence()
    note_sequence.tempos.add().qpm = qpm
    note_sequence.ticks_per_quarter = note_seq.constants.STANDARD_PPQ
    note_sequence.total_time = total_time
    return note_sequence


# Load and test the model.
model = create_model()
x = np.array([[5]])
r = np.array([[0]])
p = np.array([[0]])
y = model((x, r, p))

# Load the weights.
model.load_weights(f"{experiment_name}-weights.h5")


# Method for sample generation.
def generate():
    print("Generating...")
    temperature = np.random.uniform(0.25, 0.75)
    print(f"Temperature: {temperature}")
    qpm = int(np.random.uniform(65, 85))
    print(f"QPM: {qpm}")

    bars = 64
    steps_per_bar = 16
    voices = 4
    generated_sequence, repetitions, positions = model.generate(max_steps=bars * steps_per_bar * voices, stop_on_end=False, temperature=temperature)
    print("Transforming into note sequence...")
    note_sequence = to_note_sequence(generated_sequence)
    factor = 120.0 / qpm
    for note in note_sequence.notes:
        note.start_time *= factor
        note.end_time *= factor
    note_sequence.total_time *= factor
    note_sequence.tempos[0].qpm = qpm
    #print("Rendering...")
    #note_seq.plot_sequence(note_sequence)
    #print("Playing...")
    #note_seq.play_sequence(note_sequence, synth)
    filename = datetime.datetime.now().strftime("%m%d%Y%H%M%S") + ".mid"
    note_seq.midi_io.note_sequence_to_midi_file(note_sequence, filename)
    print(f"Saved {filename}")
    return filename


# Generate some samples.
for _ in range(10):
    generate()


# Method for rendering embeddings.
def tsne_plot(embeddings, vocabulary, filename):

    embeddings = embeddings.get_weights()[0]

    # Get labels and tokens.
    labels = []
    tokens = []
    for index, word in enumerate(vocabulary):
        if word not in ["song_start", "song_end", "bar_start", "chord_E_augmented"]:
            tokens.append(embeddings[index])
            labels.append(word.replace("pitch_", "").replace("chord_", ""))

    # Run TSNE.
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    values = tsne_model.fit_transform(tokens)

    # Get the coordinates.
    x_list = [value[0] for value in values]
    y_list = [value[1] for value in values]

    # Render.
    plt.figure(figsize=(16, 16))
    for x, y, label in zip(x_list, y_list, labels):
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords="offset points", ha="right", va="bottom")
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.savefig(filename)
    plt.close()


# Plot the vocabulary.
tsne_plot(model.embedding_x, vocabulary, "tsne_vocabulary.png")

# Done!
