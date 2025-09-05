import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Bidirectional, LSTM, TimeDistributed
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Parameters
img_size = 64        # Resize images to 64x64
seq_length = 5       # Number of frames in a sequence (for LSTM)
data_path = r'D:\MSC\sem 3\RP\code\demo\Prototype\data'


# Load images and labels
def load_images(data_path):
    sequences = []    # List of sequences (arrays)
    labels = []       # Corresponding character labels
    skipped = 0      # Count of skipped sequences due to errors
    
    # Force numpy to use float32 for memory efficiency
    np.set_printoptions(precision=3)
    
    # Recursively find all character folders inside Full/Half Sleeves for each age group
    for group in sorted(os.listdir(data_path)):
        group_path = os.path.join(data_path, group)
        if not os.path.isdir(group_path):
            continue
        print(f"Checking group: {group}")
        for sleeve_type in sorted(os.listdir(group_path)):
            sleeve_path = os.path.join(group_path, sleeve_type)
            if not os.path.isdir(sleeve_path):
                continue
            print(f"  Sleeve type: {sleeve_type}")
            for category in sorted(os.listdir(sleeve_path)):
                category_path = os.path.join(sleeve_path, category)
                if not os.path.isdir(category_path):
                    continue
                print(f"    Category: {category}")
                for character in sorted(os.listdir(category_path)):
                    char_path = os.path.join(category_path, character)
                    if not os.path.isdir(char_path):
                        continue
                    images = [img for img in sorted(os.listdir(char_path)) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                    print(f"      Character: {character} - {len(images)} images found")
                    if len(images) < seq_length:
                        print(f"        WARNING: Not enough images for a sequence in {char_path} (found {len(images)}, need {seq_length})")
                        continue
                    # Collect sequential frames in groups of seq_length
                    for i in range(0, len(images) - seq_length + 1, seq_length):
                        seq_images = []
                        from PIL import Image
                        for j in range(seq_length):
                            img_path = os.path.join(char_path, images[i + j])
                            if not os.path.exists(img_path):
                                print(f"        WARNING: File does not exist: {img_path}")
                                continue
                            try:
                                with Image.open(img_path) as pil_img:
                                    pil_img = pil_img.resize((img_size, img_size))
                                    img = np.array(pil_img)
                                    if img.ndim == 2:  # grayscale
                                        img = np.stack([img]*3, axis=-1)
                                    elif img.shape[2] == 4:  # RGBA
                                        img = img[:, :, :3]
                                    img = img / 255.0
                            except Exception as e:
                                print(f"        WARNING: Could not read image {img_path} with PIL: {e}")
                                continue
                            # Ensure float32 for memory efficiency
                            seq_images.append(img.astype(np.float32))
                        if len(seq_images) == seq_length:
                            sequences.append(np.array(seq_images, dtype=np.float32))
                            labels.append(character)
                        else:
                            skipped += 1
    print(f"\nTotal sequences loaded: {len(sequences)}")
    print(f"Sequences skipped due to errors: {skipped}")
    
    # Convert to float32 array for memory efficiency
    sequences_array = np.array(sequences, dtype=np.float32)
    print(f"Final array shape: {sequences_array.shape}")
    print(f"Memory usage: {sequences_array.nbytes / (1024*1024*1024):.2f} GB")
    
    return sequences_array, labels

print("Loading data...")
X, y = load_images(data_path)
print(f"Loaded {len(X)} sequences.")

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)
y_cat = to_categorical(y_enc)

# Create data generator for training
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, X, y, batch_size=16, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.X))
        batch_indexes = self.indexes[start_idx:end_idx]
        
        # Get batch data
        X_batch = np.array([self.X[i] for i in batch_indexes])
        y_batch = np.array([self.y[i] for i in batch_indexes])
        
        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# Free up memory
print("Converting data to float32...")
X = X.astype(np.float32)
print("Converting labels...")
y_cat = y_cat.astype(np.float32)

# Train-test split
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Create generators
print("Creating data generators...")
train_generator = DataGenerator(X_train, y_train, batch_size=16)
val_generator = DataGenerator(X_test, y_test, batch_size=16)

# Free up memory
del X
del y_cat
import gc
gc.collect()

# Model: CNN + Bidirectional LSTM
print("Creating model...")
model = Sequential()

# Apply CNN on each frame via TimeDistributed
model.add(TimeDistributed(Conv2D(32, (3,3), activation='relu'), input_shape=(seq_length, img_size, img_size, 3)))
model.add(TimeDistributed(MaxPooling2D(2,2)))
model.add(TimeDistributed(Conv2D(64, (3,3), activation='relu')))
model.add(TimeDistributed(MaxPooling2D(2,2)))
model.add(TimeDistributed(Flatten()))
model.add(Dropout(0.3))

# Bi-directional LSTM to learn temporal features between frames
model.add(Bidirectional(LSTM(64)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(len(le.classes_), activation='softmax'))

# Compile and train
print("Compiling model...")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

print("\nStarting training...")
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator
)

# Save model
model.save('cnn_bdlstm_model.h5')
print("Model saved as cnn_bdlstm_model.h5")
