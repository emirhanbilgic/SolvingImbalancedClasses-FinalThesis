import os
import pandas as pd
import numpy as np
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import random

from google.colab import drive
drive.mount('/content/drive')

# Set directories
base_dir = '/content/drive/MyDrive/alzheimer'
train_dir = os.path.join(base_dir, 'train')
train_dir_copy = os.path.join(base_dir, 'train_copy')
test_dir = os.path.join(base_dir, 'test')

# Make a copy of the train directory
if os.path.exists(train_dir_copy):
    shutil.rmtree(train_dir_copy)
shutil.copytree(train_dir, train_dir_copy)

# Set model parameters
img_width, img_height = 224, 224 #default for DenseNet
batch_size = 32

# Create a DataFrame to keep track of augmentation and deletion
df = pd.DataFrame(columns=['Image', 'Class', 'AugCount', 'Deleted'])

# Create ImageDataGenerators
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Important for correct label correspondence
)

# Function to update train_generator with the modified training dataset
def update_train_generator():
    global train_generator
    train_generator = train_datagen.flow_from_directory(
        train_dir_copy,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical'
    )

num_classes = 4  # Number of grades

# Load the base model
base_model = DenseNet169(weights='imagenet', include_top=False,
                         input_shape=(img_width, img_height, 3))

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully-connected layer and a logistic layer with num_classes nodes (one for each class)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Model definition
model = Model(inputs=base_model.input, outputs=predictions)

# compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Initialize accuracy
accuracy = 0
class_names = list(train_generator.class_indices.keys())

while accuracy < 0.6:

    base_model = DenseNet169(weights='imagenet', include_top=False,
                            input_shape=(img_width, img_height, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_generator,
              steps_per_epoch=train_generator.n // train_generator.batch_size,
              epochs=3
              )

    # Make predictions on the test set
    test_predictions = model.predict(test_generator)
    pred_classes = np.argmax(test_predictions, axis=-1)
    test_labels = test_generator.labels
    
    # Calculate accuracy
    accuracy = np.mean(pred_classes == test_labels)
    print(f'Current accuracy: {accuracy}')

    # Get the predicted class labels for the current batch
    batch_pred_labels = [class_names[pred_class] for pred_class in pred_classes]

    # Print the predicted class labels
    print("Predictions:", batch_pred_labels)

    if accuracy < 0.6:
        wrong_predictions = np.where(pred_classes != test_labels)[0]
        for idx in wrong_predictions:
            true_class = test_labels[idx]
            predicted_class = pred_classes[idx]
            true_class_image = test_generator.filenames[idx]

            # Extract the actual directory name where the original image resides
            true_class_dir = os.path.dirname(true_class_image).split('/')[-1]

            # Use true_class_dir instead of the numerical true_class when defining the path
            true_class_dir = os.path.join(train_dir_copy, true_class_dir)

            # Check if the directory exists, if not create it
            if not os.path.exists(true_class_dir):
                os.makedirs(true_class_dir)

            # Determine the filenames of the images in the true class directory
            filenames = os.listdir(true_class_dir)

            # Filter the filenames to include only the non-augmented images
            non_augmented_filenames = [filename for filename in filenames if 'aug' not in filename]

            # Randomly choose a non-augmented image for augmentation
            if non_augmented_filenames:
                chosen_filename = random.choice(non_augmented_filenames)
            else:
                chosen_filename = random.choice(filenames)  # If all images are already augmented, choose any image

            # Augment the chosen image and save it in the true class directory
            img = load_img(os.path.join(true_class_dir, chosen_filename))
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            # Find the augmentation count of the chosen image
            aug_count = int(chosen_filename.split('_')[-1].split('.')[0].replace('aug', '')) if 'aug' in chosen_filename else 0

            # Augment the chosen image and save it in the true class directory
            for batch in train_datagen.flow(x, batch_size=1,
                                            save_to_dir=true_class_dir,
                                            save_prefix=chosen_filename.split('.')[0] + '_aug' + str(aug_count + 1),
                                            save_format='jpeg'):
                break 

            # Update the DataFrame with the augmented image details
            new_row = pd.DataFrame({"Image": [chosen_filename],
                        "Class": [true_class],
                        "AugCount": [aug_count + 1],
                        "Deleted": [False]}).astype({"Deleted": bool})

            df = pd.concat([df, new_row], ignore_index=True).astype({"Deleted": bool})
            predicted_class_dir = os.path.join(train_dir_copy, class_names[predicted_class])

            # Determine the filenames of the images in the predicted class directory
            predicted_filenames = os.listdir(predicted_class_dir)

            # Filter the filenames to include only the augmented images
            augmented_filenames = [filename for filename in predicted_filenames if 'aug' in filename]

            # If there are augmented images in the predicted class directory, delete one
            if augmented_filenames:
                chosen_delete_filename = random.choice(augmented_filenames)
                delete_path = os.path.join(predicted_class_dir, chosen_delete_filename)
                os.remove(delete_path)

                # Mark the deleted image in the DataFrame
                df.loc[(df['Image'] == chosen_delete_filename) & (df['Class'] == predicted_class), 'Deleted'] = True

        # Remove deleted images from DataFrame
        df = df[df['Deleted'] == False]

        # Update train_generator with the modified training dataset
        update_train_generator()

# Print the final DataFrame
print(df)
