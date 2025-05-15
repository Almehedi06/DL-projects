import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(data_dir, input_shape=(150, 150), batch_size=32):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2,
                                       zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=input_shape,
                                                        batch_size=batch_size,
                                                        class_mode='binary')

    test_generator = test_datagen.flow_from_directory(test_dir,
                                                      target_size=input_shape,
                                                      batch_size=batch_size,
                                                      class_mode='binary')

    return train_generator, test_generator
