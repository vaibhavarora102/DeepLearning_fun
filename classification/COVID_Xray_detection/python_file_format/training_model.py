import tensorflow as tf
import model_generator 

def training_model(training_dir, validation_dir,batch, trainig_len):
    '''
        this function is to fit the model and start the training

        input:
            training_dir : training data directory
            validation_dir : validation data directory
            batch : batch size
            training_len : total size of trainig data

        output:
            starts training
    '''

    train_datagen= tf.keras.preprocessing.image.ImageDataGenerator(
        rescale = 1./255,
        shear_range= 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
    )

    test_dataset= tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_generator= train_datagen.flow_from_directory(
        training_dir,
        target_size =(224,224),
        batch_size = 32,
        class_mode ='binary'
    )

    val_generator= test_dataset.flow_from_directory(
        validation_dir,
        target_size=(224,224),
        batch_size = 32,
        class_mode ='binary'
    )

    print('classes are ', train_generator.class_indices)
    
    model= model_generator.get_model()

    history = model.fit_generator(
        train_generator,
        batch_size = batch,
        steps_per_epoch = trainig_len//batch_size,
        epochs = 10,
        validation_data=val_generator,
        validation_steps = 2
    )



training_dir = 'CovidDataset/Train'
validation_dir = 'CovidDataset/Val'
batch = 32
training_len = 486


training_model(training_dir, validation_dir,batch_size, trainig_len)