import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.applications import VGG16, ResNet50, ResNet101
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Model
import keras
from keras.layers import Input, Dropout, Conv2D, LeakyReLU, MaxPool2D, MaxPooling2D, BatchNormalization, Flatten, Dense, ReLU, GlobalAveragePooling2D
from keras.models import Model



CIFAR_SHAPE = (32, 32, 3)


def conv_block(input_tensor, filters, kernel_size, strides=(2, 2), use_bias=True, name=None):
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=use_bias, name=name + '_conv')(input_tensor)
    x = layers.BatchNormalization(name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    return x

def identity_block(input_tensor, filters, kernel_size, name=None):
    x = layers.Conv2D(filters, kernel_size=kernel_size, padding='same', use_bias=True, name=name + '_conv1')(input_tensor)
    x = layers.BatchNormalization(name=name + '_bn1')(x)
    x = layers.Activation('relu', name=name + '_relu1')(x)

    x = layers.Conv2D(filters, kernel_size=kernel_size, padding='same', use_bias=True, name=name + '_conv2')(x)
    x = layers.BatchNormalization(name=name + '_bn2')(x)

    x = layers.add([x, input_tensor], name=name + '_add')
    x = layers.Activation('relu', name=name + '_relu2')(x)
    return x

def RESNET50Base(input_shape, num_classes, dense_units=512, lr_init=0.001, sgd=False, weight_path=None):
    input_layer = layers.Input(shape=input_shape)
    x = conv_block(input_layer, 64, 7, strides=(2, 2), use_bias=True, name='conv1')
    x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)

    # Residual blocks
    x = conv_block(x, 64, 3, strides=(1, 1), name='conv2_block1')
    x = identity_block(x, 64, 3, name='conv2_block2')
    x = identity_block(x, 64, 3, name='conv2_block3')

    x = conv_block(x, 128, 3, name='conv3_block1')
    x = identity_block(x, 128, 3, name='conv3_block2')
    x = identity_block(x, 128, 3, name='conv3_block3')
    x = identity_block(x, 128, 3, name='conv3_block4')

    x = conv_block(x, 256, 3, name='conv4_block1')
    x = identity_block(x, 256, 3, name='conv4_block2')
    x = identity_block(x, 256, 3, name='conv4_block3')
    x = identity_block(x, 256, 3, name='conv4_block4')
    x = identity_block(x, 256, 3, name='conv4_block5')
    x = identity_block(x, 256, 3, name='conv4_block6')

    x = conv_block(x, 512, 3, name='conv5_block1')
    x = identity_block(x, 512, 3, name='conv5_block2')
    x = identity_block(x, 512, 3, name='conv5_block3')

    # Fully Connected Layers
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(units=dense_units, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.5)(x)

    output_layer = layers.Dense(units=num_classes, activation='softmax', name='predictions')(x)

    model = Model(input_layer, output_layer)

    if sgd:
        opt = SGD(learning_rate=lr_init, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        opt = Adam(learning_rate=lr_init)
    
    model.compile(optimizer=opt, loss=categorical_crossentropy, metrics=['accuracy'])
    print(f"Loading weights from {weight_path}")
    if weight_path is not None:
        model.load_weights(weight_path)
    return model

def get_RESNET50_CIFAR100(input_shape=(32, 32, 3), num_classes=100, dense_units=512, lr_init=0.001, sgd=False):
    return RESNET50Base(input_shape, num_classes, dense_units, lr_init, sgd)

def get_RESNET50_CIFAR10(input_shape=(32, 32, 3), num_classes=10, dense_units=512, lr_init=0.001, sgd=False):
    return RESNET50Base(input_shape, num_classes, dense_units, lr_init, sgd)

def get_RESNET50_MNIST(input_shape=(28, 28, 1), num_classes=10, dense_units=512, lr_init=0.001, sgd=False):
    return RESNET50Base(input_shape, num_classes, dense_units, lr_init, sgd)

def get_RESNET50_FASHION(input_shape=(28, 28, 1), num_classes=10, dense_units=512, lr_init=0.001, sgd=False):
    return RESNET50Base(input_shape, num_classes, dense_units, lr_init, sgd)

def get_RESNET50_SVHN(input_shape=(32, 32, 3), num_classes=10, dense_units=512, lr_init=0.001, sgd=False):
    return RESNET50Base(input_shape, num_classes, dense_units, lr_init, sgd)

def get_RESNET50_GTSRB(input_shape=(32, 32, 3), num_classes=43, dense_units=512, lr_init=0.001, sgd=False):
    return RESNET50Base(input_shape, num_classes, dense_units, lr_init, sgd)


def VGG16Base(input_shape, num_classes, dense_units=512, lr_init=0.001, sgd=False, weight_path=None):
    input_shape = input_shape
    num_classes = num_classes
    dense_units = dense_units
    lr_init = lr_init
    sgd = sgd

    input_layer = Input(shape=input_shape)
    
    
    # Block 1
    model = Conv2D(filters=64, kernel_size=3, padding='same')(input_layer)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=64, kernel_size=3, padding='same')(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = MaxPool2D(pool_size=2)(model)
    model = Dropout(0.3)(model)
    
    # Block 2
    model = Conv2D(filters=128, kernel_size=3, padding='same')(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=128, kernel_size=3, padding='same')(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = MaxPool2D(pool_size=2)(model)
    model = Dropout(0.4)(model)
    
    # Block 3
    model = Conv2D(filters=256, kernel_size=3, padding='same')(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=256, kernel_size=3, padding='same')(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = Conv2D(filters=256, kernel_size=3, padding='same')(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)
    model = MaxPool2D(pool_size=2)(model)
    model = Dropout(0.4)(model)
    
    # Fully Connected Layers
    model = Flatten()(model)
    model = Dense(units=4096)(model)
    model = BatchNormalization()(model)
    model = ReLU()(model)
    model = Dropout(0.5)(model)
    
    model = Dense(units=4096)(model)
    model = BatchNormalization()(model)
    model = ReLU()(model)
    model = Dropout(0.5)(model)

    output_layer = Dense(units=num_classes, activation='softmax')(model)

    model = Model(input_layer, output_layer)

    if sgd:
        model.compile(optimizer=SGD(learning_rate=lr_init), loss=categorical_crossentropy, metrics='accuracy')
    else:
        model.compile(optimizer=Adam(learning_rate=lr_init, amsgrad=True),
                      loss=categorical_crossentropy, metrics='accuracy')
        

    print(f"Loading weights from {weight_path}")
    if weight_path is not None:
        model.load_weights(weight_path)
    # model.summary()
    return model

def get_VGG16_CIFAR100( input_shape=(32, 32, 3), num_classes=100, dense_units=512, lr_init=0.001, sgd=False):
    return VGG16Base((32, 32, 3), num_classes, dense_units, lr_init, sgd)


def get_VGG16_CIFAR10(input_shape=(32, 32, 3), num_classes=10, dense_units=512, lr_init=0.001, sgd=False):
    return VGG16Base((32, 32, 3), num_classes, dense_units, lr_init, sgd)

def get_VGG16_MNIST(input_shape=(28, 28, 1), num_classes=10, dense_units=512, lr_init=0.001, sgd=False):
    print("VGG16 MNIST model")
    return VGG16Base((28, 28, 1), num_classes, dense_units, lr_init, sgd)

def get_VGG16_FASHION(input_shape=(28, 28, 1), num_classes=10, dense_units=512, lr_init=0.001, sgd=False):
    return VGG16Base((28, 28, 1), num_classes, dense_units, lr_init, sgd)

def get_VGG16_SVHN(input_shape=(32, 32, 3), num_classes=10, dense_units=512, lr_init=0.001, sgd=False):
    return VGG16Base((32, 32, 3), num_classes, dense_units, lr_init, sgd)

def get_VGG16_GTSRB(input_shape=(32, 32, 3), num_classes=43, dense_units=512, lr_init=0.001, sgd=False):
    return VGG16Base((32, 32, 3), num_classes, dense_units, lr_init, sgd)




def extractfeatures_VGG16(input_shape=(32, 32, 3), num_classes=10, dense_units=512, lr_init=0.001, sgd=False):
    base_model = VGG16(weights='imagenet', include_top=False)
    
    for layer in base_model.layers:
        layer.trainable = False
    inputs = Input(shape=input_shape, name='image_input')
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(dense_units, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    feature_extractor = Model(inputs=inputs, outputs=output)
    feature_extractor.compile(optimizer='adam', loss=categorical_crossentropy, metrics=['accuracy'])

    return feature_extractor

def classifier_VGG16(input_shape=(32, 32, 3), num_classes=10, dense_units=512, lr_init=0.001, sgd=False):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    
    # Freeze the layers of VGG16
    for layer in base_model.layers:
        layer.trainable = False
    
    inputs = Input(shape=input_shape, name='image_input')
    x = base_model(inputs, training=False) 
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    
    optimizer = SGD(learning_rate=lr_init) if sgd else Adam(learning_rate=lr_init)
    model.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=['accuracy'])
    
    return model

def extractfeatures_RESNET50(input_shape=(32, 32, 3), num_classes=10, dense_units=512, lr_init=0.001, sgd=False):
    base_model = ResNet50(weights='imagenet', include_top=False)
    
    for layer in base_model.layers:
        layer.trainable = False
    inputs = Input(shape=input_shape, name='image_input')
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(dense_units, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)
    feature_extractor = Model(inputs=inputs, outputs=output)
    feature_extractor.compile(optimizer='adam', loss=categorical_crossentropy, metrics=['accuracy'])

    return feature_extractor



def classifier_VGG16_CIFAR100(input_shape=(32, 32, 3), num_classes=100, dense_units=512, lr_init=0.001, sgd=False):
    return classifier_VGG16(input_shape, num_classes, dense_units, lr_init, sgd)

def classifier_RESNET50_CIFAR100(input_shape=(32, 32, 3), num_classes=100, dense_units=512, lr_init=0.001, sgd=False):
    return classifier_RESNET50(input_shape, num_classes, dense_units, lr_init, sgd)

def extractfeatures_VGG16_CIFAR100(input_shape=(32, 32, 3), num_classes=100, dense_units=512, lr_init=0.001, sgd=False):
    return extractfeatures_VGG16(input_shape, num_classes, dense_units, lr_init, sgd)

def extractfeatures_RESNET50_CIFAR100(input_shape=(32, 32, 3), num_classes=100, dense_units=512, lr_init=0.001, sgd=False):   
    return extractfeatures_RESNET50(input_shape, num_classes, dense_units, lr_init, sgd)

def classifier_RESNET50(input_shape=(32, 32, 3), num_classes=10, dense_units=512, lr_init=0.001, sgd=False):
    base_model = ResNet50(weights="imagenet", include_top=False)
    inputs = Input(shape=input_shape, name='image_input')
    x = base_model(inputs)
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss=categorical_crossentropy, metrics=['accuracy'])
    return model    