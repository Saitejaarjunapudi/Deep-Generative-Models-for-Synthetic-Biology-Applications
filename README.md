# Deep Generative Models for Synthetic Biology Applications
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions

# Load and preprocess image
def load_and_process_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Deprocess image
def deprocess_image(img):
    img = img.reshape((img.shape[1], img.shape[2], 3))
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype('uint8')
    return img

# Compute content loss
def content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

# Compute style loss
def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

def style_loss(base_style, gram_target):
    height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)
    return tf.reduce_mean(tf.square(gram_style - gram_target))

# Compute total variation loss
def total_variation_loss(x):
    a = tf.square(x[:, :-1, :-1, :] - x[:, 1:, :-1, :])
    b = tf.square(x[:, :-1, :-1, :] - x[:, :-1, 1:, :])
    return tf.reduce_sum(tf.pow(a + b, 1.25))

# Load VGG19 model for style transfer
def get_model():
    vgg = VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    content_layers = ['block5_conv2'] 
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    outputs = [vgg.get_layer(name).output for name in (style_layers + content_layers)]
    model = Model([vgg.input], outputs)
    return model, style_layers, content_layers

# Compute style and content features
def get_feature_representations(model, content_path, style_path, target_size):
    content_image = load_and_process_image(content_path, target_size)
    style_image = load_and_process_image(style_path, target_size)

    content_outputs = model(content_image)
    style_outputs = model(style_image)

    style_features = [style_layer for style_layer in style_outputs[:len(style_layers)]]
    content_features = [content_layer for content_layer in content_outputs[len(style_layers):]]

    return style_features, content_features

# Perform style transfer
def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights
    outputs = model(init_image)

    style_output_features = outputs[:len(style_layers)]
    content_output_features = outputs[len(style_layers):]

    style_score = tf.add_n([style_loss(comb, gram) for comb, gram in zip(style_output_features, gram_style_features)])
    style_score *= style_weight / len(style_layers)

    content_score = tf.add_n([content_loss(comb, target) for comb, target in zip(content_output_features, content_features)])
    content_score *= content_weight / len(content_layers)

    total_loss = style_score + content_score
    return total_loss

@tf.function
def compute_grads(cfg):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    total_loss = all_loss
    return tape.gradient(total_loss, cfg['init_image']), all_loss

def run_style_transfer(content_path, style_path, iterations=1000, content_weight=1e3, style_weight=1e-2):
    target_size = (224, 224)  # Resize images to this size
    model, style_layers, content_layers = get_model()
    for layer in model.layers:
        layer.trainable = False

    style_features, content_features = get_feature_representations(model, content_path, style_path, target_size)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

    init_image = load_and_process_image(content_path, target_size)
    init_image = tf.Variable(init_image, dtype=tf.float32)

    opt = tf.optimizers.Adam(learning_rate=5.0)

    best_loss, best_img = float('inf'), None
    cfg = {
        'model': model,
        'loss_weights': (style_weight, content_weight),
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features
    }

    for i in range(iterations):
        grads, all_loss = compute_grads(cfg)
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, -103.939, 151.061)
        init_image.assign(clipped)

        if all_loss < best_loss:
            best_loss = all_loss
            best_img = init_image.numpy()

        if i % 100 == 0:
            print(f"Iteration {i}: Loss: {all_loss}")

    return deprocess_image(best_img)

# Main script
def main():
    content_path = 'content.jpg'  # Path to content image
    style_path = 'style.jpg'  # Path to style image

    result = run_style_transfer(content_path, style_path, iterations=1000)

    plt.figure(figsize=(10, 10))
    plt.imshow(result)
    plt.title("Styled Image")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
