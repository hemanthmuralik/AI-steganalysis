import argparse, numpy as np, tensorflow as tf
from tensorflow.keras.models import load_model

def representative_dataset_gen(sample_paths, preprocess_fn):
    def gen():
        for p in sample_paths:
            img = preprocess_fn(p)
            img = np.expand_dims(img, axis=0).astype(np.float32)
            yield [img]
    return gen

def simple_preprocess(p, size=(256,256)):
    import cv2
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size).astype('float32') / 255.0
    img = np.expand_dims(img, -1)
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--out', default='stego_model_int8.tflite')
    parser.add_argument('--samples', nargs='*', default=[])
    args = parser.parse_args()

    model = load_model(args.model_path, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if len(args.samples) > 0:
        converter.representative_dataset = representative_dataset_gen(args.samples, simple_preprocess)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    open(args.out, 'wb').write(tflite_model)
    print('Wrote', args.out)

if __name__ == '__main__':
    main()
