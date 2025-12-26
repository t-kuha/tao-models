import argparse
import os

import tensorflow as tf
import tf2onnx


def convert_h5_to_onnx(h5_path, onnx_path, opset_version=13):
    """
    Converts a TensorFlow Keras .h5 file to ONNX format.
    """

    # Check if input file exists
    if not os.path.exists(h5_path):
        print(f"Error: Input file '{h5_path}' not found.")
        return

    try:
        print(f"Loading Keras model from {h5_path}...")
        # Load the Keras model
        model = tf.keras.models.load_model(h5_path, compile=False)

        # Display model summary (optional, helps verify correct load)
        model.summary()

        print("Converting to ONNX...")
        # Convert the model to ONNX
        # input_signature is optional; tf2onnx usually infers it from the model
        _ = tf2onnx.convert.from_keras(
            model,
            opset=opset_version,
            output_path=onnx_path
        )

        print("Successfully converted model.")
        print(f"Saved to: {onnx_path}")

    except Exception as e:
        print(f"An error occurred during conversion: {e}")

if __name__ == "__main__":
    # Initialize Argument Parser
    parser = argparse.ArgumentParser(description="Convert TensorFlow .h5 model to ONNX.")

    # Add arguments
    parser.add_argument("input_file", help="Path to the input .h5 or .hdf5 file")
    parser.add_argument("output_file", help="Path to the output .onnx file")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version (default: 13)")

    args = parser.parse_args()

    # Run conversion
    convert_h5_to_onnx(args.input_file, args.output_file, args.opset)
