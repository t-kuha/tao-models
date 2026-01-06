import argparse
import os

from onnx2torch import convert
import torch


def main():
    parser = argparse.ArgumentParser(description='Convert ONNX model to PyTorch .pth (state_dict)')
    parser.add_argument('input_onnx', help='Path to input .onnx file')
    parser.add_argument('-o', '--output', help='Path to output .pth file (optional)')
    args = parser.parse_args()

    input_path = args.input_onnx
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f'input file does not exist: {input_path}')

    output_path = args.output if args.output else os.path.splitext(os.path.basename(input_path))[0] + '.pth'

    backbone = convert(input_path)
    backbone.eval()

    # Optional quick sanity check with a dummy input
    dummy_input = torch.ones(1, 3, 224, 224)
    with torch.no_grad():
        output = backbone(dummy_input)
        # print(output)
        print(torch.argmax(output, dim=1))

    torch.save(backbone.state_dict(), output_path)

    print(f'[INFO] Saved PyTorch state_dict to: {output_path}')


if __name__ == '__main__':
    main()
