# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Example using PyCoral to classify a given image using an Edge TPU.

To run this code, you must attach an Edge TPU attached to the host and
install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
device setup instructions, see coral.ai/docs/setup.

Example usage:
```
bash examples/install_requirements.sh classify_image.py

python3 examples/classify_image.py \
  --model test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite  \
  --labels test_data/inat_bird_labels.txt \
  --input test_data/parrot.jpg
```
"""

import os
import argparse
import time
import datetime

import numpy as np
from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter, load_edgetpu_delegate

DATASETS_WO_INT8 = ['vit_cifar100']

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '-d', '--dataset', required=True, help='Name of the dataset.')
  parser.add_argument(
      '-i', '--input_model_dir', required=True, help='Input directory of tflite models')
  parser.add_argument(
      '-o', '--output_log_dir', required=True, help='Output directory of benchmark logs')
  parser.add_argument(
      '-w', '--warm_count', type=int, default=3,
      help='Number of times to run warmup inference')
  parser.add_argument(
      '-c', '--count', type=int, default=10,
      help='Number of times to run inference')
  parser.add_argument(
      '-p', '--print_interval', type=int, default=1000,
      help='The count of model between two printouts')
  parser.add_argument(
      '-b', '--break_flag', type=int, default=0,
      help='The flag of break')
  args = parser.parse_args()

  input_dataset_dir = os.path.join(args.input_model_dir, args.dataset)
  tflite_names = next(os.walk(input_dataset_dir))[1]
  #tflite_names = ["efficientnet_cifar10_seed1000_tf_use_stats"]
  print("Found", len(tflite_names), "models.")

  output_dataset_log_dir = os.path.join(args.output_log_dir, args.dataset)
  if not os.path.exists(output_dataset_log_dir):
    os.mkdir(output_dataset_log_dir)

  model_postfix = "_INT8.tflite"
  if args.dataset in DATASETS_WO_INT8:
    model_postfix = ".tflite"

  # load delegate
  delegate = load_edgetpu_delegate()

  # Init the np array
  inference_time_array = np.zeros(args.count)

  # Looping through all the tflite models
  for c, tflite_name in enumerate(tflite_names):
    tflite_dir = os.path.join(input_dataset_dir, tflite_name)
    model_path = os.path.join(tflite_dir, "tflites", tflite_name + model_postfix)
    log_path = os.path.join(output_dataset_log_dir, tflite_name + '.log')

    interpreter = make_interpreter(model_path_or_content=model_path, delegate=delegate)
    interpreter.allocate_tensors()

    # Model must be uint8 quantized
    if common.input_details(interpreter, 'dtype') != np.uint8:
      raise ValueError('Only support uint8 input type.')

    input_tensor = interpreter.tensor(interpreter.get_input_details()[0]['index'])
    np.random.seed(12345)
    input_tensor()[0] = np.random.randint(0, 256, size=input_tensor().shape[1:], dtype=np.uint8)

    # Run inference
    # print('----INFERENCE TIME----')
    # Warmup Rounds
    for _ in range(args.warm_count):
      interpreter.invoke()

    for i in range(args.count):
      start = time.perf_counter()
      interpreter.invoke()
      inference_time = time.perf_counter() - start
      inference_time_array[i] = inference_time
      print('%.1fms' % (inference_time * 1000))

    # print('-------RESULTS--------')
    f = open(log_path, "w")
    f.write("avg std count")
    f.write('%f %f %d' % (inference_time_array.mean(), inference_time_array.std(), args.count))
    f.close()

    if (c+1) % args.print_interval == 0:
      print(c+1, "models measured @", datetime.datetime.now())
    if args.break_flag != 0:
      break


if __name__ == '__main__':
  main()
