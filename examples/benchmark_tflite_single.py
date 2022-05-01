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

import numpy as np
from pycoral.utils.edgetpu import make_interpreter, load_edgetpu_delegate

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '-i', '--input_model_path', required=True, help='Input directory of tflite models')
  parser.add_argument(
      '-o', '--output_dataset_log_dir', required=True, help='Output directory of benchmark logs')
  parser.add_argument(
      '-w', '--warm_count', type=int, default=3,
      help='Number of times to run warmup inference')
  parser.add_argument(
      '-c', '--count', type=int, default=10,
      help='Number of times to run inference')
  args = parser.parse_args()

  # load delegate
  delegate = load_edgetpu_delegate()
  # Init the np array
  inference_time_array = np.zeros(args.count)

  # Looping through all the tflite models
  model_path = args.input_model_path
  model_name = model_path.split('.')[0].split('/')[-1]
  log_path = os.path.join(args.output_dataset_log_dir, model_name + '.log')

  interpreter = make_interpreter(model_path_or_content=model_path, delegate=delegate)
  interpreter.allocate_tensors()
  input_tensor = interpreter.tensor(interpreter.get_input_details()[0]['index'])
  np.random.seed(12345)
  input_tensor()[0] = np.random.randint(-128, 128, size=input_tensor().shape[1:], dtype=np.int8)

  # Run inference
  # print('----INFERENCE TIME----')
  # Warmup Rounds
  for _ in range(args.warm_count):
    interpreter.invoke()

  for i in range(args.count):
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    inference_time_array[i] = inference_time * 1000
  # print('%.1fms' % (inference_time * 1000))

  # print('-------RESULTS--------')
  f = open(log_path, "w")
  f.write('inference_time (ms)\n')
  f.write('avg std count\n')
  f.write('%f %f %d\n' % (inference_time_array.mean(), inference_time_array.std(), args.count))
  f.close()


if __name__ == '__main__':
  main()
