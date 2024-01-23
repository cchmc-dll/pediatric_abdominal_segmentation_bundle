# Model Overview
A Pediatric 3D Abdominal Organ Segmentation model, pretrained on large public datasets and fine tuned for institutional pediatric data. 

## Data
Organs Segmented:
- Liver
- Spleen
- Pancreas

Pre-training data:
- Total Segmentator (815)
- BTCV (30)
- TCIA Pediatric (282)

Fine-tuning data:
- Children's Liver Spleen CT dataset (275)
- Children's Pancreas CT dataset (146)

Testing data:
- Children's Liver-Spleen (57)
- Children's Pancreas (35)
- TCIA-Pediatric (74)
- Total Segmentator (50)

### Model Architectures
- DynUNet
- SegResNet
- SwinUNETR 

### Hyper-Parameter Tuning
Weights and Biases was used to extensively tune each model for learning rate, scheduler and optimizer. For fine-tuning the fraction of trainable layers was also optimized. DynUNet performed overall better on all test datasets. The Total Segmentator model was also compared and the DynUNet model significantly outperformed Total Segmentator on institutional test data while maintaining relatively stable performance on adult and TCIA datasets. 

### Input
One channel
- CT image

### Output
Four channels
- Label 3: pancreas
- Label 2: spleen
- Label 1: liver
- Label 0: everything else

## Performance
 - MedArxiv to be linked


## MONAI Bundle Commands
In addition to the Pythonic APIs, a few command line interfaces (CLI) are provided to interact with the bundle. The CLI supports flexible use cases, such as overriding configs at runtime and predefining arguments in a file.

For more details usage instructions, visit the [MONAI Bundle Configuration Page](https://docs.monai.io/en/latest/config_syntax.html).


#### Execute training:

```
python -m monai.bundle run --config_file configs/train.yaml
```

Please note that if the default dataset path is not modified with the actual path in the bundle config files, you can also override it by using `--dataset_dir`:

```
python -m monai.bundle run --config_file configs/train.yaml --dataset_dir <actual dataset path>
```

#### `train` config to execute multi-GPU training:

```
torchrun --nnodes=1 --nproc_per_node=8 -m monai.bundle run --config_file configs/train-multigpu.yaml
```

#### Override the `train` config to execute evaluation with the trained model:

```
python -m monai.bundle run --config_file "['configs/train.yaml','configs/evaluate.yaml']"
```

#### Execute inference:

```
python -m monai.bundle run --config_file configs/inference.yaml
```

#### Execute standalone `evaluate`:
```
python -m monai.bundle run --config_file configs/evaluate.yaml
```


#### Execute standalone `evaluate` in parallel:
```
torchrun --nnodes=1 --nproc_per_node=8 -m monai.bundle run --config_file configs/evaluate-standalone.yaml
```


#### Export checkpoint for TorchScript:

```
python -m monai.bundle ckpt_export network_def --filepath models/dynunet_FT.ts --ckpt_file models/dynunet_FT.pt --meta_file configs/metadata.json --config_file configs/inference.yaml
```

#### Export checkpoint to TensorRT based models with fp32 or fp16 precision:

```
python -m monai.bundle trt_export --net_id network_def --filepath models/A100/dynunet_FT_trt_16.ts --ckpt_file models/dynunet_FT.pt --meta_file configs/metadata.json --config_file configs/inference.yaml  --precision <fp32/fp16> --use_trace "True" --dynamic_batchsize "[1, 4, 8]" --converter_kwargs "{'truncate_long_and_double':True, 'torch_executed_ops': ['aten::upsample_trilinear3d']}"
```

#### Execute inference with the TensorRT model:

```
python -m monai.bundle run --config_file "['configs/inference.yaml', 'configs/inference_trt.yaml']"
```

# References

[1] To be added

# License
Copyright (c) MONAI Consortium

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
