# tacotron2-pytorch
 
Pytorch implementation of [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884)    

  
<img src="https://user-images.githubusercontent.com/42150335/94840259-1cfbe900-0453-11eb-8803-cac2ea30b425.png" width=500>
  
This implementation focuses as much as possible on the readability and extensibility of the code and the reproduction as it is in the paper (without Wavenet). I would appreciate it if you could feedback or contribution at any time if there was a mistake or an error.
  
## Usage
```python
import torch
import numpy as np
from tacotron2 import Tacotron2
from test.args import DefaultArgument

batch_size, seq_length = 3, 3

inputs = torch.LongTensor(np.arange(batch_size * seq_length).reshape(batch_size, seq_length))
input_lengths = torch.LongTensor([3, 3, 2])
targets = torch.FloatTensor(batch_size, 100, 80).uniform_(-0.1, 0.1)
args = DefaultArgument()

model = Tacotron2(args)
output = model(inputs, targets, input_lengths)
```
  
## Installation
Currently we only support installation from source code using setuptools. Checkout the source code and run the
following commands:  
```
pip install -e .
```
  
## Troubleshoots and Contributing
If you have any questions, bug reports, and feature requests, please [open an issue](https://github.com/sooftware/tacotron2-pytorch/issues) on github or   
contacts sh951011@gmail.com please.
  
I appreciate any kind of feedback or contribution.  Feel free to proceed with small issues like bug fixes, documentation improvement.  For major contributions and new features, please discuss with the collaborators in corresponding issues.  
  
### Code Style
I follow [PEP-8](https://www.python.org/dev/peps/pep-0008/) for code style. Especially the style of docstrings is important to generate documentation.  
  
## Reference
  
- [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884)
- [NVIDIA/tacotron2](https://github.com/NVIDIA/tacotron2)    
  
## Author
  
* Soohwan Kim [@sooftware](https://github.com/sooftware)
* Contacts: sh951011@gmail.com
