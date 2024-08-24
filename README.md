# PyDecomposer

PyDecomposer is a Python package for advanced signal decomposition using Variational Mode Decomposition (VMD) and Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (CEEMDAN).

## Features

- Decompose signals using VMD and CEEMDAN
- Automatically adjust the number of Intrinsic Mode Functions (IMFs)
- Classify IMFs into high-frequency and low-frequency components
- Visualize decomposed signals and IMFs

## Installation

You can install PyComposer using pip:

```bash
pip install pycomposer
```

## Usage

Here's a quick example of how to use PyComposer:

```python
import numpy as np
from pydecomposer import DecompositionModel

# Generate a sample signal
t = np.linspace(0, 1, 1000)
signal = np.sin(2*np.pi*10*t) + 0.5*np.sin(2*np.pi*50*t)

# Create a DecompositionModel instance
model = DecompositionModel()

# Execute the decomposition
model.run(signal)

# Get the decomposed signals
high_freq, medium_freq, low_freq, residual = model.get_signals()

```

## Documentation

For more detailed information about the API and its usage, please refer to the [full documentation](link_to_your_documentation).

## Dependencies

- numpy
- matplotlib
- vmdpy
- EntropyHub
- PyEMD

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact

For any issues or questions, please contact <*<yc2349@ac.ic.uk>*.
