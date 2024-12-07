# Chemocaller: Inferring Nucleotide Chemical Structures from Nanopore Sequencing Ionic Currents

**Date**: From 2024-01 to present  
**Author**: Qinixia  
**Current Status**: Ongoing

## Project Overview

Chemocaller is a deep learning-based project focused on analyzing ionic current signals from nanopore sequencing to infer the chemical structures of nucleic acids. By leveraging advanced machine learning techniques, particularly deep learning models, this project aims to:

1. **Signal Interpretation**: Decode the chemical structure of nucleotides by analyzing the ionic current data generated during nanopore sequencing.
   
2. **Novel Modification Detection**: Develop a generative model capable of identifying novel RNA modifications. This model learns the underlying patterns of known chemical structures and can generalize to predict previously uncharacterized modifications.

3. **Integration of Chemical Structure Insights**: Integrate detailed chemical structure insights into the model to improve its performance in interpreting and classifying nanopore sequencing signals.

The ultimate goal is to expand our understanding of RNA modifications and improve the precision of nanopore sequencing data analysis.

## Key Tools and Versions

The project relies on the following tools and versions:

- **Bonito**: 0.8.1  
- **Remora**: 3.2.2  
- **Samtools**: 1.16.1 (module load: `apps/samtools/1.16.1-gnu485torch`)  
- **Torch**: 2.1.2  
- **Pysam**: 0.22.1  

For further details, please refer to the [tools_version.md](tools_version.md) file.

## Installation and Setup

This project is based on **Remora**. For installation and setup instructions, please follow the guidelines provided in the official [Remora GitHub repository](https://github.com/nanoporetech/remora).

Please note that:

- The dataset used in this project is currently not publicly available.
- Some code components are not open-source at this stage.


## How to Run

The project is built upon the **Remora** platform. To run the project, please follow these steps:

1. Clone the **Remora** repository:
   ```bash
   git clone https://github.com/nanoporetech/remora.git
   ```

2. Install the required dependencies listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. The key code for the project can be found in the `scripts/decoder` directory. Please read and understand this code carefully as it forms the core functionality of the project.

4. Follow the setup instructions for Remora in the [official Remora repository](https://github.com/nanoporetech/remora) to configure your environment and ensure all dependencies are properly set up.

For any questions or issues regarding the implementation, please feel free to reach out to the author directly:

**Contact**: [qinixia77@gmail.com](mailto:qinixia77@gmail.com)


## Contributing

This project is still under development. If you are interested in contributing or collaborating, please reach out to the author directly via email.

## Acknowledgements

- The development of this project is supported by ongoing research in nanopore sequencing and deep learning techniques.
- Special thanks to the developers of Remora and Bonito for their foundational tools.
