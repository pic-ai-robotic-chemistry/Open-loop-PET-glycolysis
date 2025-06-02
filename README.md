# AI-assisted Catalysis

## Overview

This project leverages large language models (LLMs) in combination with Bayesian optimization to accelerate the discovery and screening of Lewis acid/pair catalysts for degrading PET. 

The three-stage process involves:

1. **LLM-embedded Bayesian Optimization**: Bayesian optimization over a discrete Lewis acid/base pair space using LLM embeddings and random embeddings to expand the search space.

2. **LLM-based hypotheses generation**: Using LLM to extract information and generate hypotheses for Lewis acid/pair catalysts.

3. **Chemical knowledge inspired heuristic out-of-loop search**: Using chemical knowledge and heuristics to guide the out-of-loop search for Lewis acid/pair catalysts.


## Directory Structure

```
AI-assisted-Catalysis/
├── figures/                  # Generated plots and visualizations
├── notebooks/                # Jupyter notebooks for exploration and demos
├── results/                  # Output results from experiments
├── src/                      # Source code
│   ├── data/                 # Data files (raw, processed)
│   ├── data.py               # Data loading and preprocessing utilities
│   ├── llm_embedding_bo.py   # Integrates LLM embeddings with Bayesian optimization
│   ├── llm_prompt_generation.py  # Prompt templates and generation routines
│   ├── openai_api_key        # File storing your OpenAI API key
│   └── utils.py              # Helper functions and utilities
├── requirements.txt          # Python dependencies
└── LICENSE                   # MIT License
```

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/AI-assisted-Catalysis.git
   cd AI-assisted-Catalysis
   ```

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate    # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Configuration

* **OpenAI API Key**: Place your OpenAI API key in the file `src/openai_api_key`, or set the environment variable:

  ```bash
  export OPENAI_API_KEY="your_api_key_here"
  ```

## Usage

### Running Notebooks

* Navigate to the `notebooks/` directory and launch JupyterLab or Jupyter Notebook:

  ```bash
  cd notebooks
  jupyter lab
  ```

* Open and run the example notebooks to see figure generations.

### Scripts

* **LLM embedding + Bayesian optimization**:

  ```bash
  cd src
  python llm_embedding_bo.py
  ```

* **Prompt generation utilities**:

  ```bash
  cd src
  python llm_prompt_generation.py
  ```

## File Descriptions

* **`src/data.py`**: Functions to load, clean, and preprocess catalyst datasets.
* **`src/llm_embedding_bo.py`**: Core integration of LLM embeddings with Bayesian optimization routines (e.g., using `scikit-optimize` or `GPyOpt`).
* **`src/llm_prompt_generation.py`**: Utilities to generate and manage prompt templates for LLM calls.
* **`src/utils.py`**: Miscellaneous helper functions (logging, argument parsing, feature scaling, etc.).
* **`src/data/`**: Directory containing raw and processed data files.
* **`src/openai_api_key`**: Plain-text file with your OpenAI API key (one line).

## Results & Figures

* Experimental results are written to `results/` (e.g., JSON or CSV of evaluated candidates).
* Plots and visualizations (convergence curves, embedding spaces) are saved in `figures/`.

## Requirements

See `requirements.txt` for a full list of dependencies, which may include:

* `openai`
* `scikit-learn`
* `scikit-optimize`
* `numpy`, `pandas`
* `matplotlib`, `seaborn`
* `jupyterlab`

## Contributing

Contributions, issues, and feature requests are welcome! Please open an issue or submit a pull request.

## License

Distributed under the MIT License. See `LICENSE` for more information.
