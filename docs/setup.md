# Setup & Installation

[cite_start]The code was developed entirely in **Python 3**[cite: 186554, 186638].

## Requirements

[cite_start]The primary dependencies are[cite: 186550]:
- **Cantera** (`cantera==2.6.0`)
- **Tensorflow**
- **Matplotlib**
- **NumPy** (`numpy==1.26.4`)
- **Scikit-learn**
- **NetworkX**
- **SL-GPS Library** (`git+https://github.com/ctftamu/SL-GPS.git`)

## Installation Steps

1.  **Install base libraries:**
    ```bash
    pip install matplotlib tensorflow
    ```

2.  **Install specific Python dependencies:**
    ```bash
    pip install "numpy==1.26.4" networkx scikit-learn
    ```

3.  **Install Cantera:** (Ensure the correct version)
    ```bash
    pip install --no-cache-dir "cantera==2.6.0"
    ```

4.  **Install the SL-GPS core library:**
    ```bash
    pip install "git+[https://github.com/ctftamu/SL-GPS.git](https://github.com/ctftamu/SL-GPS.git)"
    ```
    [cite_start]_Alternatively, if the package is published, a simple `pip install slgps` would suffice._ [cite: 186638]

5.  [cite_start]**Verify installation:** Test your installation by running any of the files in the `tests/` folder[cite: 186554, 186638].