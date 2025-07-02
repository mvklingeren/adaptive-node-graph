# Guide to Running Your CUDA Kernel in Google Colab

This guide provides step-by-step instructions to compile and run your custom CUDA kernel in a Google Colab notebook.

### Step 1: Package Your Project

First, you need to package the necessary files into a single archive. The required files are:
- `main.cpp` (The C++ host application)
- `generated-shakespear-llm-kernel.cu` (Your generated CUDA kernel)
- `Makefile` (The build instructions)
- `src/cuda-work/shakespeare.txt` (The dataset)

Open a terminal in the project's root directory (`/Users/mvklingeren/source/adaptive-node`) and run the following command to create `cuda_project.tar.gz`:

```bash
tar -czvf cuda_project.tar.gz main.cpp Makefile generated-shakespear-llm-kernel.cu src/cuda-work/shakespeare.txt
```

### Step 2: Setup Google Colab

1.  **Open Colab and Set GPU Runtime:**
    *   Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook.
    *   In the menu, navigate to `Runtime` -> `Change runtime type`.
    *   Select `T4 GPU` from the "Hardware accelerator" dropdown and click `Save`.

2.  **Upload the Project Archive:**
    *   Create a new code cell in your notebook.
    *   Paste and run the following Python code. This will display an "Upload" button.
    *   Click the button and select the `cuda_project.tar.gz` file you created in the previous step.

    ```python
    from google.colab import files
    uploaded = files.upload()
    ```

3.  **Unpack the Archive:**
    *   In a new cell, run this shell command to extract your project files. The `!` tells Colab to execute it as a shell command.

    ```bash
    !tar -xzvf cuda_project.tar.gz
    ```

### Step 3: Compile and Run

1.  **Compile the Code:**
    *   Use the `make` command in a new cell to compile your C++ and CUDA source files using the provided `Makefile`.

    ```bash
    !make
    ```
    *   You should see output from the `nvcc` compiler, finishing with a message like "Linking complete. Executable created: run_model".

2.  **Execute the Kernel:**
    *   Finally, run your compiled program.

    ```bash
    !./run_model
    ```
    *   The program will now execute on the Colab GPU. You will see the log messages from `main.cpp`, including the sample output logits, confirming that your kernel ran successfully.
