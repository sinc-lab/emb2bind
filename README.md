# emb2bind: protein binding prediction tool
emb2bind is a protein binding prediction tool that takes as input a FASTA file and a directory containing precomputed residue-level embeddings, and returns binding predictions per amino acid in CAID-style format.

## Model overview
The model was trained to predict protein binding residues using binding annotations from [DisProt](https://disprot.org/), release 25_12. Residues annotated as binding were used as positive examples, while all other residues were treated as negative examples.

The input representation is composed of two components:

* **ESM-2 residue-level embeddings**, extracted with the `esm2_t6_8M_UR50D` model [Lin et al., 2023](https://doi.org/10.1126/science.ade2574).
* **AIUPred energy-based features**, which provide residue-level energy information [Erdős and Dosztányi, 2024](https://doi.org/10.1093/nar/gkae385).

These features are concatenated and used as input to the binding prediction model.

## Environment setup

1. **Clone the repository:**
```bash
git clone https://github.com/sinc-lab/emb2bind.git
cd emb2bind
```

2. **Create and activate a virtual environment**:
```bash
conda create -n emb2bind python=3.11
conda activate emb2bind
```

3. **Install required packages:**
```bash
pip install -r requirements.txt
```

## Usage
The main prediction script is `predict.py`. 

To run the predictor, you need:

1. A FASTA file containing the protein sequences.
2. A directory containing the precomputed embeddings for the same sequences.

Each embedding file must correspond to one FASTA record and should be named `{protein_id}.npy`, where `{protein_id}` is the identifier of the protein in the FASTA file. The embeddings should have shape `(320, L)`, where `L` is the length of the protein sequence.

### Generate embeddings
If needed, compute the embeddings for the input FASTA file using `compute_embeddings.py`:

```bash
python compute_embeddings.py \
  --fasta <path_to_fasta> \
  --output-dir <output_directory> \
  [--device <device>] \
  [--skip-existing]
```
For example, using the provided sample FASTA file:

```bash
python compute_embeddings.py \
  --fasta data/samples.fasta \
  --output-dir data/embeddings/ \
  --device cuda
```

This writes one `{protein_id}.npy` file per FASTA record (shape `(320, L)`).

## Run prediction

After computing the embeddings, run:

```bash
python predict.py \
  --fasta data/samples.fasta \
  --embedding-dir data/embeddings/
```

This script will:

* Read all sequences from the input FASTA file.
* Load the corresponding precomputed embeddings.
* Predict residue-level binding scores using a sliding-window approach.
* Save the output files in the selected output directory (./results/ by default).


## Command-line arguments

| Argument          | Short | Description                                                                            |
| ----------------- | ----: | -------------------------------------------------------------------------------------- |
| `--fasta`         |  `-f` | Path to the input FASTA file. Required.                                                |
| `--embedding-dir` |  `-e` | Directory containing one precomputed `.npy` embedding file per FASTA record. Required. |
| `--output-dir`    |  `-o` | Directory where predictions and additional outputs are saved. Default: `./results/`.   |
| `--device`        |  `-d` | Device used for prediction: `cpu`, `cuda`, `cuda:0`, etc. Default: `cuda`.             |
| `--threads`       |       | Number of CPU threads to use.                                                          |
| `--verbose`       |  `-v` | Enable detailed progress messages. Default: disabled.                                  |



## Container usage for CAID challenge

For the CAID challenge container, embeddings must be precomputed outside the container and mounted at runtime. The container includes the trained classifier and the minimal dependencies needed for CPU inference. The container is designed to run without internet access during prediction.

### 1. Precompute embeddings

First, generate the embeddings as described above. This step must be done before running the container.

### 2. Pull the Docker image

The image is available on Docker Hub:

```bash
docker pull sofiaaduarte/emb2bind:caid-test
```

### 3. Run the container offline

```bash
docker run --rm --network none \
  -v /absolute/path/to/samples.fasta:/data/input.fasta:ro \
  -v /absolute/path/to/embeddings:/data/embeddings:ro \
  -v /absolute/path/to/output:/output \
  sofiaaduarte/emb2bind:caid-test \
  --threads 4
```

The paths on the left side of each `:` correspond to paths on the host machine and can be changed by the user. The paths on the right side are fixed inside the container.

The required mounts are:

| Host path                         | Container path      | Description                                                                              |
| --------------------------------- | ------------------- | ---------------------------------------------------------------------------------------- |
| `/absolute/path/to/samples.fasta` | `/data/input.fasta` | Input FASTA file. Mounted as read-only.                                                  |
| `/absolute/path/to/embeddings`    | `/data/embeddings`  | Directory containing one `{protein_id}.npy` file per FASTA record. Mounted as read-only. |
| `/absolute/path/to/output`        | `/output`           | Directory where predictions are written.                                                 |

Relative paths can also be used, but they should be explicitly written with `./` when appropriate. For example, using the current working directory and the provided sample FASTA and embeddings:

```bash
docker run --rm --network none \
  -v ./data/samples.fasta:/data/input.fasta:ro \
  -v ./data/embeddings:/data/embeddings:ro \
  -v ./results:/output \
  sofiaaduarte/emb2bind:caid-test \
  --threads 4
```
The container will write one `{protein_id}.caid` file per protein in the output directory, along with a `timings.csv` file containing per-sequence execution times in milliseconds.


### 3. (Optional) Build and publish Docker Hub

The Docker image is already available on Docker Hub. To build it locally from this repository:

```bash
docker build --network=host -t emb2bind:caid .
```

In order to publish to Docker Hub, log in:
```bash
docker login
```

Tag the local image:
```bash
docker tag emb2bind:caid <dockerhub-user>/emb2bind:caid
```
And then push it:

```bash
docker push <dockerhub-user>/emb2bind:caid
```
