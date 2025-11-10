# LACE-Bench

## Description
LACE-Bench is a benchmarking tool designed to evaluate the performance of various algorithms and models in a standardized manner. It provides a framework for running experiments and collecting metrics to facilitate comparisons and improvements.

## Installation
To install the necessary dependencies for LACE-Bench, please follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/LACE-Bench.git
   ```
2. Navigate to the project directory:
   ```
   cd LACE-Bench
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Dataset
### Visual Genome
1) Download VisualGenome zip data
```bash
mkdir -p "$HOME/data/visual_genome"
cd "$HOME/data/visual_genome"
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
```

2) 수동 명령 (zip 파일을 이미 가지고 있는 경우)
```bash
# 대상 경로
BASE="$HOME/data/visual_genome"
TARGET="$BASE/VG_100k_all"

mkdir -p "$TARGET"

# 압축 해제
unzip -q "$BASE/images.zip"  -d "$TARGET"/
unzip -q "$BASE/images2.zip" -d "$TARGET"/
```

4) 병합 확인
```bash
ls "$HOME/data/visual_genome/VG_100k_all" | head
```

### LACE data
```bash
mkdir -p "$HOME/data/etri/core"
cd "$HOME/data/etri/core"
```


## Usage
To execute the code in this project, follow these steps:

1. Ensure that you have Python installed on your system.
2. Open a terminal and navigate to the project directory.
3. Run the main script:
   ```
   MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct" python run_vllm.py
   ```
4. You can customize the execution by modifying the configuration files or passing command-line arguments as needed.
