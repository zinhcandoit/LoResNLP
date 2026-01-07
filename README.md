# LoResNLP
It's a full pipeline for OCR accuracy enhancement using scanned OCR dataset
## Training
### 1. Synthetic data generation
#### Prerequisites
```bash
# Create conda environment
conda create -n prep python=3.9 -y
conda activate prep

# Install dependencies
pip install numpy pillow opencv-python matplotlib tqdm
```

#### Generate Synthetic Training Data

```bash
# Preprocess background images (first time only)
python prepare_backgrounds.py

# Generate 5 base images + 20 noisy variants
python generate_ocr_data.py --base 5

# Generate larger datasets
python generate_ocr_data.py --base 50  # 50 base + 200 variants
```
### 2. Training Diffusion Model
Follow the instruction of [ResShift's pipeline](https://github.com/zsyOAOA/ResShift.git)
### 3. Training for Post-OCR Correction
#### Prepare data
```bash
cd CER_full_pipeline
python prepare_data_for_byt5.py
```
#### Training ByT5 for Post-OCR Correction
 Follow the [Hugging Face ByT5 guide](https://huggingface.co/docs/transformers/main/en/model_doc/byt5)

## Trained models
### **ByT5 Post-OCR Correction**:
```bash 
cd CER_full_pipeline
```
### **Image Restoration**:
#### Prepare:
1. Download the pre-trained VQGAN model from this [link](https://github.com/zsyOAOA/ResShift/releases) and put it in the folder of 'weights'
2. Adjust the model's path in the [config](configs) file. 
#### Image deblurring

```
python inference_resshift.py -i [image folder/image path] -o [result folder] --task deblur --scale 1 
```
