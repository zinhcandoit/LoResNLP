"""
Script để chuẩn bị dữ liệu từ Label.txt (raw OCR) và Label.txt (final correct) 
để train ByT5 cho task OCR correction
"""

import json
import os
import pandas as pd
from pathlib import Path

def parse_label_file(label_path):
    """
    Parse PPOCRLabel format file
    Returns: dict với key là tên file ảnh, value là list các transcription
    """
    data = {}
    
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Split filename và annotations
            parts = line.split('\t', 1)
            if len(parts) != 2:
                continue
            
            filename = parts[0].strip()
            annotations = json.loads(parts[1])
            
            # Extract all transcriptions
            transcriptions = []
            for ann in annotations:
                text = ann.get('transcription', '')
                if text and text not in ['###', '']:  # Ignore empty or marked as difficult
                    transcriptions.append(text)
            
            data[filename] = transcriptions
    
    return data

def create_training_data(raw_label_path, final_label_path, output_dir):
    """
    Tạo training data từ raw và final labels
    """
    print("Đọc file raw label...")
    raw_data = parse_label_file(raw_label_path)
    
    print("Đọc file final label...")
    final_data = parse_label_file(final_label_path)
    
    # Tạo dataset
    train_data = []
    
    for filename in raw_data:
        if filename not in final_data:
            print(f"Warning: {filename} không có trong final label, bỏ qua")
            continue
        
        raw_texts = raw_data[filename]
        final_texts = final_data[filename]
        
        # Pair up raw và final texts
        # Giả sử chúng đã được align (cùng thứ tự và số lượng)
        min_len = min(len(raw_texts), len(final_texts))
        
        for i in range(min_len):
            raw_text = raw_texts[i]
            final_text = final_texts[i]
            
            # Chỉ thêm vào nếu có sự khác biệt (tức là có lỗi cần sửa)
            # Ignore những cái có CER = 0 (raw == final, OCR đã đúng)
            if raw_text != final_text and raw_text.strip() and final_text.strip():
                train_data.append({
                    'source': raw_text,  # Input: OCR sai
                    'target': final_text,  # Target: OCR đúng
                    'filename': filename
                })
    
    print(f"\nTổng số mẫu training: {len(train_data)}")
    
    # Tạo thư mục output
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Lưu thành CSV
    df = pd.DataFrame(train_data)
    csv_path = output_dir / 'train_data.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"Đã lưu CSV: {csv_path}")
    
    # Lưu thành text files (format cho T5)
    source_path = output_dir / 'train.source'
    target_path = output_dir / 'train.target'
    
    with open(source_path, 'w', encoding='utf-8') as f_src, \
         open(target_path, 'w', encoding='utf-8') as f_tgt:
        for item in train_data:
            f_src.write(item['source'] + '\n')
            f_tgt.write(item['target'] + '\n')
    
    print(f"Đã lưu source: {source_path}")
    print(f"Đã lưu target: {target_path}")
    
    # Hiển thị một số ví dụ
    print("\n=== VÍ DỤ MẪU TRAINING ===")
    for i, item in enumerate(train_data[:5]):
        print(f"\nMẫu {i+1}:")
        print(f"  Input  (raw):   {item['source']}")
        print(f"  Target (final): {item['target']}")
    
    return train_data

def main():
    # Cấu hình đường dẫn
    base_dir = Path("LoResNLP/data/midterm")
    
    # Đường dẫn file raw và final cho T1
    raw_label = base_dir / "raw" / "T1" / "Label.txt"
    final_label = base_dir / "final" / "T1" / "Label.txt"
    
    # Thư mục output
    output_dir = base_dir / "training_data" / "T1"
    
    # Tạo training data
    print("="*60)
    print("Xử lý T1...")
    print("="*60)
    train_data = create_training_data(raw_label, final_label, output_dir)
    
    # Tương tự cho T6 nếu có
    print("\n" + "="*60)
    print("Xử lý T6...")
    print("="*60)
    
    raw_label_t6 = base_dir / "raw" / "T6" / "Label.txt"
    final_label_t6 = base_dir / "final" / "T6" / "Label.txt"
    output_dir_t6 = base_dir / "training_data" / "T6"
    
    if raw_label_t6.exists() and final_label_t6.exists():
        train_data_t6 = create_training_data(raw_label_t6, final_label_t6, output_dir_t6)
    else:
        print("T6 data không tồn tại")
    
    print("\n✓ Hoàn thành!")

if __name__ == "__main__":
    main()
