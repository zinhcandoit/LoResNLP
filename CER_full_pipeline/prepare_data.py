import os

# --- CẤU HÌNH ---
# Sử dụng đường dẫn tuyệt đối (có chữ r đằng trước) để tránh lỗi không tìm thấy file
base_folder = r"C:\Users\VINH\OneDrive - VNU-HCMUS\Attachments\Desktop\P_OCR\OCR_Lab-20260106T152002Z-3-001\OCR_Lab"
image_folder_name = "images_test"
labels_folder_name = "labels"  # Thư mục con chứa file txt
output_file = "rec_gt.txt"

# Đường dẫn đầy đủ
img_dir_full = os.path.join(base_folder, image_folder_name)
lbl_dir_full = os.path.join(img_dir_full, labels_folder_name)
output_path_full = os.path.join(base_folder, output_file)

valid_images = [".jpg", ".jpeg", ".png", ".bmp"]

print(f"Đang quét ảnh tại: {img_dir_full}")
print(f"Đang lấy nhãn tại: {lbl_dir_full}")

if not os.path.exists(lbl_dir_full):
    print(f"❌ LỖI: Không tìm thấy thư mục labels: {lbl_dir_full}")
    exit()

label_files = [fn for fn in os.listdir(lbl_dir_full) if fn.lower().endswith(".txt")]
print("Số file labels.txt:", len(label_files))

image_files = [fn for fn in os.listdir(img_dir_full) if os.path.splitext(fn)[1].lower() in valid_images]
print("Số file ảnh:", len(image_files))


with open(output_path_full, "w", encoding="utf-8") as f_out:
    # Lấy danh sách tất cả file trong thư mục ảnh
    files = os.listdir(img_dir_full)
    
    count = 0
    for filename in files:
        name, ext = os.path.splitext(filename)
        
        # Nếu là file ảnh
        if ext.lower() in valid_images:
            # Tìm file txt tương ứng trong thư mục labels
            txt_path = os.path.join(lbl_dir_full, name + ".txt")
            
            if os.path.exists(txt_path):
                # Đọc nội dung file txt
                with open(txt_path, "r", encoding="utf-8") as f_in:
                    content = f_in.read()

                # Chuẩn hóa: gom về 1 dòng
                content = content.replace("\r\n", "\n").replace("\r", "\n")
                content = content.replace("\n", " ")       # bỏ xuống dòng
                content = content.replace("\t", " ")       # bỏ tab để không phá format
                content = " ".join(content.split())        # gộp nhiều space

                content = content.strip()
                if not content:
                    continue

                line = f"{image_folder_name}/{filename}\t{content}\n"

                f_out.write(line)
                count += 1
            else:
                # Báo nếu có ảnh mà không có file text (để kiểm tra)
                pass 

print(f"✅ Đã tạo xong file '{output_file}' tại thư mục dự án.")
print(f"👉 Tổng số mẫu tìm thấy: {count}")

