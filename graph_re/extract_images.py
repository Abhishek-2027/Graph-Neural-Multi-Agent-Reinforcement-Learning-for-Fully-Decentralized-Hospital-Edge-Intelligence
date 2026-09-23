import fitz
import os

pdf_path = "Task prioritization and distributed deep reinforcement learning for --healthcare management in Cloud-Edge environments.pdf"
output_dir = "extracted_graphs"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

doc = fitz.open(pdf_path)

img_idx = 1
for page_index in range(len(doc)):
    page = doc.load_page(page_index)
    image_list = page.get_images(full=True)
    
    if image_list:
        print(f"Found {len(image_list)} images on page {page_index}")
    for image_index, img in enumerate(image_list, start=1):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        image_name = f"image_{img_idx}.{image_ext}"
        image_path = os.path.join(output_dir, image_name)
        
        with open(image_path, "wb") as f:
            f.write(image_bytes)
        print(f"Saved {image_path}")
        img_idx += 1
