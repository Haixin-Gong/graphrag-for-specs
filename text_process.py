import os
import PyPDF2

def process_pdf(pdf_file):
    """处理PDF文件"""
    # 使用PyPDF2读取PDF文件
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

if __name__ == "__main__":
    text_dir = "./technical_standards"
    for file in os.listdir(text_dir):
        if file.endswith(".pdf"):
            file_name = file.split(".")[0]
            save_path = f"./technical_standards/{file_name}.txt"
            pdf_file = os.path.join(text_dir, file)
            print(f"Processing {file}...")
            text = process_pdf(pdf_file)
            with open(save_path, "w") as f:
                f.write(text)
    ref_file = "./eval_demo/16060 接地-B16.pdf"
    print("Processing reference file...")
    text = process_pdf(ref_file)
    with open("./eval_demo/16060 接地-B16.txt", "w") as f:
        f.write(text)