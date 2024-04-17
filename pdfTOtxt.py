import os
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        pdf_reader = PdfReader(f)
        num_pages = len(pdf_reader.pages)
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def save_text_to_file(text, output_file):
    with open(output_file, "w") as f:
        f.write(text)

if __name__ == "__main__":
    pdf_folder = "programas"  # Change this to the folder containing your PDF files
    output_folder = "programas_texts"  # Change this to the folder where you want to save the text files

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all PDF files in the folder
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            output_file = os.path.join(output_folder, filename.replace(".pdf", ".txt"))

            # Extract text from PDF and save to text file
            extracted_text = extract_text_from_pdf(pdf_path)
            save_text_to_file(extracted_text, output_file)
