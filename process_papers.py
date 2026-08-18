import os
import glob
from pypdf import PdfReader
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# Initialize Azure OpenAI Client
client = AzureOpenAI(
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

# Azure requires the specific deployment name of your model
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Directory and file setup
papers_dir = "papers"
processed_tracker = os.path.join(papers_dir, "processed_papers.txt")
main_tex_path = "main.tex"

def get_processed_papers():
    """Returns a set of filenames that have already been processed."""
    if not os.path.exists(processed_tracker):
        return set()
    with open(processed_tracker, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f)

def mark_as_processed(filename):
    """Appends a filename to the tracker."""
    with open(processed_tracker, "a", encoding="utf-8") as f:
        f.write(f"{filename}\n")

def extract_pdf_text(filepath):
    """Extracts text from the first 10 pages of a PDF to respect context limits."""
    reader = PdfReader(filepath)
    text = ""
    for i, page in enumerate(reader.pages):
        if i >= 10: 
            break
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def generate_latex_summary(text):
    """Sends the text to Azure OpenAI and asks for the filled LaTeX macro."""
    system_prompt = (
        "You are a research assistant. Extract key details from the provided paper and output ONLY the following LaTeX command filled with the data. "
        "Keep summaries brief and focus on technical details.\n\n"
        "\\papersummary\n"
        "{Title}\n"
        "{Authors}\n"
        "{Year, Venue}\n"
        "{Core Problem addressed}\n"
        "{Architecture / Method used}\n"
        "{Hardware, Sensors & Datasets used}\n"
        "{Key Results}\n"
        "{Results Analysis & Tables (Use LaTeX tabular if needed)}\n"
        "{Relevance / Limitations}"
    )
    
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Output the filled LaTeX block for this text:\n\n{text[:30000]}"}
        ]
    )
    
    return response.choices[0].message.content.strip()

def append_to_latex(latex_content):
    """Inserts the new block right before the \\end{document} tag."""
    if not os.path.exists(main_tex_path):
        print(f"Error: Could not find {main_tex_path}")
        return False
        
    with open(main_tex_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Inject right before the document ends so it compiles seamlessly
    if "\\end{document}" in content:
        # FIX: Switched to standard string concatenation to prevent f-string variable errors
        content = content.replace("\\end{document}", latex_content + "\n\n\\end{document}")
        with open(main_tex_path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    return False

def main():
    if not os.path.exists(papers_dir):
        os.makedirs(papers_dir)
        print(f"Created '{papers_dir}' directory. Please drop your PDFs there and run again.")
        return

    processed = get_processed_papers()
    pdfs = glob.glob(os.path.join(papers_dir, "*.pdf"))
    
    if not pdfs:
        print("No PDFs found in the 'papers' directory.")
        return

    for pdf in pdfs:
        filename = os.path.basename(pdf)
        if filename in processed:
            print(f"Skipping already processed: {filename}")
            continue
            
        print(f"Processing {filename}...")
        try:
            text = extract_pdf_text(pdf)
            latex_block = generate_latex_summary(text)
            
            # Clean up markdown code blocks if the AI includes them
            if latex_block.startswith("```latex"):
                latex_block = latex_block[8:]
            if latex_block.startswith("```"):
                latex_block = latex_block[3:]
            if latex_block.endswith("```"):
                latex_block = latex_block[:-3]
            latex_block = latex_block.strip()
            
            success = append_to_latex(latex_block)
            if success:
                mark_as_processed(filename)
                print(f"Successfully added {filename} to {main_tex_path}")
            else:
                print(f"Failed: \\end{{document}} not found in {main_tex_path}")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    main()