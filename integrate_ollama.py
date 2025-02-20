from dotenv import load_dotenv
import numpy as np
import pickle
import openai
from sentence_transformers import SentenceTransformer
from store_embedding import store_emb
from google.cloud import translate_v2 as translate
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# client = ollama.Client()
model = SentenceTransformer("all-mpnet-base-v2")
load_dotenv()

openai.api_key=os.getenv("OPENAI")

job_data=[]

def get_jobs(input_query):
    index=store_emb()
    with open("metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    query = model.encode(input_query).astype('float32')
    D, I = index.search(np.array([query]), k=5)
    return I,metadata

def resume_job_list(input_query):
    index=store_emb()
    with open("metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    query = model.encode(input_query).astype('float32')
    D, I = index.search(np.array([query]), k=5)
    list=[]

      
    for idx in I[0]:
        context = f"""
        You act as a resume-job finder. 
        Your task is to provide response about the job query by using the given data: {metadata[idx]}. Summarize and give the response in json format with id,title,occupation,employer_description as fixed attributes
        """
        openai_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": context}]
        )

        list.append(openai_response["choices"][0]["message"]["content"]) 
    return list

def generate_cv(resume_data,jobs):
    context = f"""
        You act as a Cover Letter generator. 
        Your task is to create cover letter with resume-data - {resume_data}, job-data - {jobs}. use the given data in the cover letter and generate conver letter
        """
    openai_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": context}]
        )
    return openai_response["choices"][0]["message"]["content"]


def is_query_related_to_job(query, index, metadata, threshold=0.400, k=10):
    query_embedding = model.encode(query).astype('float32')
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    distances, indices = index.search(np.array([query_embedding]), k)

    relevant_docs = [
        {"distance": dist, "document": metadata[idx]} 
        for dist, idx in zip(distances[0], indices[0]) 
        if dist > threshold and idx < len(metadata)
    ]
    
    is_relevant = len(relevant_docs) > 0
    return is_relevant, relevant_docs

def integrate(query):
    index = store_emb()

    with open("metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    is_related, docs=is_query_related_to_job(query,index,metadata)
    print(is_related)
    if not is_related:
        print('not related')
        context=f"for the given {query}, given proper response"
        openai_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": context}]
        )
        return openai_response["choices"][0]["message"]["content"]
    response = []
    d={}
    for doc in docs:
        d.update(doc['document'])

    context = f"""
    You act as a job-searching AI agent. 
    Your task is to provide response about the job query by using user query: {query} and the given data: {d}.
    if asked about any job details add company name as well to the response
    Be more specific while giving response
    """
    
    openai_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": context}]
    )
    response.append(openai_response["choices"][0]["message"]["content"])
  
    return response

def parse_response(response):
    
    l=[]
    client = translate.Client(api_key=os.getenv('GOOGLE_TRANSLATE'))
    for res in response:
    
        result = client.translate(res, target_language="en")
        l.append(result['translatedText'])
    return l

def create_cover_letter_pdf(filename,content):
    directory = "src/static/files"
    
     # Ensure the directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Create the full file path
    file_path = os.path.join(directory, filename)
    
    # Create a canvas object with the file path
    c = canvas.Canvas(file_path, pagesize=letter)
    width, height = letter

    # Set up title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 40, "Cover Letter")

    # Set font for the body
    c.setFont("Helvetica", 10)
    
    # Starting position for the text
    x = 100
    y = height - 100
    line_height = 12  # Adjust this value to set the line spacing

    # Max width for the text
    max_width = width - 2 * x  # Subtract margins from the page width
    
    # Function to wrap text based on the max width
    def wrap_text(text, max_width):
        lines = []
        words = text.split(' ')
        current_line = words[0]
        
        for word in words[1:]:
            if c.stringWidth(current_line + ' ' + word, 'Helvetica', 10) < max_width:
                current_line += ' ' + word
            else:
                lines.append(current_line)
                current_line = word
        
        lines.append(current_line)  # Add the last line
        return lines

    # Split content into paragraphs and wrap each paragraph
    paragraphs = content.split('\n')
    for paragraph in paragraphs:
        lines = wrap_text(paragraph, max_width)
        for line in lines:
            c.drawString(x, y, line)
            y -= line_height  # Move to the next line
            if y < 40:  # Check if the text is about to overflow the page
                c.showPage()  # Create a new page
                c.setFont("Helvetica", 10)  # Reset font for the new page
                y = height - 40  # Reset position for the new page

    # Save the PDF
    c.save()