
from flask import Blueprint, jsonify, render_template, request, session, send_from_directory
from integrate_ollama import create_cover_letter_pdf, generate_cv, get_jobs, integrate, resume_job_list
from pypdf import PdfReader
import json
core_bp = Blueprint("core", __name__)

job_datas=[]
company=[]
resume_data=""
def jobs_resume(data):
    reader = PdfReader(data)
    page = reader.pages[0]
    return page.extract_text()

@core_bp.route("/",methods=['GET', 'POST'])
def home():
    return render_template("core/index.html",message="")

@core_bp.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        print(user_message)
        response = integrate(user_message)
        print(response)
        print('response done')
        return {'reply': response}
    except Exception as e:  # Catch all exceptions for now
        print(f"Error: {e}")  # Log the error for debugging
        return jsonify({'error': 'Internal Server Error'}), 500

@core_bp.route('/upload_resume', methods=['GET','POST'])
def upload_resume():
    return render_template('core/job_resume.html')

@core_bp.route('/resume', methods=['GET','POST'])
def resume():
    # Handle the resume upload form submission
    resume_file=request.files['resume_file']
    if 'resume_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    if resume_file:
        
        resume_list=jobs_resume(resume_file)
       # print(resume_list)
        I,metadata=get_jobs(resume_list)
        data = resume_job_list(resume_list)
        print(type(data))
        print(data)
        job_data2 = []
        for job_json in data:
            try:
        # Parse the JSON string into a Python dictionary
               job_dict = json.loads(job_json)

        # Extract only the relevant information and create a new dictionary
               job_info = {
            'id': job_dict.get('id', 'Unknown'),
            'title': job_dict.get('title', 'Unknown'),
            'occupation': job_dict.get('occupation', 'Unknown'),
            'employer_description': job_dict.get('employer_description', 'No description available')
        }
               job_data2.append(job_info)
            except json.JSONDecodeError:
              print("Error decoding JSON")
              continue
        for idx in I[0]:
                job_datas.append(metadata[idx])
        return render_template('core/job_resume.html', job_data=job_data2)

@core_bp.route("/download/<int:id>")
def download(id):
    for jobs in job_datas:
        if jobs['id']==id:
            data=generate_cv(resume_data,jobs)
            print(data)
            create_cover_letter_pdf(str(id)+'.docx',data)
    return render_template("core/job_resume.html")