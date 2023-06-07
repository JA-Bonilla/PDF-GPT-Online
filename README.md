# PDF-Review-Online
Use the power of OpenAI's GPT-3.5 turbo model to get a better understanding of your documents! 

### Note: OpenAI collects all data transmitted to it's servers, do not input proprietary or confidential information into this program.

This was built with [OpenAI GPT-3.5](Openai.com), [Langchain](https://github.com/hwchase17/langchain), [Tkinter-Designer](https://github.com/ParthJadhav/Tkinter-Designer), [Figma](www.figma.com), and [PyPdfLoader](https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/pdf.html).

![Screenshot of PDF-GPT Output](/init/PDF-GPT.png)

# Enviorment Setup
In order to set your environment up to run the code here, first install all requirements using:
`pip install -r requirements.txt`

In the `GUI.py` users will need to insert their OpenAI API Key to use this program as OpenAI charges for the use of their API. The variable for setting the OpenAI API Key is `os.environ["OPENAI_API_KEY"]=""` where users will input their key in the empty quotations `""`.

-----------------------------------------

# System Requirements
## Python

This project works with Python 3.10 or later (though later versions are untested). Earlier versions of Python will not compile.

-----------------------------------------

# Ingestion 
Uploaded documents, specifically PDF's, must be readable (Optical Character Recognition (OCR) funcionality must be applicable).

When running `Gui.py` click upload and a filedialog will open, requesting you to select a PDF file. Once ingestion is complete the red image next to the file name will turn green and you can proceed to communicate with your documents.

-----------------------------------------

# Response Generation

Once a question has been typed the user clicks the Generate button and relavent documents will be compiled and delivered to the OpenAI Model through their API. The model will take the query and surmize a response based on the given documents. Reference pages of the most relavent information will be show in the references section of the GUI.
