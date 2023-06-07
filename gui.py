import os
import sys
import webbrowser
import re
import math
import ntpath
import textwrap

from pathlib import Path

from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
from tkinter.filedialog import askopenfilename
from tkPDFViewer import tkPDFViewer as pdf

from dotenv import find_dotenv, load_dotenv
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents.load_tools import get_all_tool_names
from langchain import ConversationChain
from langchain.utilities import SerpAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


from langchain.document_loaders import PyPDFLoader


load_dotenv(find_dotenv())

os.environ["OPENAI_API_KEY"] = ""

embeddings = OpenAIEmbeddings()

def Create_db_from_PDF():
    if m_FileName.lower().endswith(('.pdf')):
        loader = PyPDFLoader(load_path)

    print("Motor Library Found, Database Initializing...")
    
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)

    print("Database Set")
    return db

def get_response_from_query(db, query, k=4):
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature = 0.2)

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"C:\Users\j.bonilla\Desktop\Motor-GPT\New\build\assets\frame0")

f_types = [('PDF Files', '*.pdf'), 
           ('CSV Files', '*.csv'),
           ('Word Docx File', '*.docx'),
           ('PowerPoint File', '*.pptx')]
iterate = 0

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def Upload_Clicked():
    global load_path, db, m_FileName
    FileName.config(state="normal")
    load_path = askopenfilename(filetypes=f_types)
    m_FileName = ntpath.basename(load_path)
    FileName.insert(0 , m_FileName)

    db = Create_db_from_PDF()

    canvas.delete(image_2)
    FileName.config(state="readonly")

def iteration():
    global iterate
    print("In Loop")
    if iterate == 1:
        Generate()
        iterate = 0 
    else:
        window.after(1000, iteration)  # run again after 1000ms (1s)

def Pre_Gen():
    global iterate
    button_1.configure(image=button_image_3)
    iterate = 1
    print(iterate)

def Generate():
    Answer.config(state="normal")
    Ref1.config(state="normal")
    Ref2.config(state="normal")
    Ref3.config(state="normal")
    Answer.delete(1.0, "end")
    Ref1.delete(0, "end")
    Ref2.delete(0, "end")
    Ref3.delete(0, "end")

    print(Question.get(1.0,"end"))

    query = Question.get(1.0,"end")

    response, docs = get_response_from_query(db, query)

    typeit(Answer, "1.0", response)

    if m_FileName.lower().endswith(('.pdf')):

        doc1_str = str(docs[0])
        doc1 = len(doc1_str)

        RefP1 = "Page: " + doc1_str[doc1-3] + doc1_str[doc1-2]

        Ref1.insert(0, RefP1)

        doc1_str = str(docs[1])
        doc1 = len(doc1_str)

        RefP1 = "Page: " + doc1_str[doc1-3] + doc1_str[doc1-2]

        Ref2.insert(0, RefP1)

        doc1_str = str(docs[2])
        doc1 = len(doc1_str)

        RefP1 = "Page: " + doc1_str[doc1-3] + doc1_str[doc1-2]

        Ref3.insert(0, RefP1)

    Ref1.config(state="readonly")
    Ref2.config(state="readonly")
    Ref3.config(state="readonly")

    button_1.configure(image=button_image_1)
    window.after(1000, iteration)  

def typeit(widget, index, string):
   Answer.config(state="normal")
   if len(string) > 0:
      widget.insert(index, string[0])
      Answer.config(state="disabled")
      if len(string) > 1:
         # compute index of next char
         index = widget.index("%s + 1 char" % index)

         # type the next character in half a second
         widget.after(25, typeit, widget, index, string[1:])

template = """
        You are a helpful assistant that can answer questions about the given Document: {docs}

        Only use the factual information from the handbook to answer the question. 

        When asked for equations only provide reference as to where in the document the related equation could be found.
        
        If you feel like you do not have enough information to answer the question, say "I don't know".

        Your answers should be verbose and detailed.
        """

window = Tk()
window.title("PDF-GPT")

window.geometry("580x900")
window.configure(bg = "#3B3B3B")

canvas = Canvas(
    window,
    bg = "#3B3B3B",
    height = 900,
    width = 1080,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
entry_image_1 = PhotoImage(
    file=relative_to_assets("entry_1.png"))
entry_bg_1 = canvas.create_image(
    341.0,
    81.0,
    image=entry_image_1
)
FileName = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0,
    font='Bahnschrift 14'
)
FileName.place(
    x=201.0,
    y=56.0,
    width=280.0,
    height=48.0
)

entry_image_2 = PhotoImage(
    file=relative_to_assets("entry_2.png"))
entry_bg_2 = canvas.create_image(
    290.0,
    205.0,
    image=entry_image_2
)
Question = Text(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0,
    font='Bahnschrift 12',
    wrap='word'
)
Question.place(
    x=145.0,
    y=165.0,
    width=375.0,
    height=85.0
)

image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    94.0,
    176.0,
    image=image_image_1
)

entry_image_3 = PhotoImage(
    file=relative_to_assets("entry_3.png"))
entry_bg_3 = canvas.create_image(
    289.0,
    492.5,
    image=entry_image_3
)
Answer = Text(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0,
    font='Bahnschrift 12',
    wrap='word'
)
Answer.place(
    x=130.0,
    y=380.0,
    width=385.0,
    height=235.0
)

entry_image_4 = PhotoImage(
    file=relative_to_assets("entry_4.png"))
entry_bg_4 = canvas.create_image(
    290.0,
    666.0,
    image=entry_image_4
)
Ref1 = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0,
    font='Bahnschrift 12'
)
Ref1.place(
    x=180.0,
    y=640.0,
    width=348.0,
    height=50.0
)

button_image_3 = PhotoImage(file=relative_to_assets("button_3.png"))

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    bg="#3B3B3B",
    command=Pre_Gen,
    activebackground="#3B3B3B",
    relief="flat"
)
button_1.place(
    x=120.0,
    y=262.0,
    width=342.0,
    height=41.0
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    bg="#3B3B3B",
    command= Upload_Clicked,
    activebackground="#3B3B3B",
    relief="flat"
)
button_2.place(
    x=40.0,
    y=61.0,
    width=126.0,
    height=41.0
)

image_image_3 = PhotoImage(file=relative_to_assets("image_3.png"))
image_3 = canvas.create_image(
        530.0,
        81.0,
        image=image_image_3
    )

image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    530.0,
    81.0,
    image=image_image_2
)

image_image_4 = PhotoImage(
    file=relative_to_assets("image_4.png"))
image_4 = canvas.create_image(
    86.0,
    390.0,
    image=image_image_4
)

image_image_5 = PhotoImage(
    file=relative_to_assets("image_5.png"))
image_5 = canvas.create_image(
    112.0,
    662.0,
    image=image_image_5
)

entry_image_5 = PhotoImage(
    file=relative_to_assets("entry_5.png"))
entry_bg_5 = canvas.create_image(
    289.0,
    737.0,
    image=entry_image_5
)
Ref2= Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0,
    font='Bahnschrift 12'
)
Ref2.place(
    x=180.0,
    y=712.0,
    width=348.0,
    height=50.0
)

image_image_6 = PhotoImage(
    file=relative_to_assets("image_6.png"))
image_6 = canvas.create_image(
    114.0,
    733.0,
    image=image_image_6
)

entry_image_6 = PhotoImage(
    file=relative_to_assets("entry_6.png"))
entry_bg_6 = canvas.create_image(
    290.0,
    808.0,
    image=entry_image_6
)
Ref3 = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0,
    font='Bahnschrift 12'
)
Ref3.place(
    x=180.0,
    y=783.0,
    width=348.0,
    height=50.0
)

image_image_7 = PhotoImage(
    file=relative_to_assets("image_7.png"))
image_7 = canvas.create_image(
    115.0,
    804.0,
    image=image_image_7
)

window.after(1000, iteration)
window.resizable(False, False)
window.mainloop()
