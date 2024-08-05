from flask import Flask, request, jsonify, Response, stream_with_context, render_template_string
from openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate

app = Flask(__name__)

OPENAI_API_KEY = 'ollama'
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="http://127.0.0.1:11434/v1"
)

llm = Ollama(model="llama3.1:8b")
memory = ConversationBufferMemory()

MY_FAISS_INDEX_DIR = "faiss_index"
MY_FAISS_INDEX_FILE = os.path.join(MY_FAISS_INDEX_DIR, "index")

def load_documents_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(folder_path, filename))
            documents.extend(loader.load())
    return documents

def create_vectorstore(documents):
    if not documents:
        print("No documents to process")
        return None
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)
    
    if not splits:
        print("No splits to process")
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )

    vectorstore = FAISS.from_documents(splits, embedding=embeddings)
    vectorstore.save_local(MY_FAISS_INDEX_FILE)
    return vectorstore

def update_vectorstore(new_documents):
    if not new_documents:
        print("No new documents to process")
        return None
    
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )
    vectorstore = FAISS.load_local(MY_FAISS_INDEX_FILE,
                                   embeddings,
                                   allow_dangerous_deserialization=True)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    splits = text_splitter.split_documents(new_documents)
    
    if not splits:
        print("No splits to add to vectorstore")
        return vectorstore

    vectorstore.add_documents(splits, embedding=embeddings)

    vectorstore.save_local(MY_FAISS_INDEX_FILE)
    return vectorstore

documents_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'documents')

if not os.path.exists(MY_FAISS_INDEX_FILE + ".faiss"):
    os.makedirs(MY_FAISS_INDEX_DIR, exist_ok=True)

    documents = load_documents_from_folder(documents_folder)

    vectorstore = create_vectorstore(documents)
else:
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )

    vectorstore = FAISS.load_local(MY_FAISS_INDEX_FILE,
                                   embeddings,
                                   allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}) if vectorstore else None

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Answer in Korean.

# Question: 
{question} 
# Context: 
{context} 

# Answer:"""
)

class SimpleOutputParser:
    def __call__(self, output):
        return output

output_parser = SimpleOutputParser()

if retriever:
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | output_parser
    )
else:
    chain = None

system_prompt = """
친절한 인공지능과 호기심많은 사용자의 대화.
당신은 인공지능 챗봇이며 당신의 이름은 알프스입니다. 
항상 사용자의 질문에 자세하게 답변하고 대화를 자연스럽게 이어나가야합니다.
"""

def is_pdf_request(query):
    keywords = ["#"]
    return any(keyword in query for keyword in keywords)

def is_embedding_request(query):
    return "임베딩" in query

def generate_stream(query):
    global retriever
    previous_context = [message.content for message in memory.chat_memory.messages]

    if is_embedding_request(query):
        new_documents = load_documents_from_folder(documents_folder)
        vectorstore = update_vectorstore(new_documents)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6}) if vectorstore else None
        yield "새 문서가 추가되고 임베딩이 갱신되었습니다."
        return

    if is_pdf_request(query):
        retrieved_docs = retriever.get_relevant_documents(query) if retriever else []
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        full_prompt = system_prompt + "\n".join(previous_context) + "\n" + prompt_template.format(context=context, question=query)

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": full_prompt,
                }
            ],
            model="llama3.1:8b",
            stream=True  # 스트리밍 활성화
        )

        user_input = {"query": query}
        assistant_response = {"response": ""}
        for chunk in chat_completion:
            assistant_response["response"] += chunk.choices[0].delta.content
            yield chunk.choices[0].delta.content

        memory.save_context(user_input, assistant_response)
    else:
        full_prompt = system_prompt + "\n".join(previous_context) + "\n사용자: " + query + "\n답변:"

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": full_prompt,
                }
            ],
            model="llama3.1:8b",
            stream=True  # 스트리밍 활성화
        )

        user_input = {"query": query}
        assistant_response = {"response": ""}
        for chunk in chat_completion:
            assistant_response["response"] += chunk.choices[0].delta.content
            yield chunk.choices[0].delta.content

        memory.save_context(user_input, assistant_response)

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    query = data['query']
    
    return Response(stream_with_context(generate_stream(query)), content_type='text/event-stream')

@app.route('/')
def index():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <title>ALPS</title>
        <style>
            body, html {
                height: 100%;
                margin: 0;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                padding: 10px;
            }

            #title {
                text-align: center;
            }

            #chat-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                border-radius: 10px;
                width: 350px;
                max-width: 350px;
                position: relative;
            }

            #chat-box {
                width: 100%;
                height: 70vh;
                border: 3px solid black;
                border-radius: 0.5rem;
                padding: 10px;
                overflow-y: scroll;
                box-sizing: border-box;
                display: flex;
                flex-direction: column;
            }

            #input-container {
                display: flex;
                align-items: center;
                width: 100%;
                box-sizing: border-box;
                margin-top: 3px;
            }

            #input-wrapper {
                display: flex;
                align-items: center;
                width: 100%;
                border: 1px solid #ced4da;
                border-radius: 0.5rem;
                padding: 5px;
            }

            #message {
                flex: 1;
                border: none;
                outline: none;
                padding: 0 3px;
                border-radius: 0.5rem;
                box-sizing: border-box;
                resize: none;
                overflow: hidden;
                height: 38px; /* 한 줄 높이로 설정 */
                line-height: 38px; /* 텍스트 중앙 정렬 */
            }

            #input-icon {
                width: 24px;
                height: 24px;
                margin-left: 3px;
                margin-right: 6px;
                cursor: pointer;
                background: url('./img/arrow_up_circle.svg') no-repeat center center;
                background-size: contain;
                border: none;
                outline: none;
            }

            #mic-icon {
                width: 24px;
                height: 24px;
                margin-left: 3px;
                cursor: pointer;
                background: url('./img/mic.svg') no-repeat center center;
                background-size: contain;
                border: none;
                outline: none;
                display: none; /* 기본적으로 숨김 */
            }

            #file-input {
                display: none;
            }

            .user-message {
                text-align: right;
                margin-bottom: 5px;
                background-color: #d4edda; /* 연두색 배경 */
                padding: 5px;
                border-radius: 5px;
                display: inline-block;
                align-self: flex-end;
            }

            .bot-message {
                text-align: left;
                margin-bottom: 5px;
                background-color: #cce5ff; /* 하늘색 배경 */
                padding: 5px;
                border-radius: 5px;
                display: inline-block;
                align-self: flex-start;
            }

            .recognition-message {
                text-align: center;
                margin-bottom: 5px;
                background-color: #ffeeba; /* 노란색 배경 */
                padding: 5px;
                border-radius: 5px;
                display: inline-block;
                align-self: center;
            }

            @media (max-width: 768px) {
                #chat-box {
                    height: 60vh;
                }
            }

            @media (max-width: 480px) {
                #chat-container {
                    width: 322px;
                    max-width: 322px;
                }
                #chat-box {
                    width: 100%;
                    height: 70vh;
                }
                #input-container {
                    width: 100%;
                }
                #input-wrapper {
                    width: 100%;
                }
                #message {
                    width: 100%;
                }

                #mic-icon {
                    display: block; /* 모바일에서만 보이도록 설정 */
                }
            }
        </style>
    </head>
    <body>
    <h1 id="title">ALPHA-EDU ALPS</h1>
    <div id="chat-container">
        <div id="chat-box">
            <!--<p class="bot-message">알프스에게 궁금한점을 물어봐주세요.</p>-->
        </div>
        <div id="input-container">
            <div id="input-wrapper">
                <button id="input-icon" onclick="resetFileInput()"></button>
                <input type="file" id="file-input" onchange="uploadFile()" />
                <input type="hidden" name="mode" id="mode" value="upload"/>
                <textarea id="message" name="mode" value="chat" placeholder="내용을 입력하세요." class="form-control" rows="1"></textarea>
                <button id="mic-icon" onclick="toggleRecognition()"></button>
            </div>
        </div>
    </div>

    <script>
        let recognition;
        let recognizing = false;

        async function sendMessage() {
            var messageElement = document.getElementById("message");
            var message = messageElement.value;
            if (message.trim() === "") return;

            // 메시지 전송 전에 입력창 초기화
            messageElement.value = "";

            displayMessage(message, "user");

            const response = await fetch("/process", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ query: message })
            });

            if (response.status !== 200) {
                alert("Error: " + (await response.json()).error);
                return;
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let result = "";
            let botMessageDiv = displayMessage("", "bot");

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                result += decoder.decode(value, { stream: true });
                updateMessage(botMessageDiv, result);
            }

            // 음성으로 출력
            speak(result);
        }

        function displayMessage(message, sender) {
            var chatBox = document.getElementById("chat-box");
            var messageDiv = document.createElement("div");
            messageDiv.className = sender === "user" ? "user-message" : "bot-message";
            messageDiv.textContent = message;
            chatBox.appendChild(messageDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
            return messageDiv;
        }

        function updateMessage(messageDiv, message) {
            messageDiv.textContent = message;
            if (messageDiv.className.includes("bot-message")) {
                speak(message); // 메시지를 소리로 출력
            }
        }

        async function uploadFile() {
            const fileInput = document.getElementById('file-input');
            const modeInput = document.getElementById('mode');
            const file = fileInput.files[0];
            const mode = modeInput.value;

            if (file) {
                const formData = new FormData();
                formData.append('file', file);
                formData.append('mode', mode);

                const response = await fetch('/process', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    alert('파일 업로드 성공!');
                } else {
                    alert('파일 업로드 실패!');
                }
            }
        }

        function resetFileInput() {
            const fileInput = document.getElementById('file-input');
            fileInput.value = '';
            fileInput.click();
        }

        document.getElementById("message").addEventListener("keydown", function(event) {
            if (event.key === "Enter" && !event.shiftKey) {
                sendMessage();
                event.preventDefault(); // 기본 엔터 동작 방지
            }
        });

        function toggleRecognition() {
            if (!recognition) {
                if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
                    alert('Your browser does not support speech recognition. Please use a supported browser like Chrome.');
                    return;
                }

                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                recognition = new SpeechRecognition();
                recognition.lang = 'ko-KR';
                recognition.continuous = false;
                recognition.interimResults = true;

                recognition.onstart = function() {
                    console.log('Voice recognition started.');
                    displayRecognitionMessage("음성 인식 중...");
                    recognizing = true;
                };

                recognition.onresult = function(event) {
                    let interimTranscript = '';
                    for (let i = 0; i < event.results.length; i++) {
                        if (event.results[i].isFinal) {
                            document.getElementById("message").value = event.results[i][0].transcript;
                            sendMessage();
                        } else {
                            interimTranscript += event.results[i][0].transcript;
                        }
                    }
                    document.getElementById("message").value = interimTranscript;
                };

                recognition.onerror = function(event) {
                    if (event.error === 'aborted') {
                        displayRecognitionMessage("음성 인식 중단");
                    } else {
                        console.error('Voice recognition error:', event.error);
                        displayRecognitionMessage("음성 인식 오류: " + event.error);
                    }
                };

                recognition.onend = function() {
                    console.log('Voice recognition ended.');
                    removeRecognitionMessage();
                    recognizing = false;
                };
            }

            if (recognizing) {
                recognition.stop();
            } else {
                recognition.start();
            }
        }

        function displayRecognitionMessage(message) {
            removeRecognitionMessage();
            var chatBox = document.getElementById("chat-box");
            var messageDiv = document.createElement("div");
            messageDiv.className = "recognition-message";
            messageDiv.id = "recognition-message";
            messageDiv.textContent = message;
            chatBox.appendChild(messageDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        function removeRecognitionMessage() {
            var messageDiv = document.getElementById("recognition-message");
            if (messageDiv) {
                messageDiv.remove();
            }
        }

        function speak(text) {
            if ('speechSynthesis' in window) {
                const synth = window.speechSynthesis;
                if (synth.speaking) {
                    console.error('speechSynthesis.speaking');
                    return;
                }
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.lang = 'ko-KR';
                utterance.onend = function(event) {
                    console.log('SpeechSynthesisUtterance.onend');
                };
                utterance.onerror = function(event) {
                    console.error('SpeechSynthesisUtterance.onerror: ', event.error);
                };
                synth.speak(utterance);
            } else {
                console.log('Speech synthesis not supported in this browser.');
            }
        }
    </script>
    </body>
    </html>
    """
    return render_template_string(html_content)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
