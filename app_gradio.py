import os
import re
import gradio as gr
import pandas as pd
from queue import Queue
from threading import Thread
from dotenv import load_dotenv

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

from ingest_blog import ingest_blog_url
from ingest_from_excel import main as ingest_main

from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant as QdrantVectorStore
from langchain_qdrant import Qdrant


# === Load .env ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === Qdrant + Embeddings Setup ===
EMBED_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-4o"

COLLECTION_NAME = "vendor_master"
CONTENT_PAYLOAD_FIELD = "text"
BLOG_COLLECTION_NAME = "blog_master"

embeddings = OpenAIEmbeddings(
    model=EMBED_MODEL,
    openai_api_key=OPENAI_API_KEY,
)

qdrant_client = QdrantClient(
    url="https://6161be3c-d699-4a62-88d4-18da8e947236.us-east-1-0.aws.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.TnC7y1uf_6YZMWpxDluxrxNMvy_1qR40-NKM4NUdtaE"
)

vectorstore = Qdrant(
    client=qdrant_client,
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    content_payload_key=CONTENT_PAYLOAD_FIELD,
)

blog_vectorstore = Qdrant(
    client=qdrant_client,
    embeddings=embeddings,
    collection_name=BLOG_COLLECTION_NAME,
    content_payload_key="chunk"
)

prompt_template = """
Always search all documents, suggest vendor according to relevant question in the higher priority of score first.

Guidelines:
- Suggest vendors that align with the user's criteria (location, service type, etc.).
- Rank vendors by descending score.
- If asked for a specific vendor, provide detailed info.
- Don't mention competitors.
- Don't hallucinate—respond from verifiable data.

{documents}

Query: {query}
"""

prompt = PromptTemplate(
    input_variables=["documents", "query"],
    template=prompt_template
)

blog_prompt_template = """
You are a helpful wedding assistant that provides information from wedding blogs.
Today's date is 9 july 2025
Guidelines:
- Show more textual data for easy understanding to user as per query.
- Use the provided blog content to answer questions related to weddings, fashion, makeup, venues, etc.
- ONLY show images, videos, and links that are **clearly relevant to the specific question asked**. Do **NOT** include decorative, unrelated, or generic images (like social media icons, logos, separators, etc.).
- Avoid showing media that does not enhance or directly support your answer.
- If helpful media (images/videos) is available and matches the topic, mention it in your response.
- Be clear, practical, and friendly in tone.
- Do **not** fabricate or assume details that aren't present in the provided blog content.
- If the question is unrelated to weddings or if the available data in our database is insufficient to answer accurately:
 * Clearly respond with: “No relevant data available at the moment.”
 * Do not include any images, videos, or links in such cases.
 * Do not attempt to guess or generate content beyond the scope of verified wedding-related data.


Chat History:
{history}

Blog Content:
{documents}

Question: {query}

Answer:
"""

blog_prompt = PromptTemplate(
    input_variables=["documents", "query", "history"],
    template=blog_prompt_template
)

class GradioStreamHandler(BaseCallbackHandler):
    def __init__(self, send_token):
        self.send_token = send_token

    def on_llm_new_token(self, token: str, **kwargs):
        self.send_token(token)

# def chat_fn(message: str, chat_history: list):
#     user_text = message.strip().lower()
#     results = vectorstore.similarity_search_with_score(user_text, k=10)
#     results.sort(key=lambda x: x[0].metadata.get("score", 0), reverse=True)
#     docs = [doc.page_content for doc, _ in results]

#     llm = ChatOpenAI(
#         model=CHAT_MODEL,
#         openai_api_key=OPENAI_API_KEY,
#     )

#     chain = LLMChain(llm=llm, prompt=prompt)
#     answer = chain.run(documents=docs, query=user_text)

#     chat_history = chat_history or []
#     chat_history.append({"role": "user", "content": message})
#     chat_history.append({"role": "assistant", "content": answer})
#     return "", chat_history

def chat_fn_stream(message: str, chat_history: list):
    from queue import Queue
    from threading import Thread
    import re  # used to patch URLs

    user_text = message.strip().lower()

    # List of casual greetings to bypass vector search
    greetings = {"hi", "hello", "hey", "hii", "heyy", "hello!", "hi!", "hey there"}

    if user_text in greetings:
        chat_history = chat_history or []
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": "Hello! How may I assist you today?"})
        yield "", chat_history
        return

    results = vectorstore.similarity_search_with_score(user_text, k=10)
    results.sort(key=lambda x: x[0].metadata.get("score", 0), reverse=True)
    docs = [doc.page_content for doc, _ in results]

    # Queue for streamed tokens
    q = Queue()

    def token_cb(token):
        q.put(token)

    handler = GradioStreamHandler(token_cb)

    stream_llm = ChatOpenAI(
        model=CHAT_MODEL,
        streaming=True,
        callbacks=[handler],
        openai_api_key=OPENAI_API_KEY,
    )

    chain = LLMChain(llm=stream_llm, prompt=prompt)

    def run_chain():
        try:
            chain.run(documents=docs, query=user_text)
        finally:
            q.put(None)

    Thread(target=run_chain).start()

    full = ""
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": ""})

    while True:
        token = q.get()
        if token is None:
            break
        full += token
        chat_history[-1]["content"] = full
        yield "", chat_history

    # 🔧 Fix malformed or local dev URLs
    def fix_links(text):
        text = re.sub(r'(?<!https://)(www\.wedmegood\.com/[^\s)\"]+)', r'https://\1', text)
        text = re.sub(r'https?://127\.0\.0\.1:\d+/((www\.wedmegood\.com/[^\s)\"]+))', r'https://\1', text)

        return text

    chat_history[-1]["content"] = fix_links(full)
    yield "", chat_history

def render_media_links(text, payloads):
    html = f"<div>{text}</div>"

    images = []
    videos = []
    links = []

    for payload in payloads:
        images.extend(payload.get("images", [])[:3])
        videos.extend(payload.get("videos", [])[:2])
        links.extend(payload.get("links", [])[:3])

    # Show images in a row, wrap if needed
    if images:
        html += """
        <div style='display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px;'>
        """
        for img in images:
            html += f"<img src='{img}' width='200' style='border-radius: 8px; max-width: 100%; height: auto;' loading='lazy'>"
        html += "</div>"

    # Show videos in a row, wrap if needed
    if videos:
        html += """
        <div style='display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px;'>
        """
        for vid in videos:
            html += f"<iframe src='{vid}' width='300' height='180' frameborder='0' style='max-width: 100%;' allowfullscreen></iframe>"
        html += "</div>"

    # Show links
    if links:
        html += "<div style='margin-top: 10px;'>"
        for link in links:
            html += f"<a href='{link}' target='_blank' style='margin-right: 10px;'>🔗 Link</a>"
        html += "</div>"

    return html

#     user_text = message.strip().lower()
#     results = blog_vectorstore.similarity_search_with_score(user_text, k=5)

#     docs = []
#     payloads = []

  
#     for doc,score in results:
#         print("Score:", score)
       
#         text = doc.page_content.strip()
#         payload = doc.metadata or {}
#         # print(payload)
#         if any(k not in payload for k in ["images", "videos", "links"]):
#             print(payload)
#             point_id = payload.get("_id")
#             print(point_id)
#             if point_id:
#                 result = qdrant_client.retrieve(
#                     collection_name=BLOG_COLLECTION_NAME,
#                     ids=[point_id],
#                     with_payload=True
#                 )
#                 if result:
#                     full_payload = result[0].payload
#                     for key in ["images", "videos", "links"]:
#                         if key in full_payload:
#                             payload[key] = full_payload[key]

#         docs.append(text)
#         # print(payload)
#         payloads.append(payload)
#         # print(payloads)
#     q = Queue()
#     def token_cb(token):
#         q.put(token)

#     handler = GradioStreamHandler(token_cb)
#     stream_llm = ChatOpenAI(
#         model=CHAT_MODEL,
#         streaming=True,
#         callbacks=[handler],
#         openai_api_key=OPENAI_API_KEY
#     )

#     chain = LLMChain(llm=stream_llm, prompt=blog_prompt)

#     def run_chain():
#         try:
#             chain.run(documents=docs, query=user_text)
#         finally:
#             q.put(None)

#     Thread(target=run_chain).start()

#     full = ""
#     chat_history = chat_history or []
#     chat_history.append({"role": "user", "content": message})
#     chat_history.append({"role": "assistant", "content": ""})
#     while True:
#         token = q.get()
#         if token is None:
#             break
#         full += token
#         chat_history[-1]["content"] = full
#         yield "", chat_history

#     chat_history[-1]["content"] = render_media_links(full, payloads)
#     yield "", chat_history
# def blog_chat_fn_stream(message, chat_history):
#     try:
#         user_text = message.strip().lower()
#         # List of casual greetings to bypass vector search
#         greetings = {"hi", "hello", "hey", "hii", "heyy", "hello!", "hi!", "hey there"}

#         if user_text in greetings:
#             chat_history = chat_history or []
#             chat_history.append({"role": "user", "content": message})
#             chat_history.append({"role": "assistant", "content": "Hello! How may I assist you today?"})
#             yield "", chat_history
#             return

#         results = blog_vectorstore.similarity_search_with_score(user_text, k=5)
#         results.sort(key=lambda x: x[1], reverse=True)  
       
#         docs = []
#         payloads = []
#         point_ids = []
#         id_to_doc = {}

#         for doc,score in results:
#             print("Score:", score)
#             point_id = doc.metadata.get("_id")
#             print(point_id)
#             if point_id:
#                 point_ids.append(point_id)
#                 id_to_doc[point_id] = doc
#             else:
#                 docs.append(doc.page_content.strip())
#                 payloads.append(doc.metadata or {})

#         full_payloads = {}
#         if point_ids:
#             try:
#                 retrieved = qdrant_client.retrieve(
#                     collection_name=BLOG_COLLECTION_NAME,
#                     ids=point_ids,
#                     with_payload=True
#                 )
#                 for item in retrieved:
#                     full_payloads[item.id] = item.payload
#             except Exception as e:
#                 print(f"⚠️ Payload fetch error: {e}")

#         for doc, _ in results:
#             point_id = doc.metadata.get("_id")
#             docs.append(doc.page_content.strip())
#             if point_id and point_id in full_payloads:
#                 payloads.append(full_payloads[point_id])
#             else:
#                 payloads.append(doc.metadata or {})

#         q = Queue()
#         def token_cb(token):
#             q.put(token)

#         handler = GradioStreamHandler(token_cb)
#         stream_llm = ChatOpenAI(
#             model=CHAT_MODEL,
#             streaming=True,
#             callbacks=[handler],
#             openai_api_key=OPENAI_API_KEY
#         )

#         chain = LLMChain(llm=stream_llm, prompt=blog_prompt)

#         def run_chain():
#             try:
#                 chain.run(documents=docs, query=user_text)
#             finally:
#                 q.put(None)

#         Thread(target=run_chain).start()

#         full = ""
#         chat_history = chat_history or []
#         chat_history.append({"role": "user", "content": message})
#         chat_history.append({"role": "assistant", "content": ""})

#         while True:
#             token = q.get()
#             if token is None:
#                 break
#             full += token
#             chat_history[-1]["content"] = full
#             yield "", chat_history


#         print("over")
#         chat_history[-1]["content"] = render_media_links(full, payloads)
#         yield "", chat_history

#     except Exception as e:
#         error_message = f"❌ Error: {str(e)}"
#         print(error_message)
#         chat_history = chat_history or []
#         chat_history.append({"role": "user", "content": message})
#         chat_history.append({"role": "assistant", "content": error_message})
#         yield "", chat_history

def blog_chat_fn_stream(message, chat_history):
    try:
        user_text = message.strip().lower()
        # List of casual greetings to bypass vector search
        greetings = {"hi", "hello", "hey", "hii", "heyy", "hello!", "hi!", "hey there"}

        if user_text in greetings:
            chat_history = chat_history or []
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": "Hello! How may I assist you today?"})
            yield "", chat_history
            return

        results = blog_vectorstore.similarity_search_with_score(user_text, k=5)
        results.sort(key=lambda x: x[1], reverse=True)  # Sort by similarity score

        docs = []
        payloads = []
        point_ids = []
        id_to_doc = {}

        for doc, score in results:
            point_id = doc.metadata.get("_id")
            print(point_id);
            print(score)
            if point_id:
                point_ids.append(point_id)
                id_to_doc[point_id] = doc
            else:
                docs.append(doc.page_content.strip())
                payloads.append(doc.metadata or {})

        full_payloads = {}
        if point_ids:
            try:
                retrieved = qdrant_client.retrieve(
                    collection_name=BLOG_COLLECTION_NAME,
                    ids=point_ids,
                    with_payload=True
                )
                for item in retrieved:
                    full_payloads[item.id] = item.payload
            except Exception as e:
                print(f"⚠️ Payload fetch error: {e}")

        for doc, _ in results:
            point_id = doc.metadata.get("_id")
            docs.append(doc.page_content.strip())
            if point_id and point_id in full_payloads:
                payloads.append(full_payloads[point_id])
            else:
                payloads.append(doc.metadata or {})

        # 🔁 Construct chat history to provide context
        history_text = ""
        for turn in chat_history or []:
            if turn["role"] == "user":
                history_text += f"User: {turn['content']}\n"
            elif turn["role"] == "assistant":
                history_text += f"Assistant: {turn['content']}\n"

        q = Queue()
        def token_cb(token):
            q.put(token)

        handler = GradioStreamHandler(token_cb)

        stream_llm = ChatOpenAI(
            model=CHAT_MODEL,
            streaming=True,
            callbacks=[handler],
            openai_api_key=OPENAI_API_KEY
        )

        # ✅ Now the chain receives "history"
        chain = LLMChain(llm=stream_llm, prompt=blog_prompt)

        def run_chain():
            try:
                chain.run(documents=docs, query=user_text, history=history_text)
            finally:
                q.put(None)

        Thread(target=run_chain).start()

        full = ""
        chat_history = chat_history or []
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": ""})

        while True:
            token = q.get()
            if token is None:
                break
            full += token
            chat_history[-1]["content"] = full
            yield "", chat_history

        chat_history[-1]["content"] = render_media_links(full, payloads)
        yield "", chat_history

    except Exception as e:
        error_message = f"❌ Error: {str(e)}"
        print(error_message)
        chat_history = chat_history or []
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": error_message})
        yield "", chat_history


def upload_excel_and_ingest(uploaded_file):
    if uploaded_file is None:
        return "❌ Please upload a .xlsx file first."
    try:
        ingest_main(uploaded_file.name)
        return f"✅ Successfully ingested `{os.path.basename(uploaded_file.name)}`"
    except Exception as e:
        return f"❌ Ingestion error: {e}"

def process_blog_excel(file):
    if not file:
        return "❌ Please upload a valid Excel file."
    try:
        df = pd.read_excel(file.name)
        if "blog_links" not in df.columns:
            return "❌ Excel must contain a column named 'url'."

        logs = []
        for idx, row in df.iterrows():
            url = str(row["blog_links"]).strip()
            if url:
                try:
                    result = ingest_blog_url(url)
                    logs.append(result)
                except Exception as e:
                    logs.append(f"❌ Error with {url}: {e}")
        return "\n".join(logs)
    except Exception as e:
        return f"❌ Failed to process Excel: {e}"

with gr.Blocks(title="WedMeGood Vendor Assistant") as demo:
    gr.Markdown("## 💬 WedMeGood Vendor Chatbot + Ingestion")

    with gr.Tab("📥 Upload & Ingest"):
        file_upload = gr.File(label="Upload Excel (.xlsx)", file_types=[".xlsx"])
        ingest_status = gr.Textbox(label="Status", interactive=False)
        ingest_btn = gr.Button("📤 Ingest to Qdrant")
        ingest_btn.click(fn=upload_excel_and_ingest, inputs=[file_upload], outputs=[ingest_status], queue=False)

    with gr.Tab("🌐 Ingest Blog URL"):
        blog_url = gr.Textbox(label="Enter Blog URL")
        blog_status = gr.Textbox(label="Status", interactive=False)
        blog_btn = gr.Button("Ingest Single Blog")
        blog_btn.click(fn=ingest_blog_url, inputs=[blog_url], outputs=[blog_status])

    with gr.Tab("📄 Upload Blog Excel"):
        blog_excel = gr.File(label="Upload Blog Excel (with 'url' column)", file_types=[".xlsx"])
        blog_log = gr.Textbox(label="Ingestion Logs", lines=20, interactive=False)
        blog_excel_btn = gr.Button("📤 Ingest Blog URLs")
        blog_excel_btn.click(fn=process_blog_excel, inputs=[blog_excel], outputs=[blog_log])

    with gr.Tab("📰 Chat with Blogs"):
        blog_chatbot = gr.Chatbot(label="Chat with Wedding Blogs", type="messages")  # ✅
        blog_msg = gr.Textbox(placeholder="Ask about blogs (e.g., bridal makeup tips)", label="Ask something...")
        blog_clear = gr.Button("Clear Blog Chat")
        blog_msg.submit(blog_chat_fn_stream, inputs=[blog_msg, blog_chatbot], outputs=[blog_msg, blog_chatbot])

        blog_clear.click(lambda: "", None, blog_chatbot)

    with gr.Tab("💬 Chat with Vendors"):
        chatbot = gr.Chatbot(type="messages")
        msg = gr.Textbox(placeholder="Ask about photographers, venues...", label="Your Query")
        clear = gr.Button("Clear Chat")

        # msg.submit(chat_fn, [msg, chatbot], [msg, chatbot])
        msg.submit(chat_fn_stream, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()