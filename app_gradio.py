import os
import re
import gradio as gr
from dotenv import load_dotenv
import pandas as pd

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from ingest_blog import ingest_blog_url


# ❗️ Switch to the new Qdrant import
from langchain_qdrant import Qdrant  
from qdrant_client import QdrantClient

from ingest_from_excel import main as ingest_main

# === Load .env ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === Qdrant + Embeddings Setup ===
EMBED_MODEL = "text-embedding-ada-002"
CHAT_MODEL  = "gpt-4o-mini"

COLLECTION_NAME       = "vendor_master"
CONTENT_PAYLOAD_FIELD = "text"

BLOG_COLLECTION_NAME = "blog_data3"


# initialize embedding model
embeddings = OpenAIEmbeddings(
    model=EMBED_MODEL,
    openai_api_key=OPENAI_API_KEY,
)

# Qdrant client
qdrant_client = QdrantClient(url="http://localhost:6333", prefer_grpc=False)

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

# === Prompt Template ===
prompt_template = """
Always search all documents, suggest vendor according to relevant question in the higher priority of action score first.

Guidelines:
- Suggest vendors that align with the user’s criteria (location, service type, etc.).
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

Guidelines:
- Use the provided blog content to answer questions about weddings, fashion, makeup, venues, etc.
- Include relevant images and videos when mentioned in the content.
- Provide practical and actionable advice.
- If images or videos are available, mention them in your response.
- Be conversational and helpful.
- Don't make up information not present in the documents.

Blog Content:
{documents}

Question: {query}

Answer:
"""



blog_prompt = PromptTemplate(
    input_variables=["documents", "query"],
    template=blog_prompt_template
)


# === LLM Setup ===
llm = ChatOpenAI(
    model=CHAT_MODEL,
    openai_api_key=OPENAI_API_KEY,
)

def chat_fn(message: str, chat_history: list):
    # lower‐case so user casing (“Dreamweavers” vs “dreamweavers”) still matches embeddings
    user_text = message.strip().lower()

    # retrieve top-10
    results = vectorstore.similarity_search_with_score(user_text, k=10)
    # sort by our “score” in payload
    results.sort(key=lambda x: x[0].metadata.get("score", 0), reverse=True)

    docs = [doc.page_content for doc, _ in results]

    chain = LLMChain(llm=llm, prompt=prompt)
    answer = chain.run(documents=docs, query=user_text)

    chat_history = chat_history or []
    chat_history.append({"role": "user",   "content": message})
    chat_history.append({"role": "assistant", "content": answer})
    return "", chat_history

def blog_chat_fn(message: str, blog_chat_history: list):
    user_text = message.strip().lower()
    results = blog_vectorstore.similarity_search_with_score(user_text, k=5)
    docs = []
    for doc, _ in results:
        text = doc.page_content.strip()
        payload = doc.metadata or {}
        if "images" not in payload or "videos" not in payload or "links" not in payload:
            point_id = payload.get("_id")
            if point_id:
                result = qdrant_client.retrieve(
                    collection_name="blog_data3",
                    ids=[point_id],
                    with_payload=True
                )
                if result:
                    full_payload = result[0].payload
                    # Update payload with images/videos/links if they exist
                    for key in ["images", "videos", "links"]:
                        if key in full_payload:
                            payload[key] = full_payload[key]
        print(payload)
        images = payload.get("images", [])[:3]
        videos = payload.get("videos", [])[:2]
        links  = payload.get("links", [])[:3]

        img_md = "\n".join(f"[Image]({url})" for url in images)
        vid_md = "\n".join(f"[Video]({url})" for url in videos)
        link_md = "\n".join(f"[Instagram]({url})" if "instagram.com" in url else f"[Link]({url})" for url in links)

        full_chunk = f"{text}\n\n{img_md}\n\n{vid_md}\n\n{link_md}"
        docs.append(full_chunk)

    chain = LLMChain(llm=llm, prompt=blog_prompt)
    raw_response = chain.run(documents=docs, query=user_text)
    final_response = render_media_links(raw_response)

    blog_chat_history = blog_chat_history or []
    blog_chat_history.append({"role": "user", "content": message})
    blog_chat_history.append({"role": "assistant", "content": final_response})
    return "", blog_chat_history


def render_media_links(text: str) -> str:
    html = text

    image_links = re.findall(r"\[Image]\((https?://[^\)]+)\)", text)
    for img in image_links:
        html += f"<br><img src='{img}' width='300' loading='lazy'>"

    video_links = re.findall(r"\[Video]\((https?://[^\)]+)\)", text)
    for vid in video_links:
        html += f"<br><iframe src='{vid}' width='400' height='250' frameborder='0' allowfullscreen></iframe>"

    insta_links = re.findall(r"\[Instagram]\((https?://[^\)]+)\)", text)
    for link in insta_links:
        html += f"<br><a href='{link}' target='_blank'>📸 Instagram Link</a>"

    other_links = re.findall(r"\[Link]\((https?://[^\)]+)\)", text)
    for link in other_links:
        html += f"<br><a href='{link}' target='_blank'>🔗 Link</a>"

    return html


def upload_excel_and_ingest(uploaded_file):
    if uploaded_file is None:
        return "❌ Please upload a .xlsx file first."
    try:
        # ingest_main should accept a local path
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
        ingest_btn.click(
            fn=upload_excel_and_ingest,
            inputs=[file_upload],
            outputs=[ingest_status],
            queue=False
        )
        
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
        blog_chatbot = gr.Chatbot(label="Chat with Wedding Blogs", type="messages")
        blog_msg = gr.Textbox(placeholder="Ask about blogs (e.g., bridal makeup tips)", label="Ask something...")
        blog_clear = gr.Button("Clear Blog Chat")

        blog_msg.submit(blog_chat_fn, [blog_msg, blog_chatbot], [blog_msg, blog_chatbot])
        blog_clear.click(lambda: [], None, blog_chatbot, queue=False)

    with gr.Tab("💬 Chat with Vendors"):
        chatbot = gr.Chatbot(type="messages")   # <-- explicitly set type="messages"
        msg     = gr.Textbox(placeholder="Ask about photographers, venues...", label="Your Query")
        clear   = gr.Button("Clear Chat")

        msg.submit(chat_fn, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()

