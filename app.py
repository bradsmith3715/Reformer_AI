from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import json
import chainlit as cl
import asyncio
from typing import AsyncIterator

# Load documents and create retriever
with open("combined_data.json", "r") as f:
    raw_data = json.load(f)

all_docs = [
    Document(page_content=entry["content"], metadata=entry["metadata"])
    for entry in raw_data
]

# === Split documents into chunks ===
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
chunked_docs = splitter.split_documents(all_docs)

# === Use your fine-tuned Hugging Face embeddings ===
embedding_model = HuggingFaceEmbeddings(
    model_name="bsmith3715/legal-ft-demo_final"
)

# === Set up FAISS vector store ===
vectorstore = FAISS.from_documents(chunked_docs, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Define a custom RAG prompt template
rag_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a knowledgeable pilates instructor. Use the following context to answer the question accurately and comprehensively.
When possible, cite specific sections or sources from the context.

Context:
{context}

Question:
{question}

Answer:"""
)

# Create the RAG chain with the custom prompt
llm = ChatOpenAI(model="gpt-4o-mini", streaming=True, temperature=0)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": rag_template}
)

@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    await cl.Message(
        content=
        """👋 Welcome to your Reformer Pilates AI!
Here's what you can do:
• Ask questions about Reformer Pilates
• Get individualized workouts based on your level, goals, and equipment
• Get instant exercise modifications based on injuries or limitations
Let's get started! 🚀""",
        author="Assistant"
    ).send()

#@cl.on_message
#async def main(message: cl.Message):
#    """Handle incoming messages with proper streaming"""
#    
#    # Create a message placeholder for streaming
#    msg = cl.Message(content="", author="Assistant")
#    await msg.send()
#    
#    try:
#        # Get the response from the RAG chain
#        response = await rag_chain.acall({"query": message.content})
#        
#        # Update the message with the full response
#        msg.content = response["result"]
#        await msg.update()
#        
#        # Optionally, show retrieved documents as context
#        if hasattr(response, 'source_documents') or 'source_documents' in response:
#            source_docs = response.get('source_documents', [])
#            if source_docs:
#                sources_text = "\n\n**Sources:**\n"
#                for i, doc in enumerate(source_docs[:3]):  # Show top 3 sources
#                    sources_text += f"{i+1}. {doc.metadata.get('source', 'Unknown')}\n"
                
 #               sources_msg = cl.Message(content=sources_text, author="Sources")
 #               await sources_msg.send()
                
#    except Exception as e:
#        error_msg = f"I apologize, but I encountered an error: {str(e)}"
#        msg.content = error_msg
#        await msg.update()


@cl.on_message
async def main(message: cl.Message):
    # Get context
    docs = retriever.get_relevant_documents(message.content)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = rag_template.format(context=context, question=message.content)
    
    # Stream response
    msg = cl.Message(content="")
    async for chunk in llm.astream(prompt):
        await msg.stream_token(chunk.content)
    await msg.send()

# === Load LLM ===
#llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0, stream = True)
#qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# === Chainlit start event ===
#@cl.on_chat_start
#async def start():
#    await cl.Message("""👋 Welcome to your Reformer Pilates AI!
#Here's what you can do:
#• Ask questions about Reformer Pilates
#• Get individualized workouts based on your level, goals, and equipment
#• Get instant exercise modifications based on injuries or limitations
#Let's get started! 🚀""").send()
#    cl.user_session.set("qa_chain", qa_chain)

# === Chainlit message handler ===
#@cl.on_message
#async def handle_message(message: cl.Message):
#    chain = cl.user_session.get("qa_chain")
#    if chain:
#        try:
#            response = chain.run(message.content)
#        except Exception as e:
#            response = f"⚠️ Error: {str(e)}"
#        await cl.Message(response).send()
