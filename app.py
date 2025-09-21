import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import requests
from langchain.agents import create_react_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langchain import hub
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------- PDF Utilities ----------
@st.cache_data
def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF. Fallback to OCR if needed."""
    try:
        pdf_file.seek(0)
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        
        if len(doc) > 10:
            st.warning(f"PDF has {len(doc)} pages. Only processing first 10 pages for demo.")
            max_pages = 10
        else:
            max_pages = len(doc)
        
        for page_num in range(max_pages):
            page = doc[page_num]
            page_text = page.get_text()
            
            if not page_text.strip():
                # OCR not available on Streamlit Cloud
                page_text = "[OCR not available on Streamlit Cloud - please use text-based PDFs]"
                logger.warning(f"No text found on page {page_num} - OCR not available")
            
            text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        
        doc.close()
        
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF")
            
        return text
        
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

# -------- Embedding + FAISS ----------
@st.cache_resource
def build_vector_store(text):
    """Build vector store with proper text splitting."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_text(text)
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        
        if not chunks:
            raise ValueError("No valid text chunks created")
        
        embed_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        db = FAISS.from_texts(chunks, embed_model)
        return db
        
    except Exception as e:
        st.error(f"Error building vector store: {str(e)}")
        return None

# -------- Deepgram Tool ----------
def deepgram_stt(audio_file, api_key):
    """Convert audio to text using Deepgram API."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_file_path = tmp_file.name
        
        url = "https://api.deepgram.com/v1/listen"
        headers = {"Authorization": f"Token {api_key}"}
        
        with open(tmp_file_path, 'rb') as audio:
            resp = requests.post(
                url, 
                headers=headers, 
                files={"file": audio},
                timeout=30
            )
        
        os.unlink(tmp_file_path)
        
        if resp.status_code == 200:
            result = resp.json()
            transcript = result.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")
            return transcript
        else:
            logger.error(f"Deepgram API error: {resp.status_code} - {resp.text}")
            return "Error: Could not transcribe audio"
            
    except Exception as e:
        logger.error(f"Speech to text error: {str(e)}")
        return f"Error: {str(e)}"

# -------- Enhanced Search Tool ----------
def search_tool(query, db):
    """Enhanced search with engineering focus."""
    try:
        if not db:
            return "Error: No knowledge base available"
        
        # Enhance query with engineering context
        enhanced_query = f"""
        Engineering/Technical Query: {query}
        Focus on: specifications, safety requirements, procedures, compliance standards.
        """
        
        docs = db.similarity_search(enhanced_query, k=2)
        if not docs:
            return "No relevant technical information found in the uploaded document."
        
        results = []
        for i, doc in enumerate(docs, 1):
            results.append(f"**Technical Reference {i}:**\n{doc.page_content}")
        
        return "\n\n".join(results)
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return f"Search error: {str(e)}"

# -------- Agent Setup ----------
@st.cache_resource
def setup_agent(_db, gemini_api_key, deepgram_api_key):
    """Setup the LangChain agent with engineering expertise."""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=gemini_api_key,
            temperature=0.05,
            convert_system_message_to_human=True
        )
        
        tools = [
            Tool(
                name="EngineeringKnowledgeSearch",
                func=lambda q: search_tool(q, _db),
                description="Search engineering standards and technical documents for specifications, safety requirements, procedures, and compliance information. Use this for technical questions about the uploaded document."
            )
        ]
        
        try:
            prompt = hub.pull("hwchase17/react")
        except:
            # Enhanced engineering prompt
            from langchain.prompts import PromptTemplate
            prompt = PromptTemplate(
                template="""You are an AI assistant specialized in engineering standards and technical documentation.

Your expertise includes:
- Safety requirements and protocols
- Installation and maintenance procedures  
- Technical specifications and tolerances
- Compliance standards and regulations

When answering technical questions:
1. Be precise and use proper engineering terminology
2. Reference specific sections when possible
3. Highlight safety-critical information
4. Provide step-by-step guidance when applicable
5. Focus on technical accuracy over general information

You have access to these tools: {tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what technical information to find
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final technical answer
Final Answer: the final answer with technical details and proper engineering context

Begin!

Question: {input}
Thought:{agent_scratchpad}""",
                input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
            )
        
        agent = create_react_agent(llm, tools, prompt)
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,  # enable to see what's happening
            max_iterations=5,  
            max_execution_time=60,
            handle_parsing_errors=True,
            return_intermediate_steps=True  # help with debugging
        )
        
        return agent_executor
        
    except Exception as e:
        st.error(f"Error setting up agent: {str(e)}")
        return None

# -------- Streamlit App ----------
def main():
    st.set_page_config(
        page_title="AI Standards Explorer",
        page_icon="📋",
        layout="wide"
    )
    
    st.title("🤖 AI Standards Explorer")
    st.markdown("Upload engineering docs and ask questions! ⚡ **Limits: 10 pages, free APIs**")
    
    with st.sidebar:
        st.header("Configuration")
        
        # Streamlit Cloud uses st.secrets first, environment variables as fallback
        try:
            gemini_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
            deepgram_key = st.secrets.get("DEEPGRAM_API_KEY") or os.environ.get("DEEPGRAM_API_KEY")
            
            if not gemini_key:
                raise ValueError("GEMINI_API_KEY not found")
            if not deepgram_key:
                raise ValueError("DEEPGRAM_API_KEY not found")
                
            st.success("✅ API keys loaded")
            
        except Exception as e:
            st.error(f"❌ Missing API keys: {str(e)}")
            st.markdown("**Streamlit Cloud:** Add secrets in app settings")
            st.markdown("**Local:** Create `.streamlit/secrets.toml`")
            
            # Fallback: manual input for testing
            st.markdown("---")
            st.markdown("**Or enter manually for testing:**")
            gemini_key = st.text_input("Gemini API Key:", type="password")
            deepgram_key = st.text_input("Deepgram API Key:", type="password")
            
            if not gemini_key:
                st.warning("⚠️ Enter API keys to continue")
                st.stop()
        
        st.header("Demo Instructions")
        st.markdown("""
        🎯 **Hackathon Demo - Quick Test:**
        1. Upload small PDF (≤15 pages)
        2. Ask 1-2 questions max
        3. Free API limits apply
        """)
        
        if 'query_count' not in st.session_state:
            st.session_state.query_count = 0
        
        st.info(f"🔢 Questions asked: {st.session_state.query_count}/5")
        
        if st.session_state.query_count >= 5:
            st.warning("⚠️ Demo limit reached! Refresh to reset.")
    
    uploaded_pdf = st.file_uploader(
        "📄 Upload a PDF (≤ 10 pages for demo)", 
        type="pdf",
        help="Text-based PDFs work best for demo"
    )
    
    if uploaded_pdf:
        st.info(f"📁 Processing file: {uploaded_pdf.name} ({uploaded_pdf.size} bytes)")
        
        with st.spinner("🔄 Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_pdf)
        
        if text:
            st.success(f"✅ Extracted {len(text)} characters from PDF")
            
            with st.spinner("🔄 Building knowledge base..."):
                db = build_vector_store(text)
            
            if db:
                st.success("✅ Knowledge base ready!")
                
                with st.spinner("🔄 Setting up AI agent..."):
                    agent = setup_agent(db, gemini_key, deepgram_key)
                
                if agent:
                    st.success("✅ Engineering AI agent ready!")
                    st.info("🎯 **Enhanced for technical accuracy:** Lower temperature, engineering-focused prompts")
                    
                    st.subheader("💭 Ask a Question")
                    
                    tab1, tab2 = st.tabs(["💬 Text Input", "🎤 Voice Input"])
                    
                    with tab1:
                        q_text = st.text_input(
                            "Type your technical question:",
                            placeholder="e.g., What are the safety requirements for installation?"
                        )
                        
                        if q_text:
                            if st.session_state.query_count < 5:
                                st.session_state.query_count += 1
                                with st.spinner("🤔 Thinking..."):
                                    try:
                                        response = agent.invoke({"input": q_text})
                                        st.write("### 💡 Answer:")
                                        
                                        # Check if we got a proper response
                                        if "output" in response:
                                            st.write(response["output"])
                                        else:
                                            st.warning("⚠️ Agent reached time limit. Here's what I found so far:")
                                            # Try to get intermediate results
                                            if "intermediate_steps" in response:
                                                for step in response["intermediate_steps"]:
                                                    if len(step) > 1:
                                                        st.write(step[1])  # Show tool output
                                            else:
                                                st.write("Please try a simpler question or rephrase your query.")
                                                
                                    except Exception as e:
                                        # Fallback: direct search if agent fails
                                        if "Agent stopped" in str(e) or "iteration limit" in str(e):
                                            st.warning("🔄 Agent timed out, using direct search...")
                                            direct_result = search_tool(q_text, db)
                                            st.write("### 🔍 Direct Search Results:")
                                            st.write(direct_result)
                                        else:
                                            st.error(f"Error: {str(e)}")
                                            st.info("💡 Try asking a more specific question about the document.")
                            else:
                                st.warning("⚠️ Demo limit reached! Refresh page to reset.")
                    
                    with tab2:
                        q_audio = st.file_uploader(
                            "🎵 Upload audio question (mp3/wav)", 
                            type=["mp3", "wav"],
                            help="Record your question and upload the audio file"
                        )
                        
                        if q_audio:
                            st.audio(q_audio, format='audio/wav')
                            
                            with st.spinner("🎧 Transcribing audio..."):
                                transcript = deepgram_stt(q_audio, deepgram_key)
                            
                            if transcript and not transcript.startswith("Error"):
                                st.write(f"**📝 Transcript:** {transcript}")
                                
                                if st.session_state.query_count < 5:
                                    st.session_state.query_count += 1
                                    with st.spinner("🤔 Processing your question..."):
                                        try:
                                            response = agent.invoke({"input": transcript})
                                            st.write("### 💡 Answer:")
                                            
                                            # Check if we got a proper response
                                            if "output" in response:
                                                st.write(response["output"])
                                            else:
                                                st.warning("⚠️ Agent reached time limit. Here's what I found so far:")
                                                # Try to get intermediate results
                                                if "intermediate_steps" in response:
                                                    for step in response["intermediate_steps"]:
                                                        if len(step) > 1:
                                                            st.write(step[1])  # Show tool output
                                                else:
                                                    st.write("Please try a simpler question or rephrase your query.")
                                                    
                                        except Exception as e:
                                            st.error(f"Error: {str(e)}")
                                            st.info("💡 Try asking a more specific question about the document.")
                                else:
                                    st.warning("⚠️ Demo limit reached! Refresh page to reset.")
                            else:
                                st.error(f"Transcription failed: {transcript}")
                else:
                    st.error("❌ Failed to setup AI agent")
            else:
                st.error("❌ Failed to build knowledge base")
        else:
            st.error("❌ Failed to extract text from PDF")
    else:
        st.info("👆 Please upload a PDF to get started")
    
    st.markdown("---")
    st.markdown("Built using Streamlit, LangChain, and Google Gemini")

if __name__ == "__main__":
    main()
