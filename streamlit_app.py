import streamlit as st
import torch
import time
import traceback
from datetime import datetime
from pathlib import Path

# Configure Streamlit page
st.set_page_config(
    page_title="Urdu AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Noto+Nastaliq+Urdu:wght@400;500;600&display=swap');

    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --accent-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --dark-gradient: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 100%);
        --glass-bg: rgba(255, 255, 255, 0.08);
        --glass-border: rgba(255, 255, 255, 0.15);
        --text-primary: #ffffff;
        --text-secondary: #b4c6fc;
        --text-muted: #8892b0;
        --success: #64ffda;
        --shadow-lg: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.2);
    }

    .stApp {
        background: var(--dark-gradient);
        color: var(--text-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        min-height: 100vh;
    }

    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 20% 50%, rgba(120, 119, 198, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.05) 0%, transparent 50%);
        pointer-events: none;
        z-index: -1;
        animation: float 30s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-5px) rotate(0.5deg); }
    }

    .main-container {
        background: var(--glass-bg);
        backdrop-filter: blur(15px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem auto;
        max-width: 900px;
        box-shadow: var(--shadow-lg);
        position: relative;
    }

    .main-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--primary-gradient);
        opacity: 0.6;
        border-radius: 20px 20px 0 0;
    }

    .chat-message {
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        border-radius: 16px;
        max-width: 85%;
        word-wrap: break-word;
        line-height: 1.6;
        position: relative;
        backdrop-filter: blur(10px);
        animation: slideIn 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }

    @keyframes slideIn {
        from { opacity: 0; transform: translateY(15px) scale(0.98); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }

    .user-message {
        background: var(--primary-gradient);
        color: var(--text-primary);
        margin-left: auto;
        text-align: right;
        font-family: 'Noto Nastaliq Urdu', 'Inter', sans-serif;
        font-size: 1rem;
        font-weight: 500;
        box-shadow: var(--shadow-lg);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .user-message::before {
        content: '👤';
        position: absolute;
        top: -6px;
        right: -6px;
        width: 24px;
        height: 24px;
        background: var(--accent-gradient);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
        box-shadow: var(--shadow-lg);
    }

    .assistant-message {
        background: rgba(22, 33, 62, 0.7);
        color: var(--text-secondary);
        border: 1px solid rgba(116, 75, 162, 0.3);
        margin-right: auto;
        font-family: 'Noto Nastaliq Urdu', 'Inter', sans-serif;
        font-size: 0.95rem;
        line-height: 1.6;
        box-shadow: var(--shadow-lg);
        backdrop-filter: blur(15px);
    }

    .assistant-message::before {
        content: '🤖';
        position: absolute;
        top: -6px;
        left: -6px;
        width: 24px;
        height: 24px;
        background: var(--secondary-gradient);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
        box-shadow: var(--shadow-lg);
    }

    .header-title {
        text-align: center;
        background: var(--primary-gradient);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
        animation: glow 3s ease-in-out infinite alternate;
    }

    @keyframes glow {
        from { filter: drop-shadow(0 0 15px rgba(120, 119, 198, 0.2)); }
        to { filter: drop-shadow(0 0 25px rgba(120, 119, 198, 0.4)); }
    }

    .stTextArea textarea {
        border: 2px solid rgba(116, 75, 162, 0.3) !important;
        border-radius: 14px !important;
        font-size: 1rem !important;
        color: var(--text-primary) !important;
        background: rgba(22, 33, 62, 0.5) !important;
        backdrop-filter: blur(10px) !important;
        padding: 16px !important;
        font-family: 'Noto Nastaliq Urdu', 'Inter', sans-serif !important;
        line-height: 1.7 !important;
        resize: vertical !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    .stTextArea textarea:focus {
        border: 2px solid #667eea !important;
        box-shadow: 0 0 15px rgba(120, 119, 198, 0.2) !important;
        outline: none !important;
    }

    .stTextArea label {
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        margin-bottom: 10px !important;
    }

    .stButton > button[kind="primary"] {
        background: var(--primary-gradient) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 28px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        height: 50px !important;
        box-shadow: var(--shadow-lg) !important;
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5) !important;
    }

    .stButton > button:not([kind="primary"]) {
        background: rgba(116, 75, 162, 0.15) !important;
        color: var(--text-secondary) !important;
        border: 1px solid rgba(116, 75, 162, 0.3) !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        font-weight: 500 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        height: 42px !important;
        backdrop-filter: blur(10px) !important;
    }

    .stButton > button:not([kind="primary"]):hover {
        background: rgba(116, 75, 162, 0.25) !important;
        border-color: rgba(116, 75, 162, 0.5) !important;
        color: var(--text-primary) !important;
        transform: translateY(-1px) !important;
    }

    .stSidebar {
        background: rgba(12, 12, 12, 0.95) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(116, 75, 162, 0.2) !important;
    }

    .stSidebar .stMarkdown h3 {
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        margin-bottom: 1rem !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 2px solid var(--primary-gradient) !important;
        background: var(--primary-gradient) !important;
        background-clip: text !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }

    .message-time {
        font-size: 0.8rem;
        opacity: 0.6;
        margin-top: 0.75rem;
        font-weight: 400;
        color: var(--text-muted);
    }

    .loading-animation {
        display: inline-block;
        width: 36px;
        height: 36px;
        border: 3px solid rgba(120, 119, 198, 0.2);
        border-radius: 50%;
        border-top-color: #667eea;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    .word-count {
        color: var(--text-muted);
        font-size: 0.8rem;
        text-align: right;
        margin-top: 6px;
        font-weight: 500;
        opacity: 0.4;
    }

    @media (max-width: 768px) {
        .main-container {
            margin: 0.5rem;
            padding: 1rem;
            border-radius: 16px;
        }
        .chat-message {
            max-width: 95%;
            padding: 0.75rem 1rem;
            font-size: 0.9rem;
        }
        .header-title {
            font-size: 2rem;
        }
    }

    ::-webkit-scrollbar {
        width: 6px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(116, 75, 162, 0.1);
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb {
        background: var(--primary-gradient);
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# SESSION STATE INITIALIZATION
# ============================================

def init_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'model_loading' not in st.session_state:
        st.session_state.model_loading = False
    if 'error_state' not in st.session_state:
        st.session_state.error_state = None
    if 'model_info' not in st.session_state:
        st.session_state.model_info = {}


# ============================================
# MODEL LOADING (Hugging Face Integrated)
# ============================================
from huggingface_hub import hf_hub_download

@st.cache_resource(show_spinner=False)
def load_chatbot_model(model_name='xlarge'):
    """Load the Transformer chatbot model.
    Downloads from Hugging Face Hub if not found locally.
    
    Args:
        model_name: 'baseline', 'large', or 'xlarge'
    """
    try:
        from model_wrapper import TransformerChatbotInference

        # Map model names to Hugging Face filenames
        hf_repo = "AliMusaRizvi/urdu-chatbot-best-xlarge"  
        model_files = {
            'xlarge': 'best_xlarge_model.pt',
            'large': 'best_large_model.pt',
            'baseline': 'best_baseline_model.pt'
        }
        model_path = model_files.get(model_name, 'best_xlarge_model.pt')
        tokenizer_path = 'urdu_sp.model'

        # Local cache dir
        local_model_path = Path(model_path)

        # If not found locally, fetch from Hugging Face
        if not local_model_path.exists():
            st.warning(f"Downloading {model_name} model from Hugging Face...")
            token = st.secrets.get("HF_TOKEN", None)
            local_model_path = hf_hub_download(
                repo_id=hf_repo,
                filename=model_path,
                token=token
            )

        if not Path(tokenizer_path).exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

        # Load the chatbot
        chatbot = TransformerChatbotInference(model_path=local_model_path)

        # Get model info
        model_info = {
            'model_name': model_name,
            'model_path': str(local_model_path),
            'device': str(chatbot.device),
            'vocab_size': chatbot.tokenizer.get_vocab_size() if chatbot.tokenizer else 0,
            'config': chatbot.config.__dict__ if chatbot.config else {}
        }

        return chatbot, None, model_info

    except Exception as e:
        error_msg = f"Model loading failed: {str(e)}"
        print(f"Error: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        from model_wrapper import create_demo_chatbot
        return create_demo_chatbot(), error_msg, {}



def create_demo_chatbot():
    """Demo chatbot fallback"""
    class DemoChatbot:
        def __init__(self):
            self.device = torch.device('cpu')
            self.session_stats = {
                'total_conversations': 0,
                'avg_response_time': 0,
                'total_tokens_processed': 0
            }
            
        def chat(self, input_text, max_length=50, temperature=0.8, top_p=0.9):
            start_time = time.time()
            
            responses = {
                "سلام": "وعلیکم السلام! آپ کیسے ہیں",
                "آپ کیسے ہیں": "میں بہت اچھا ہوں، شکریہ! آپ کیسے ہیں؟",
                "شکریہ": "آپ کا خیر مقدم ہے!",
                "آج موسم کیسا ہے": "موسم بہت اچھا ہے، الحمدللہ!",
                "اللہ حافظ": "اللہ حافظ! خیال رکھیں!",
                "کیا حال ہے": "الحمدللہ بہت اچھا! آپ سنائیں",
                "مدد": "میں آپ کی کیسے مدد کر سکتا ہوں؟"
            }
            
            time.sleep(0.3)
            
            input_lower = input_text.strip()
            response = "معذرت، میں سمجھ نہیں پایا، کیا آپ دوبارہ بتا سکتے ہیں؟"
            
            for key, value in responses.items():
                if key in input_lower:
                    response = value
                    break
            
            response_time = time.time() - start_time
            self._update_stats(input_text, response_time)
            return response, response_time
        
        def _update_stats(self, input_text, response_time):
            self.session_stats['total_conversations'] += 1
            self.session_stats['total_tokens_processed'] += len(input_text.split())
            if self.session_stats['avg_response_time'] == 0:
                self.session_stats['avg_response_time'] = response_time
            else:
                self.session_stats['avg_response_time'] = (
                    self.session_stats['avg_response_time'] + response_time
                ) / 2
        
        def get_stats(self):
            return self.session_stats.copy()
    
    return DemoChatbot()


# ============================================
# UI COMPONENTS
# ============================================

def display_header():
    """Display header"""
    st.markdown("""
    <div class="main-container">
        <h1 class="header-title">🤖 Urdu AI Chatbot</h1>
        <p style="text-align: center; color: var(--text-muted); font-size: 0.9rem;">
            Transformer-based Neural Conversation System
        </p>
    </div>
    """, unsafe_allow_html=True)


def display_sidebar():
    """Display sidebar"""
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Model selection
        model_options = {
            'Extra Large (Best)': 'xlarge',
            'Large': 'large',
            'Baseline': 'baseline'
        }
        
        selected_model_name = st.selectbox(
            "Select Model:",
            options=list(model_options.keys()),
            index=0,
            help="Choose model size. XLarge has best quality but slower."
        )
        
        selected_model = model_options[selected_model_name]
        
        # Generation parameters
        st.markdown("---")
        st.markdown("### 🎛️ Generation Settings")
        
        temperature = st.slider(
            "Temperature:",
            min_value=0.1,
            max_value=1.5,
            value=0.8,
            step=0.1,
            help="Higher = more creative, Lower = more focused"
        )
        
        top_p = st.slider(
            "Top-p (Nucleus):",
            min_value=0.1,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="Nucleus sampling threshold"
        )
        
        max_length = st.slider(
            "Max Length:",
            min_value=20,
            max_value=100,
            value=50,
            step=5,
            help="Maximum response length"
        )
        
        st.markdown("---")
        st.markdown("### 📚 Chat History")
        
        if st.session_state.messages:
            recent_count = min(5, len(st.session_state.messages) // 2)
            for i in range(recent_count):
                idx = i * 2
                if idx < len(st.session_state.messages):
                    user_msg = st.session_state.messages[idx]['content']
                    with st.expander(f"#{i+1}", expanded=False):
                        st.write(f"**You:** {user_msg[:40]}...")
                        if idx + 1 < len(st.session_state.messages):
                            bot_msg = st.session_state.messages[idx + 1]['content']
                            st.write(f"**Bot:** {bot_msg[:40]}...")
        else:
            st.info("No conversations yet")
        
        st.markdown("---")
        st.markdown("### 🎮 Controls")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("🔄 Reload Model", use_container_width=True):
            st.session_state.model_loaded = False
            st.session_state.chatbot = None
            st.cache_resource.clear()
            st.rerun()
        
        if st.session_state.messages:
            st.markdown("---")
            if st.button("📄 Export Chat", use_container_width=True):
                export_chat()
        
        # Model info
        if st.session_state.model_info:
            st.markdown("---")
            st.markdown("### ℹ️ Model Info")
            info = st.session_state.model_info
            st.text(f"Model: {info.get('model_name', 'N/A')}")
            st.text(f"Device: {info.get('device', 'N/A')}")
            st.text(f"Vocab: {info.get('vocab_size', 0):,}")
            
            if 'config' in info and info['config']:
                config = info['config']
                st.text(f"d_model: {config.get('d_model', 'N/A')}")
                st.text(f"Layers: {config.get('n_encoder_layers', 'N/A')}")
        
        # Stats
        if st.session_state.chatbot:
            stats = st.session_state.chatbot.get_stats()
            if stats['total_conversations'] > 0:
                st.markdown("---")
                st.markdown("### 📊 Session Stats")
                st.metric("Conversations", stats['total_conversations'])
                st.metric("Avg Response Time", f"{stats['avg_response_time']:.2f}s")
        
        return temperature, top_p, max_length, selected_model


def export_chat():
    """Export chat history"""
    if not st.session_state.messages:
        st.warning("No messages to export!")
        return
    
    export_text = f"# Urdu AI Chatbot - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for i, message in enumerate(st.session_state.messages, 1):
        if message["role"] == "user":
            export_text += f"**Input {i // 2 + 1}:** {message['content']}\n"
        else:
            export_text += f"**Response:** {message['content']}\n"
            export_text += f"*Time: {message.get('response_time', 'N/A')}*\n\n"
    
    st.download_button(
        label="📥 Download",
        data=export_text,
        file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )


def display_chat():
    """Display chat messages"""
    if st.session_state.messages:
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    {message["content"]}
                    <div class="message-time">{message.get("time", "")}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                response_time = message.get("response_time", "N/A")
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    {message["content"]}
                    <div class="message-time">
                        {message.get("time", "")} • {response_time}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)


def display_input():
    """Display input form"""
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "اپنا پیغام لکھیں:",
            placeholder="یہاں اپنا پیغام لکھیں... (Type your message here...)",
            height=100,
            help="Enter your message in Urdu",
            key="user_input"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            submit = st.form_submit_button("🚀 Send", type="primary", use_container_width=True)
        with col2:
            clear = st.form_submit_button("Clear", use_container_width=True)
        with col3:
            demo_clicked = st.form_submit_button("Demo", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Word count
    if user_input:
        word_count = len(user_input.split())
        st.markdown(f'<div class="word-count">Words: {word_count} | Characters: {len(user_input)}</div>',
                    unsafe_allow_html=True)
    
    if clear:
        st.session_state.messages = []
        st.rerun()
        return "", False, False
    
    if demo_clicked:
        demo_texts = [
            "سلام علیکم",
            "آپ کیسے ہیں؟",
            "شکریہ",
            "آج موسم کیسا ہے؟",
            "کیا حال ہے؟",
            "مدد کریں"
        ]
        import random
        return random.choice(demo_texts), True, True
    
    return user_input, submit, False


def process_message(user_input, temperature, top_p, max_length):
    """Process user message"""
    if not user_input.strip():
        st.warning("Please enter a message.")
        return
    
    if not st.session_state.model_loaded or not st.session_state.chatbot:
        st.error("Model not loaded. Please wait for initialization.")
        return
    
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input.strip(),
        "time": datetime.now().strftime("%H:%M:%S")
    })
    
    # Generate response
    with st.spinner("Generating response..."):
        try:
            response, response_time = st.session_state.chatbot.chat(
                user_input.strip(),
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
            
            if response.startswith("Error:"):
                st.error(f"Response failed: {response}")
                return
            
            # Add bot message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "time": datetime.now().strftime("%H:%M:%S"),
                "response_time": f"{response_time:.2f}s"
            })
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            print(f"Error processing message: {e}")
            print(traceback.format_exc())
            return
    
    st.rerun()


def display_loading():
    """Display loading screen"""
    st.markdown("""
    <div style="
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 60vh;
        text-align: center;
        padding: 2rem;
    ">
        <div class="loading-animation" style="margin-bottom: 2rem;"></div>
        <h3 style="
            color: var(--text-secondary);
            font-weight: 600;
            margin: 0;
            background: var(--primary-gradient);
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        ">
            Initializing Urdu AI Chatbot
        </h3>
        <p style="
            color: var(--text-muted);
            font-size: 0.9rem;
            margin-top: 1rem;
            opacity: 0.8;
        ">
            Loading Transformer model...
        </p>
    </div>
    """, unsafe_allow_html=True)


def display_error_fallback(error_msg):
    """Display error fallback"""
    st.markdown(f"""
    <div class="main-container" style="text-align: center; padding: 3rem 2rem;">
        <h3 style="color: #ffb74d; margin: 1rem 0;">
            ⚠️ Running in Demo Mode
        </h3>
        <div style="
            margin: 1.5rem 0;
            padding: 1.5rem;
            background: rgba(255, 183, 77, 0.1);
            border: 1px solid #ffb74d;
            border-radius: 12px;
            color: var(--text-secondary);
            line-height: 1.6;
        ">
            <p style="margin: 0;">
                Neural model unavailable. Using demo responses for testing.
            </p>
        </div>
        <details style="margin-top: 1rem; color: var(--text-muted); font-size: 0.85rem;">
            <summary style="cursor: pointer;">Show error details</summary>
            <pre style="text-align: left; margin-top: 0.5rem; overflow-x: auto;">{error_msg}</pre>
        </details>
    </div>
    """, unsafe_allow_html=True)


# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main application"""
    init_session_state()
    
    # Load model on first visit
    if not st.session_state.model_loaded and not st.session_state.model_loading:
        display_loading()
        st.session_state.model_loading = True
        
        with st.spinner(""):
            chatbot, error, model_info = load_chatbot_model('xlarge')
            st.session_state.chatbot = chatbot
            st.session_state.model_loaded = True
            st.session_state.model_loading = False
            st.session_state.error_state = error
            st.session_state.model_info = model_info
        
        st.rerun()
    
    # Display header
    display_header()
    
    # Show error if model failed
    if st.session_state.error_state:
        display_error_fallback(st.session_state.error_state)

    # Sidebar controls
    temperature, top_p, max_length, selected_model = display_sidebar()

    # Chat area
    display_chat()

    # Input area
    user_input, submitted, is_demo = display_input()

    # Handle message submission
    if submitted and user_input.strip():
        process_message(user_input, temperature, top_p, max_length)

    elif is_demo and user_input.strip():
        process_message(user_input, temperature, top_p, max_length)


# ============================================
# RUN APPLICATION
# ============================================

if __name__ == "__main__":
    main()
