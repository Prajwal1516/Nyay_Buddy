import streamlit as st
import torch
from transformers import AutoTokenizer, BertForTokenClassification
import pandas as pd
import numpy as np
import plotly.express as px
from collections import Counter
from io import BytesIO
import re
import requests
import time
import pypdf
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Nyay Buddy",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main-header {
        font-size: 3rem; font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem; color: #2d3748; margin-top: 2rem; margin-bottom: 1rem;
        font-weight: 600; border-left: 4px solid #667eea; padding-left: 1rem;
    }
    .entity-highlight {
        padding: 0.3em 0.5em; border-radius: 8px; color: black !important;
        font-weight: 600; margin: 0 2px; display: inline-block;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .chat-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 20px; padding: 1.5rem; margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .chat-message { padding: 1rem; margin: 0.5rem 0; border-radius: 15px; }
    .user-message { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; margin-left: 20%; }
    .assistant-message { background: #ffffff; color: #2d3748; margin-right: 20%; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
    .metric-card {
        background: white; padding: 1.5rem; border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1); border-left: 4px solid #667eea; margin: 1rem 0;
    }
    .sidebar-entity {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        padding: 0.5em; border-radius: 10px; margin-bottom: 0.5em;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
</style>
""", unsafe_allow_html=True)

import os

# Try to get API key from secrets or environment variables
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

id2label = {
    "0": "O", "1": "B-CASE_NUMBER", "14": "I-CASE_NUMBER",
    "2": "B-COURT", "15": "I-COURT", "3": "B-DATE", "16": "I-DATE",
    "4": "B-GPE", "17": "I-GPE", "5": "B-JUDGE", "18": "I-JUDGE",
    "6": "B-ORG", "19": "I-ORG", "7": "B-OTHER_PERSON", "20": "I-OTHER_PERSON",
    "8": "B-PETITIONER", "21": "I-PETITIONER", "9": "B-PRECEDENT", "22": "I-PRECEDENT",
    "10": "B-PROVISION", "23": "I-PROVISION", "11": "B-RESPONDENT", "24": "I-RESPONDENT",
    "12": "B-STATUTE", "25": "I-STATUTE", "13": "B-WITNESS", "26": "I-WITNESS"
}

entity_types = sorted(list(set([label.split('-')[-1] for label in id2label.values() if label != "O"])))
color_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43', '#EE5A24', '#0FB9B1', '#3742FA', '#2F3542']
color_map = {entity: color_palette[i % len(color_palette)] for i, entity in enumerate(entity_types)}

@st.cache_resource
def load_model():
    try:
        import os
        model_path = os.path.join(os.path.dirname(__file__), 'legal-bert-ner')
        if not os.path.exists(model_path):
            return None, None
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = BertForTokenClassification.from_pretrained(model_path)
        return model, tokenizer
    except:
        return None, None

def predict(text, model, tokenizer):
    if not text.strip():
        return [], [], []
    if model is None or tokenizer is None:
        return demo_predict(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
    predicted_labels = [id2label[str(prediction.item())] for prediction in predictions[0]]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    scores = torch.nn.functional.softmax(outputs.logits, dim=2)
    confidences = [scores[0, i, predictions[0][i]].item() for i in range(len(predictions[0]))]
    return tokens, predicted_labels, confidences

def demo_predict(text):
    words = text.split()
    tokens = ['[CLS]'] + words + ['[SEP]']
    labels = ['O'] * len(tokens)
    confidences = [0.9] * len(tokens)
    for i, word in enumerate(words):
        idx = i + 1
        if re.search(r'\d{4}', word):
            labels[idx] = 'B-DATE'
        elif word.lower() in ['court', 'supreme', 'high']:
            labels[idx] = 'B-COURT'
        elif word.lower() in ['judge', 'justice']:
            labels[idx] = 'B-JUDGE'
    return tokens, labels, confidences

def get_entity_spans(tokens, labels, confidences, tokenizer=None):
    entity_spans = []
    current_entity = None
    current_tokens = []
    current_confidences = []
    for token, label, confidence in zip(tokens, labels, confidences):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            if current_entity:
                avg_conf = sum(current_confidences) / len(current_confidences) if current_confidences else 0
                text = ' '.join(current_tokens) if tokenizer is None else tokenizer.convert_tokens_to_string(current_tokens)
                entity_spans.append({"entity_type": current_entity, "text": text.replace(' ##', ''), "confidence": avg_conf})
                current_entity = None
                current_tokens = []
                current_confidences = []
            continue
        if label.startswith("B-"):
            if current_entity:
                avg_conf = sum(current_confidences) / len(current_confidences) if current_confidences else 0
                text = ' '.join(current_tokens) if tokenizer is None else tokenizer.convert_tokens_to_string(current_tokens)
                entity_spans.append({"entity_type": current_entity, "text": text.replace(' ##', ''), "confidence": avg_conf})
            current_entity = label[2:]
            current_tokens = [token]
            current_confidences = [confidence]
        elif label.startswith("I-") and current_entity and label[2:] == current_entity:
            current_tokens.append(token)
            current_confidences.append(confidence)
        elif label == "O":
            if current_entity:
                avg_conf = sum(current_confidences) / len(current_confidences) if current_confidences else 0
                text = ' '.join(current_tokens) if tokenizer is None else tokenizer.convert_tokens_to_string(current_tokens)
                entity_spans.append({"entity_type": current_entity, "text": text.replace(' ##', ''), "confidence": avg_conf})
                current_entity = None
                current_tokens = []
                current_confidences = []
    if current_entity:
        avg_conf = sum(current_confidences) / len(current_confidences) if current_confidences else 0
        text = ' '.join(current_tokens) if tokenizer is None else tokenizer.convert_tokens_to_string(current_tokens)
        entity_spans.append({"entity_type": current_entity, "text": text.replace(' ##', ''), "confidence": avg_conf})
    return entity_spans

def highlight_entities_improved(text, entities):
    if not entities:
        return text
    highlights = []
    for entity in entities:
        entity_text = entity["text"].strip()
        start = 0
        while True:
            pos = text.lower().find(entity_text.lower(), start)
            if pos == -1:
                break
            if (pos == 0 or not text[pos-1].isalnum()) and (pos + len(entity_text) == len(text) or not text[pos + len(entity_text)].isalnum()):
                highlights.append({'start': pos, 'end': pos + len(entity_text), 'entity_type': entity["entity_type"], 'text': entity_text, 'confidence': entity["confidence"]})
            start = pos + 1
    highlights.sort(key=lambda x: x['start'], reverse=True)
    filtered_highlights = []
    for highlight in highlights:
        overlap = False
        for existing in filtered_highlights:
            if not (highlight['end'] <= existing['start'] or highlight['start'] >= existing['end']):
                overlap = True
                break
        if not overlap:
            filtered_highlights.append(highlight)
    result_text = text
    for h in filtered_highlights:
        color = color_map.get(h['entity_type'], "#E2E8F0")
        html = f'''<span class="entity-highlight" style="background-color: {color};" title="Type: {h['entity_type']} | Confidence: {h['confidence']:.2f}">{h["text"]}</span>'''
        result_text = result_text[:h['start']] + html + result_text[h['end']:]
    return result_text

def generate_enhanced_charts(entities):
    if not entities:
        return None, None
    entity_counts = Counter([entity["entity_type"] for entity in entities])
    df_counts = pd.DataFrame({"Entity Type": list(entity_counts.keys()), "Count": list(entity_counts.values())}).sort_values("Count", ascending=False)
    fig1 = px.bar(df_counts, x="Entity Type", y="Count", color="Entity Type", color_discrete_map={e: color_map.get(e, "gray") for e in df_counts["Entity Type"]}, title="📊 Entity Type Distribution")
    fig1.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    df_conf = pd.DataFrame([{"Entity Type": e["entity_type"], "Confidence": e["confidence"]} for e in entities])
    fig2 = px.box(df_conf, x="Entity Type", y="Confidence", color="Entity Type", color_discrete_map={e: color_map.get(e, "gray") for e in df_conf["Entity Type"].unique()}, title="📈 Confidence Scores")
    fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig1, fig2

def chat_with_groq(messages):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-8b-8192",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1024
    }
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error connecting to GROQ API: {str(e)}"

def mock_ai_response(user_message, entities_context):
    """Mock AI response when GROQ API is not available."""
    responses = {
        "what entities were found": f"I found {len(entities_context)} entities in the analyzed text.",
        "explain the case": "Based on the entities identified, this appears to be a legal case document.",
        "summarize": "This legal document contains various entities including case numbers, court names, dates, and legal references."
    }
    for key, response in responses.items():
        if key.lower() in user_message.lower():
            return response
    return "I'm here to help you understand the legal entities found in your document."

if __name__ == "__main__":
    st.markdown('<h1 class="main-header">⚖️ Legal NER Analysis Tool</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'entities' not in st.session_state:
        st.session_state.entities = []
    if 'analyzed_text' not in st.session_state:
        st.session_state.analyzed_text = ""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Entity Types")
        for entity in entity_types:
            color = color_map.get(entity, "gray")
            st.markdown(f'<div class="sidebar-entity" style="border-left: 4px solid {color};">{entity}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    
    # Load model
    model, tokenizer = load_model()
    
    # Main content
    st.markdown('<h2 class="sub-header">📝 Input Legal Text</h2>', unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("📂 Upload a legal document (PDF/TXT)", type=['pdf', 'txt'])
    
    initial_text = ""
    if uploaded_file is not None:
        try:
            if uploaded_file.type == "application/pdf":
                pdf_reader = pypdf.PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                initial_text = text
                st.success(f"✅ Loaded {uploaded_file.name}")
            else:
                stringio = BytesIO(uploaded_file.getvalue())
                initial_text = stringio.read().decode("utf-8")
                st.success(f"✅ Loaded {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    
    user_text = st.text_area("Enter legal text here:", value=initial_text, height=300, key="input_text")
    
    if st.button("🔍 Analyze Text", type="primary"):
        if user_text:
            with st.spinner("🔄 Analyzing text..."):
                time.sleep(1)
                tokens, predicted_labels, confidences = predict(user_text, model, tokenizer)
                entity_spans = get_entity_spans(tokens, predicted_labels, confidences, tokenizer)
                filtered_entities = [entity for entity in entity_spans if entity["confidence"] >= confidence_threshold]
                st.session_state.entities = filtered_entities
                st.session_state.analyzed_text = user_text
                st.success(f"✅ Analysis complete! Found {len(filtered_entities)} entities.")
    
    # Display results
    if st.session_state.entities:
        st.markdown('<h2 class="sub-header">📊 Analysis Results</h2>', unsafe_allow_html=True)
        highlighted_text = highlight_entities_improved(st.session_state.analyzed_text, st.session_state.entities)
        st.markdown(f'<div style="background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); line-height: 1.8;">{highlighted_text}</div>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["📋 Entities Table", "📊 Charts"])
        
        with tab1:
            df_entities = pd.DataFrame([{"Entity Type": e["entity_type"], "Text": e["text"], "Confidence": e["confidence"]} for e in st.session_state.entities])
            st.dataframe(df_entities, use_container_width=True)
        
        with tab2:
            fig1, fig2 = generate_enhanced_charts(st.session_state.entities)
            if fig1:
                st.plotly_chart(fig1, use_container_width=True)
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)
    
    # AI Assistant Section
    st.markdown("---")
    st.markdown('<h2 class="sub-header">🤖 Legal Assistant</h2>', unsafe_allow_html=True)
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f'<div class="chat-message user-message">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant-message">{message["content"]}</div>', unsafe_allow_html=True)
    
    # Input area
    with st.form(key="chat_form"):
        user_input = st.text_input("Ask about the analyzed document:", key="user_query")
        submit_button = st.form_submit_button("Send 🚀")
        
    if submit_button and user_input:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get AI response
        context = f"Document Text: {st.session_state.analyzed_text[:2000]}..." if st.session_state.analyzed_text else "No document analyzed yet."
        entities_context = str([e['text'] for e in st.session_state.entities]) if st.session_state.entities else "No entities found."
        
        system_prompt = f"You are a legal assistant. Answer based on: {context}. Identified entities: {entities_context}."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        if GROQ_API_KEY:
            response = chat_with_groq(messages)
        else:
            response = mock_ai_response(user_input, st.session_state.entities)
            response += " (Note: Running in offline mode)"
            
        # Add assistant message
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()