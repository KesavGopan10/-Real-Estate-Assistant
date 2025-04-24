import streamlit as st
from PIL import Image
import numpy as np
import datetime
import os
import json
from typing import Optional, Tuple, List, Dict, Any
import asyncio
from streamlit.runtime.scriptrunner import add_script_run_ctx
import google.generativeai as genai
import base64
from io import BytesIO
from dotenv import load_dotenv

# --- Environment Variable Loading ---
def load_env_file():
    load_dotenv()
    return os.environ.get("GOOGLE_API_KEY", "")

# --- Gemini Service ---
class GeminiService:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-pro")

    def _convert_st_history_to_gemini(self, chat_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        gemini_history = []
        for message in chat_history:
            role = "user" if message["role"] == "user" else "model"
            parts = []
            
            if message.get("content"):
                parts.append({"text": message["content"]})

            if message["role"] == "user" and message.get("image_data") is not None:
                image_data = message["image_data"]
                try:
                    image = Image.fromarray(image_data)
                    max_size = (1024, 1024)
                    image.thumbnail(max_size, Image.Resampling.LANCZOS)
                    buffered = BytesIO()
                    img_format = image.format if image.format else "JPEG"
                    image.save(buffered, format=img_format, quality=85)
                    img_bytes = buffered.getvalue()
                    if len(img_bytes) > 4 * 1024 * 1024:
                        raise ValueError("Image size too large for API")
                    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                    mime_type = f"image/{img_format.lower()}"
                    parts.append({
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": img_base64
                        }
                    })
                except Exception as e:
                    st.error(f"Error processing image: {e}")
                    parts.append({"text": "[Image processing failed]"})

            if parts:
                gemini_history.append({"role": role, "parts": parts})

        return gemini_history

    async def generate_response_with_history(self, system_prompt: str, chat_history: List[Dict[str, Any]]) -> str:
        try:
            gemini_history = self._convert_st_history_to_gemini(chat_history)
            chat = self.model.start_chat(history=gemini_history[:-1])
            last_user_message_parts = gemini_history[-1]['parts']
            response = await asyncio.to_thread(
                chat.send_message,
                last_user_message_parts,
                generation_config={"temperature": 0.7}
            )
            if not response.text:
                raise ValueError("Empty response from Gemini API")
            return response.text
        except Exception as e:
            st.warning(f"Error using chat history: {e}. Falling back to single-turn generation.")
            try:
                if not chat_history or not chat_history[-1].get("content"):
                    raise ValueError("No valid user input to process")
                last_message = chat_history[-1]
                prompt = system_prompt + "\n\n" + last_message.get("content", "")
                image_data = last_message.get("image_data")
                content = [{"text": prompt}]
                if image_data is not None:
                    image = Image.fromarray(image_data)
                    buffered = BytesIO()
                    img_format = image.format if image.format else "JPEG"
                    image.save(buffered, format=img_format, quality=85)
                    img_bytes = buffered.getvalue()
                    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                    mime_type = f"image/{img_format.lower()}"
                    content.append({
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": img_base64
                        }
                    })
                response = await asyncio.to_thread(
                    self.model.generate_content,
                    content,
                    generation_config={"temperature": 0.7}
                )
                if not response.text:
                    raise ValueError("Empty response from Gemini API")
                return response.text
            except Exception as fallback_e:
                raise RuntimeError(f"Error with Gemini API request: {fallback_e}")

# --- Agent Logic ---
async def generate_agent_response(gemini_service: GeminiService, agent_type: str,
                                 chat_history: List[Dict[str, Any]], location: Optional[str] = None) -> Tuple[str, bool]:
    last_message = chat_history[-1]
    text_input = last_message.get("content", "")
    has_image = last_message.get("image_data") is not None

    property_issue_prompt = """
Role: You are a smart real estate diagnostics assistant specializing in analyzing property issues using images and text. Use conversation history for context. Identify problems and provide practical troubleshooting advice. Ask follow-up questions if needed.
Capabilities:
- Analyze images and text for issues (e.g., water stains, mold, cracks).
- Provide actionable troubleshooting steps.
- Ask relevant follow-up questions.
Response Format:
1. Acknowledge input.
2. Diagnosis: State potential issue(s).
3. Explanation: Explain causes.
4. Suggestions: Offer 1-3 steps.
5. Follow-up Question: Ask for more info if needed.
Tone: Friendly, clear, concise.
"""

    tenancy_faq_prompt = f"""
Role: You are a real estate legal assistant specializing in tenancy questions. Use conversation history for context. Provide concise, location-aware answers (location: {location if location else 'Not specified'}). Ask for location if needed.
Capabilities:
- Answer tenancy FAQs (rent, leases, rights).
- Adapt to location if provided.
- Explain in plain language.
- Advise consulting local authorities for specifics.
Response Format:
1. Address the question.
2. Provide general or location-specific info.
3. Ask for location if needed.
4. Suggest next steps.
Tone: Supportive, clear, professional.
"""

    unclear_prompt = """
Role: You are a real estate assistant. The request is unclear.
Task: Ask for clarification.
1. Acknowledge message.
2. Ask if they need help with:
   a) Property issues (offer photo upload).
   b) Tenancy questions.
Tone: Concise, friendly.
"""

    needs_location = False
    if agent_type == "tenancy_faq":
        location_dependent_keywords = ["notice", "deposit", "evict", "legal", "right", "law", "return", "withhold", "enter", "increase rent", "break lease"]
        if any(keyword in text_input.lower() for keyword in location_dependent_keywords) and not location:
            needs_location = True
            tenancy_faq_prompt += "\n\nIMPORTANT: Since location is needed and not provided, ask for city/state or country."

    system_prompt = {
        "property_issue": property_issue_prompt,
        "tenancy_faq": tenancy_faq_prompt,
        "unclear": unclear_prompt
    }.get(agent_type, unclear_prompt)

    response = await gemini_service.generate_response_with_history(system_prompt, chat_history)
    return response, needs_location

# --- Streamlit App ---
async def main():
    st.set_page_config(page_title="Real Estate Assistant", page_icon="🏠", layout="wide")

    # Initialize Session State
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'user_location' not in st.session_state:
        st.session_state.user_location = None
    if 'current_agent' not in st.session_state:
        st.session_state.current_agent = "unclear"
    if 'waiting_for_location' not in st.session_state:
        st.session_state.waiting_for_location = False
    if 'api_key' not in st.session_state:
        st.session_state.api_key = load_env_file()
    if 'pending_image_data' not in st.session_state:
        st.session_state.pending_image_data = None
    if 'pending_image_name' not in st.session_state:
        st.session_state.pending_image_name = None

    # Limit chat history
    max_history_length = 20
    if len(st.session_state.chat_history) > max_history_length:
        st.session_state.chat_history = st.session_state.chat_history[-max_history_length:]

    # UI Elements
    st.markdown("<h1 style='color: #4CAF50;'>🏠 Real Estate Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #555;'>Your AI helper for property issues and tenancy questions.</p>", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("<h2 style='color: #4CAF50;'>Settings</h2>", unsafe_allow_html=True)
        api_key = st.text_input(
            "Google API Key", type="password", value=st.session_state.api_key,
            help="Enter your Google API key or set it in .env file as GOOGLE_API_KEY"
        )
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key

        st.markdown("<h2 style='color: #4CAF50;'>Assistant Mode</h2>", unsafe_allow_html=True)
        agent_options = ["Let Assistant Decide", "Property Issue", "Tenancy Question"]
        agent_map = {"property_issue": 1, "tenancy_faq": 2, "unclear": 0}
        default_index = agent_map.get(st.session_state.current_agent, 0)
        selected_agent_mode = st.radio(
            "Manually select assistance type:",
            agent_options,
            index=default_index
        )
        if selected_agent_mode == "Property Issue":
            st.session_state.current_agent = "property_issue"
        elif selected_agent_mode == "Tenancy Question":
            st.session_state.current_agent = "tenancy_faq"
        else:
            pass

        st.markdown("---")
        st.info("Your chat history provides context for follow-up questions.")
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.current_agent = "unclear"
            st.session_state.user_location = None
            st.session_state.waiting_for_location = False
            st.session_state.pending_image_data = None
            st.session_state.pending_image_name = None
            st.rerun()

    # Main Chat Area
    uploaded_file = st.file_uploader(
        "Upload Property Image (Optional - then describe below)",
        type=["jpg", "jpeg", "png"],
        key=f"file_uploader_{len(st.session_state.chat_history)}"
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            max_size = (1024, 1024)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            image_array = np.array(image)
            st.session_state.pending_image_data = image_array
            st.session_state.pending_image_name = uploaded_file.name
            st.success(f"🖼️ Image '{uploaded_file.name}' ready. Add description below and send.")
            st.image(image, caption=f"Ready to send: {uploaded_file.name}", width=300)
        except Exception as e:
            st.error(f"Error processing image: {e}")
            st.session_state.pending_image_data = None
            st.session_state.pending_image_name = None

    # Chat History
    st.markdown("<h2 style='color: #4CAF50;'>Chat</h2>", unsafe_allow_html=True)
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                if message.get("content"):
                    st.write(message["content"])
                if message.get("image_data") is not None:
                    try:
                        image = Image.fromarray(message["image_data"])
                        st.image(image, width=300, caption="Uploaded Image")
                    except Exception as e:
                        st.error(f"Error displaying image: {e}")

    # Chat Input
    user_input = st.chat_input(
        "Describe the issue or ask your question here...",
        disabled=not st.session_state.api_key
    )
    if not st.session_state.api_key and user_input is None:
        st.error("⚠️ Please provide a Google API key in the sidebar settings.")

    # Handle User Input
    if user_input:
        if not st.session_state.api_key:
            st.error("⚠️ Please provide a Google API key in the sidebar settings.")
            st.stop()

        current_message = {
            "role": "user",
            "content": user_input,
            "timestamp": datetime.datetime.now(),
            "image_data": st.session_state.pending_image_data
        }
        st.session_state.chat_history.append(current_message)
        st.session_state.pending_image_data = None
        st.session_state.pending_image_name = None

        try:
            gemini_service = GeminiService(st.session_state.api_key)
        except Exception as e:
            st.error(f"Error initializing Gemini service: {e}")
            st.stop()

        with st.spinner("Thinking..."):
            mode_from_sidebar = st.session_state.current_agent
            if mode_from_sidebar in ["property_issue", "tenancy_faq"]:
                agent_type_to_use = mode_from_sidebar
            elif st.session_state.current_agent in ["property_issue", "tenancy_faq"] and len(st.session_state.chat_history) > 1:
                agent_type_to_use = st.session_state.current_agent
            else:
                has_image = current_message.get("image_data") is not None
                text_content = current_message.get("content", "").lower()
                property_keywords = ["mold", "leak", "damage", "broken", "fix", "repair", "water", "crack", "pest", "paint"]
                tenancy_keywords = ["lease", "rent", "evict", "deposit", "notice", "legal", "tenant", "landlord", "agreement", "rights"]
                property_score = sum(1 for keyword in property_keywords if keyword in text_content) + (5 if has_image else 0)
                tenancy_score = sum(1 for keyword in tenancy_keywords if keyword in text_content)
                if property_score > tenancy_score:
                    agent_type_to_use = "property_issue"
                elif tenancy_score > 0:
                    agent_type_to_use = "tenancy_faq"
                else:
                    agent_type_to_use = "unclear"
            
            st.session_state.current_agent = agent_type_to_use

            if st.session_state.waiting_for_location:
                provided_location = user_input.strip()
                if any(keyword in provided_location.lower() for keyword in ["city", "state", "country", ",", " "]):
                    st.session_state.user_location = provided_location
                    st.session_state.waiting_for_location = False
                    st.info(f"Location set to: {provided_location}")
                else:
                    ai_response = "Please share your city/state or country for accurate tenancy advice, or type 'skip' to continue with general advice."
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": ai_response,
                        "timestamp": datetime.datetime.now()
                    })
                    st.rerun()

            try:
                ai_response, needs_location_next = await generate_agent_response(
                    gemini_service,
                    agent_type_to_use,
                    st.session_state.chat_history,
                    st.session_state.user_location
                )
                st.session_state.waiting_for_location = needs_location_next
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": ai_response,
                    "timestamp": datetime.datetime.now()
                })
            except Exception as e:
                st.error(f"Error generating response: {e}")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"Sorry, I encountered an error: {e}",
                    "timestamp": datetime.datetime.now()
                })

        st.rerun()

if __name__ == "__main__":
    asyncio.run(main())
