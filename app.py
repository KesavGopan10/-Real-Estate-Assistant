import streamlit as st
from PIL import Image
import numpy as np
import datetime
import os
import json
from typing import Optional, Tuple
import asyncio
import nest_asyncio
from streamlit.runtime.scriptrunner import add_script_run_ctx
import google.generativeai as genai
import base64
from io import BytesIO

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Load environment variables
def load_env_file():
    """Load environment variables from .env file if it exists"""
    env_path = ".env"
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value
    return os.environ.get("GOOGLE_API_KEY", "")

class GeminiService:
    """
    Class for generating responses using Gemini API with vision support.
    """

    def __init__(self, api_key: str):
        """Initialize Gemini client with API key"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-pro")  # Supports text + images

    async def generate_response(self, prompt: str, image_data: Optional[np.ndarray] = None) -> str:
        """
        Generate response using Gemini API.

        Args:
            prompt (str): The prompt to send to the AI
            image_data (Optional[np.ndarray]): Image data if available

        Returns:
            str: AI response
        """
        try:
            # Prepare content for Gemini
            content = [{"text": prompt}]
            
            if image_data is not None:
                # Convert numpy array to PIL Image
                image = Image.fromarray(image_data)
                
                # Convert PIL Image to base64-encoded string
                buffered = BytesIO()
                image_format = image.format if image.format else "JPEG"
                image.save(buffered, format=image_format)
                img_bytes = buffered.getvalue()
                img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                
                # Determine MIME type
                mime_type = f"image/{image_format.lower()}"
                
                # Add image part to content
                content.append({
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": img_base64
                    }
                })

            # Generate content using Gemini
            response = await asyncio.to_thread(
                self.model.generate_content,
                content,
                generation_config={
                    "temperature": 0.7,
                }
            )

            return response.text

        except Exception as e:
            raise RuntimeError(f"Error with Gemini API request: {e}")

# Async helper for generating responses
async def generate_agent_response(gemini_service: GeminiService, agent_type: str,
                                 text_input: str, image_data: Optional[np.ndarray] = None,
                                 location: Optional[str] = None) -> Tuple[str, bool]:
    """
    Generate a response using the appropriate agent via the Gemini service.

    Args:
        gemini_service (GeminiService): The Gemini service to use
        agent_type (str): Type of agent ("property_issue" or "tenancy_faq")
        text_input (str): User's text input
        image_data (Optional[np.ndarray]): Image data if available
        location (Optional[str]): User's location if available

    Returns:
        Tuple[str, bool]: (Response, whether location is needed)
    """
    # Check for location requirement for tenancy questions
    if agent_type == "tenancy_faq":
        location_dependent_keywords = ["notice", "deposit", "evict", "legal", "right", "law", "return", "withhold"]
        needs_location = any(keyword in text_input.lower() for keyword in location_dependent_keywords)

        if needs_location and not location:
            return "To give you the most accurate information about tenancy laws, I need to know your location (city/state or country). Laws vary significantly by jurisdiction.", True

    # Create the appropriate prompt based on agent type
    if agent_type == "property_issue":
        prompt = f"""
Role:
You are a smart and helpful real estate diagnostics assistant that specializes in analyzing property-related issues using both images and text. Your main goal is to identify potential problems in a house or property from visual evidence and provide practical troubleshooting advice. You may ask follow-up questions to gather more details and improve your recommendations.

Capabilities:

Accepts user-uploaded property images and optional textual descriptions or questions.

Detects and identifies visible issues in images (e.g., water stains, mold, cracks, lighting problems, broken fixtures).

Gives actionable troubleshooting suggestions that a homeowner or real estate agent can follow.

Asks clear and helpful follow-up questions if the image is unclear or if additional context is needed to make a better diagnosis.

Response Format:

Begin with a diagnosis: What issue you see in the image and where it appears.

Provide a brief explanation of what might be causing the issue.

Offer at least one or two practical suggestions (e.g., who to contact, what product to use).

End with a follow-up question, if more information is needed.

Tone:
Friendly, informative, and supportive. Speak in plain language that a typical homeowner or real estate buyer/seller would understand.

Example Interaction:

User: “What’s wrong with this wall?” (uploads image)
Agent:
“It looks like there’s mold growth in the upper corner of the wall. This could be due to poor ventilation or a water leak from the ceiling.
I suggest inspecting for roof leaks and using an anti-mold treatment. A dehumidifier might also help reduce moisture.
Can you tell me if this room tends to feel damp or if you’ve noticed any recent water leaks?”



{'The user has provided an image of a property issue. Please analyze the image for visible issues.' if image_data is not None else ''}

User description: {text_input if text_input else 'No description provided.'}

"""
    elif agent_type == "tenancy_faq":
        prompt = f"""
Role:
You are a knowledgeable and reliable real estate legal assistant specializing in tenancy-related questions. You help users understand rental laws, lease agreements, tenant and landlord rights, and rental processes. Your answers should be informative, concise, and location-aware when the user provides geographic details.

Capabilities:

Answers frequently asked questions about tenancy laws, rental contracts, and tenant/landlord responsibilities.

Adjusts answers based on the user’s location (e.g., country, state, or city), if provided.

Gives clear, plain-language explanations suitable for non-lawyers.

Provides general guidance and encourages users to consult local authorities or legal professionals when needed.

Politely prompts for more details if the question is too broad or lacks location context.

Response Format:

Start with a direct and clear answer to the question.

Add context or conditions (e.g., "In most places..." or "This varies by location...").

Offer guidance on next steps (e.g., "You may want to contact a local tenancy board.").

Ask for the user's location if it helps tailor the response.

Tone:
Supportive, clear, and respectful. Speak like a helpful government agency rep or legal advisor who's good at explaining things simply.

Example Interaction:

User: “Can my landlord evict me without notice?”
Agent:
“In most places, a landlord cannot evict a tenant without proper notice. The exact notice period depends on your location and the reason for eviction.
For example, in many U.S. states, non-payment of rent requires at least 3–5 days’ notice.
Could you let me know what city or country you're in? I can provide more specific information based on local laws.”
User question: {text_input}
User location: {location if location else 'Not specified'}


"""
    else:  # unclear route
        prompt = f"""
You are a real estate assistant. The user has sent a message that could be about a property issue or a tenancy question.

User message: {text_input}

Please:
1. Ask if they're inquiring about a property maintenance issue or a tenancy/legal question
2. Briefly explain what kind of help you can provide for each category
3. Be concise and helpful
"""

    # Generate the response using the service
    response = await gemini_service.generate_response(prompt, image_data)

    # Return the response and whether location is needed
    return response, agent_type == "tenancy_faq" and needs_location and not location

# Streamlit app
async def main():
    # Set page configuration
    st.set_page_config(
        page_title="Real Estate Assistant",
        page_icon="🏠",
        layout="wide"
    )

    # Initialize session state variables if they don't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'user_location' not in st.session_state:
        st.session_state.user_location = None
    if 'current_agent' not in st.session_state:
        st.session_state.current_agent = None
    if 'waiting_for_location' not in st.session_state:
        st.session_state.waiting_for_location = False
    if 'api_key' not in st.session_state:
        st.session_state.api_key = load_env_file()

    # Title and description with colorful UI
    st.markdown(
        """
        <h1 style='color: #4CAF50;'>🏠 Real Estate Assistant</h1>
        <p style='color: #555;'>Get help with property issues or tenancy questions.</p>
        """,
        unsafe_allow_html=True
    )

    # Sidebar for settings and chat type selection
    with st.sidebar:
        st.markdown("<h2 style='color: #4CAF50;'>Settings</h2>", unsafe_allow_html=True)

        # API key input
        api_key = st.text_input(
            "Google API Key",
            type="password",
            value=st.session_state.api_key,
            help="Enter your Google API key or set it in .env file as GOOGLE_API_KEY"
        )

        # Save API key to session state when changed
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key

        st.markdown("<h2 style='color: #4CAF50;'>Choose Assistance Type</h2>", unsafe_allow_html=True)
        assistance_type = st.radio(
            "What do you need help with?",
            ["Let the assistant decide", "Property Issue", "Tenancy Question"]
        )

        st.markdown("---")
        st.markdown("### About this Assistant")
        st.markdown("""
        This assistant can help with:
        - 🔍 Property issue analysis (upload an image or describe the problem)
        - 📝 Tenancy questions and legal information

        Your chat history is only stored for this session.
        """)

    # Main chat interface
    st.markdown("<h2 style='color: #4CAF50;'>Chat</h2>", unsafe_allow_html=True)

    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
            if "image" in message and message["image"] is not None:
                st.chat_message("user").image(message["image"], caption="Uploaded Image")
        else:
            st.chat_message("assistant").write(message["content"])

    # Handle user input
    user_input = st.chat_input("Type your message here...")
    col1, col2 = st.columns([3, 1])

    with col1:
        uploaded_file = st.file_uploader("Upload an image of the property issue (optional)",
                                        type=["jpg", "jpeg", "png"])

    # Process the user input
    if user_input or uploaded_file:
        # Validate API key
        if not st.session_state.api_key:
            st.error("Please provide a Google API key in the sidebar or set it in .env file as GOOGLE_API_KEY.")
            return

        # Process uploaded image
        image_data = None
        if uploaded_file:
            # Convert uploaded image to a format we can use
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            image_data = image_array

            # Add to chat history (only store the fact that an image was uploaded)
            st.chat_message("user").write(user_input if user_input else "I have a property issue (image uploaded)")
            st.chat_message("user").image(image, caption="Uploaded Image")
            image_indicator = True
        else:
            image_indicator = False
            st.chat_message("user").write(user_input)

        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input if user_input else "I have a property issue (image uploaded)",
            "image": image_data if image_indicator else None,
            "timestamp": datetime.datetime.now()
        })

        # Create the Gemini service
        try:
            gemini_service = GeminiService(st.session_state.api_key)
        except Exception as e:
            st.error(f"Error initializing Gemini service: {e}")
            return

        # Use a loading indicator while waiting for AI response
        with st.spinner("Thinking..."):
            # If we were waiting for location, handle that specially
            if st.session_state.waiting_for_location:
                st.session_state.waiting_for_location = False
                st.session_state.user_location = user_input

                # Get previous question from history
                prev_question = st.session_state.chat_history[-2]["content"]

                # Generate response with location
                response, need_location = await generate_agent_response(
                    gemini_service,
                    "tenancy_faq",
                    prev_question,
                    None,
                    user_input
                )
                agent_type = "tenancy_faq"
            else:
                # Determine which agent to use based on user selection or auto-routing
                if assistance_type == "Property Issue":
                    agent_type = "property_issue"
                elif assistance_type == "Tenancy Question":
                    agent_type = "tenancy_faq"
                else:  # Let the assistant decide
                    # Simple routing logic
                    if uploaded_file is not None:
                        agent_type = "property_issue"
                    elif any(keyword in user_input.lower() for keyword in ["lease", "rent", "evict", "deposit", "notice", "legal"]):
                        agent_type = "tenancy_faq"
                    elif any(keyword in user_input.lower() for keyword in ["mold", "leak", "damage", "broken", "toilet", "water"]):
                        agent_type = "property_issue"
                    else:
                        agent_type = "unclear"

                # Generate response with the appropriate agent
                response, need_location = await generate_agent_response(
                    gemini_service,
                    agent_type,
                    user_input,
                    image_data,
                    st.session_state.user_location
                )

            # Update session state
            st.session_state.current_agent = agent_type
            if need_location:
                st.session_state.waiting_for_location = True

            # Display assistant response
            st.chat_message("assistant").write(response)

            # Add assistant response to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.datetime.now()
            })

# Create a new event loop and run the main function
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# Add the Streamlit script runner context to the loop
add_script_run_ctx(loop)

# Run the main function
if __name__ == "__main__":
    loop.run_until_complete(main())