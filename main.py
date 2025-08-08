import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable or Streamlit secrets
def get_api_key():
    # Try to get from Streamlit secrets first (for Streamlit Cloud)
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("✅ API key loaded from Streamlit secrets")
        return api_key
    except Exception as e:
        st.info(f"Streamlit secrets not available: {str(e)}")
        # Fall back to environment variable
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            st.success("✅ API key loaded from environment variable")
        else:
            st.error("❌ API key not found in environment variables")
        return api_key

# Initialize Gemini Model
def create_model():
    api_key = get_api_key()
    if not api_key:
        st.error("Google API key not found. Please set GOOGLE_API_KEY in your environment variables or Streamlit secrets.")
        return None
    
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest", 
            google_api_key=api_key
        )
    except Exception as e:
        st.error(f"Error initializing Gemini model: {str(e)}")
        return None

# Prompt Template
tweet_template = """Generate {number} engaging tweets about {topic}. 
Make sure each tweet is:
- Under 280 characters
- Engaging and informative
- Use relevant hashtags when appropriate
- Vary in tone and style

Format each tweet on a new line with a number."""

tweet_prompt = PromptTemplate(
    input_variables=["number", "topic"],
    template=tweet_template
)

# Streamlit UI
st.header("Tweet Generator")
st.subheader("Generate tweets about your favorite topic")

# Initialize model
gemini_model = create_model()

if gemini_model:
    # LangChain Chain
    tweet_chain = tweet_prompt | gemini_model
    
    topic = st.text_input("Enter a topic", placeholder="e.g., artificial intelligence, climate change, cooking")
    number_of_tweets = st.number_input("Enter the number of tweets to generate", min_value=1, max_value=10, value=1)

    if st.button("Generate Tweets"):
        if not topic.strip():
            st.error("Please enter a topic!")
        else:
            try:
                with st.spinner("Generating tweets..."):
                    response = tweet_chain.invoke({"number": number_of_tweets, "topic": topic.strip()})
                    
                if hasattr(response, "content"):
                    st.success("Tweets generated successfully!")
                    st.write(response.content)
                else:
                    st.write(response)
                    
            except Exception as e:
                st.error(f"Error generating tweets: {str(e)}")
                st.info("Please check your API key and internet connection.")
else:
    st.error("Model initialization failed. Please check your API key configuration.")


