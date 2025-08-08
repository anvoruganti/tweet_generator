from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st


# Initialize Gemini Model
gemini_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest", 
    google_api_key=GOOGLE_API_KEY
)

# Prompt Template
tweet_template = "Give me {number} tweets on {topic}"
tweet_prompt = PromptTemplate(
    input_variables=["number", "topic"],
    template=tweet_template
)

# LangChain Chain
tweet_chain = tweet_prompt | gemini_model

# Streamlit UI
st.header("Tweet Generator")
st.subheader("Generate a tweet about your favorite topic")

topic = st.text_input("Enter a topic")
number_of_tweets = st.number_input("Enter the number of tweets to generate", min_value=1, max_value=10, value=1)

if st.button("Generate Tweet"):
    response = tweet_chain.invoke({"number": number_of_tweets, "topic": topic})
    st.write(response.content if hasattr(response, "content") else response)
