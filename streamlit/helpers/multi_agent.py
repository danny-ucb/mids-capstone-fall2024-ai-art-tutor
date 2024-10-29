import getpass
import os
import openai
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.utilities import WikipediaAPIWrapper
from langchain.agents import AgentExecutor, create_openai_tools_agent
import functools
import operator
from typing import Sequence, TypedDict, List
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START
from typing import Annotated
from langchain.utilities import WikipediaAPIWrapper
from langchain.agents import load_tools
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display
from langchain.chat_models import ChatOpenAI
import getpass
import os
import streamlit as st
import base64
import urllib3
from openai import OpenAI
import random
import string
import requests



# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # human_feedback: Annotated[Sequence[HumanMessage], operator.add]
    #facts: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str

def agent_node(state, agent, name):
    #print(f"Config input: {config}")
    # create tagging for memory
    recall_str = (
        "<recall_memory>\n" + "\n".join(state["recall_memories"]) + "\n</recall_memory>"
    )
    result = agent.invoke({
        "messages": state["messages"],
        "recall_memories": recall_str,
    })
    return {"messages": [result]}


def create_agent(openai_key:str, 
                 llm: ChatOpenAI,  
                 tools: list, 
                 system_prompt: str):
    # Each worker node will be given a name and some tools.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ]
    )

    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, return_intermediate_steps=True, verbose=True)
    return executor

# Define agent nodes and human feedback nodes
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [AIMessage(content=result["output"], name=name)]}



wikipedia = WikipediaAPIWrapper()

@tool("wikipedia", return_direct=True)
def wikipedia_tool(query: str):
    '''Useful for when you need to look up a topic on wikipedia'''
    return wikipedia.run(query)

@tool("check_story_completion")
def check_story_completion(query: str):
    """Check if the story is complete based on the following criteria:
    1. Number of entities: There're at least 2 entities in the story. For example, caterpillars and a garden.
    2. Interactions: The entities interact with each other. For example, caterpillars eating leaves in the garden.
    3. User feedback: Ask if user is happy with the story, if not, ask for more details.
    """
    return ""
@tool("moderator_tool")
def moderator_tool(query:str):
    """
    Used for moderation throughout the app
    """
    return ""

class DalleInput(BaseModel):
    query: str = Field(description="should be a single prompt for image generation")

# @tool("generate_image", args_schema=DalleInput, return_direct=True)
# def generate_image(query: str):
#     '''Generate image based on query'''
#     return DallEAPIWrapper().run(query)

@tool("generate_image", args_schema=DalleInput, return_direct=True)
def generate_image(query: str):
    '''Generate image from query, in a style relatable to children 8-10 years old'''
    client = OpenAI()
    response = client.images.generate(
        model = "dall-e-2",
        prompt = query,
        size = "512x512",
        style = "vivid",
        n = 1
    )
    return response.data[0].url


def create_nodes(openai_key):
    
    """
    Create Supervisor
    """ 
    members = ["storyteller", "critic","visual_artist", "silly"]
    options = members

    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        " following workers:  {members}. Given the following user request,"
        " respond with the worker to act next."
        "Ask storyteller if user wants to build a storyline or find more inspiration"
        "Ask critic if user wants to improve their work, get feedback on it, or ask about specific art techniques."
        "Also ask critic if users want to know about art history or art theory. For example, color theory, or stories about artists"
        "Ask visual_artist if user specifically wants to visualize, don't ask visual_artist for text feedback."
        "Ask silly is a user is no longer asking about art or not making sense and saying silly phrases."
    )

    # Using openai function calling to make llm output more specific
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                }
            },
            "required": ["next"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), members=", ".join(members))

    gpt_4o_llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key = openai_key)
    gpt_35_turbo_llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key = openai_key)

    supervisor_chain = (
        prompt
        | gpt_4o_llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )



    """
    Create Other Agent Nodes
    """
    
    
    conversation_moderator = (
    """You are moderating and adjusting AI-generated content to ensure it is appropriate for children. 
    If the AI response is not suitable for children, rephrase, otherwise, keep it the same. 
    It should avoid complex language, sensitive topics  (e.g., violence, inappropriate language) 
    and be presented in a friendly, encouraging tone. If the content is inappropriate or too complex, 
    adjust it to be simpler and suitable for children. Maintain the same idea of the input text and 
    keep it about the same length. Only invoke once per AI response.
    """
    )

    conversation_moderator_agent = create_agent(
        openai_key, 
        gpt_4o_llm,
        tools=[moderator_tool],
        system_prompt=conversation_moderator
    )
    conversation_moderator_node = functools.partial(agent_node, agent=conversation_moderator_agent, name="conversation_moderator_agent")

    
    storyteller = create_agent(openai_key, gpt_4o_llm,[check_story_completion],"Talk in a teacher's tone to 8-10 years old.\
    You help user complete a storyline. Use check_story_completion tool to check completion\
    Only finish when complete\
        Otherwise keep building storyline with user.\
            Return 'story_complete' when story is complete. Otherwise return 'story_incomplete'")
    storyteller_node = functools.partial(agent_node, agent=storyteller, name="storyteller")

    # visual_artist
    visual_artist = create_agent(openai_key, gpt_4o_llm,[generate_image],"You're a visual artist \
        You draw in a style that is similar to children's drawings from age 8 to 10, \
            Make the style as similar as possible to user's original drawings\
            Your primary job is to help users visualize ideas\
            Input to artist_tool should be a single image description")
    visual_artist_node = functools.partial(agent_node, agent=visual_artist, name="visual_artist")

    # critic
    critic = create_agent(openai_key, gpt_4o_llm,[wikipedia_tool],"You give feedback on user's artwork and how to improve.\
        Talk in an encouraging teacher's tone to 8-10 years old, be consice for each user query \
            say no more than 3-4 sentences. Use wikipedia to look up information when users asked for \
                detailed explanation of art concepts or theories")
    critic_node = functools.partial(agent_node, agent=critic, name="critic")


    silly = create_agent(openai_key, gpt_4o_llm, [moderator_tool], "You gently redirect the user back to the focus of learning art. \
    If the child is getting off track, for example, saying silly phrases, repeating words, typing the alphabet, or \
    talking about something unrelated to art, remind them that you are an art teacher. Talk in one or two sentences.")

    silly_node = functools.partial(agent_node, agent = silly, name = "silly")
    
    multiagent = StateGraph(AgentState)

    # multiagent.add_node("image_moderator_node", image_moderator_node)
    multiagent.add_node("conversation_moderator_node", conversation_moderator_node)

    multiagent.add_node("supervisor", supervisor_chain)
    multiagent.add_node("visual_artist", visual_artist_node)
    multiagent.add_node("critic", critic_node)
    multiagent.add_node("storyteller", storyteller_node)
    multiagent.add_node("silly", silly_node)

    memory = MemorySaver()

    # Start conditions
    # multiagent.add_edge(START, "image_moderator_node")

    # multiagent.add_edge("image_moderator_node", "supervisor")
    multiagent.add_edge(START, "supervisor")


    # for supervisor to delegate
    conditional_map = {k: k for k in members} 
    multiagent.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

    multiagent.add_edge("storyteller", "conversation_moderator_node")
    multiagent.add_edge("critic","conversation_moderator_node")
    multiagent.add_edge("silly","conversation_moderator_node")

    # End conditions
    multiagent.add_edge("visual_artist", END)
    multiagent.add_edge("conversation_moderator_node", END)

    graph = multiagent.compile(checkpointer=memory)
    
    return graph 

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def generate_random_string(length):
    """Generates a random string of the specified length."""
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(length))


def download_image_requests(url, file_name):
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_name, 'wb') as file:
            file.write(response.content)
    else:
        pass

def stream_messages(graph, text: str, thread: dict, image_path: str= None):
    # Initialize the content with the text message
    content = [{"type": "text", "text": text}]

    # If image_url is provided, append the image content
    if image_path:
        base64_image = encode_image(image_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })

    # Define the input for the graph stream
    input_data = {
        "messages": [
            HumanMessage(content=content)
        ]
    }

        # Initialize a variable to store the final output message
    final_message = ""

    # Stream the graph and capture only the final message output
    for s in graph.stream(input_data, config=thread):
        if "__end__" not in s:
            # Capture the most recent message (final one)
            final_message = s
            # st.write(final_message)
            
    # Display just the final message
    if 'conversation_moderator_node' in final_message:
        final_message_str = final_message['conversation_moderator_node']['messages'][0].content
        # st.write(final_message_str)
    elif 'visual_artist' in final_message:
        img_url = s['visual_artist']['messages'][0].content
        
        # img_file_path = f"produced_images/AI_generated_image_{generate_random_string(10)}.png"
        # download_image_requests(url=img_url, file_name=img_file_path)
        # st.image(img_file_path, width = 300)        
        final_message_str = img_url
    else:
        # st.write("Warning! Wrong Node")
        final_message_str = final_message
    
    
    return(final_message)


def image_moderation(openai_key, image_path):
    """
    Moderates images to verify if they are children's drawings and appropriate for 8-10 year olds.
    Returns True only for appropriate children's drawings, False otherwise.
    
    Args:
        openai_key (str): OpenAI API key
        image_path (str): Path to the image file
        
    Returns:
        bool: True if image is an appropriate child's drawing, False otherwise
    """
    gpt_4o_llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_key)
    
    # More specific system prompt for better accuracy
    image_moderator = (
        "You are an expert at identifying children's artwork. Your task is to:"
        "\n1. Determine if the image is a drawing/sketch made by a child (ages 5-12)"
        "\n2. If it is a child's drawing, verify it contains no inappropriate content (violence, explicit themes, etc)"
        "\n3. Return 'True' ONLY if both conditions are met:"
        "\n   - The image is clearly a child's drawing/sketch"
        "\n   - The content is appropriate for children ages 8-10"
        "\n4. Return 'False' for:"
        "\n   - Any photographs"
        "\n   - Adult artwork"
        "\n   - Digital art"
        "\n   - Inappropriate children's drawings"
        "\nProvide only a one-word response: True or False"
    )
    
    image_moderator_agent = create_agent(
        openai_key,
        gpt_4o_llm,
        tools=[moderator_tool],
        system_prompt=image_moderator
    )
    
    base64_image = encode_image(image_path)
    
    # More detailed prompt for better context
    input_messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "Analyze this image and determine if it is: "
             "1) A drawing made by a child (not a photograph or adult artwork) AND "
             "2) Contains appropriate content for children ages 8-10. "
             "Answer True only if both conditions are met."},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                },
            },
        ]}
    ]
    
    try:
        response = image_moderator_agent.invoke({"messages": input_messages})
        result = response['output'].lower().strip() == 'true'
        return result
    except Exception as e:
        print(f"Error during image moderation: {str(e)}")
        return False



    