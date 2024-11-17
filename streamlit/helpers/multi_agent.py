#multi_agent.py
# Standard library imports
import base64
import functools
import operator
import os
import random
import string
from typing import Annotated, List, Sequence, TypedDict
import chromadb
from chromadb.utils import embedding_functions
import uuid
import datetime
import json
#Other Scripts
from helpers.general_helpers import *
from helpers.memory_utils import *
from helpers.image_helpers import * 
from helpers.api_keys import * 
from helpers.consent_utils import * 

# Third-party imports
import requests
import streamlit as st
import tiktoken
import urllib3
from IPython.display import Image, display
from openai import OpenAI

#Semantic Router
from semantic_router import Route
from semantic_router import RouteLayer
from semantic_router.encoders import OpenAIEncoder

# Langchain core imports
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, get_buffer_string
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.runnables import RunnableConfig
from langchain_core.runnables import RunnablePassthrough, RunnableSequence, RunnableConfig
from langchain_core.tools import BaseTool, tool
from langchain_core.vectorstores import InMemoryVectorStore

# Langchain specific imports
from langchain.agents import AgentExecutor, create_openai_tools_agent, load_tools
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool
from langchain.utilities import WikipediaAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings


# Langgraph imports
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode



# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    recall_memories: Annotated[Sequence[str], operator.add]
    # The 'next' field indicates where to route to next
    next: str
    is_appropriate: bool
    moderator_response: str


def create_agent(openai_key: str, 
                 llm: ChatOpenAI,  
                 tools: list, 
                 system_prompt: str):
    """Create an agent using create_openai_tools_agent that's compatible with LangGraph."""
    
    # Create the prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Create the agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)

    # Create a chain that processes both the messages and recall_memories
    chain = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: []
        )
        | agent_executor
        | (lambda x: {"messages": [AIMessage(content=x["output"])]})
    )

    return chain

def agent_node(state, agent, name):
    """Process messages with the agent."""
    # Only use the last 5 messages for each node
    recent_messages = state["messages"][-min(5, len(state["messages"])):]
    relevant_memories = state["recall_memories"]
    
    recall_str = (
        "<recall_memory>\n" + "\n".join(relevant_memories) + "\n</recall_memory>"
    )

    # Add recall memories to the last message if it's from the user
    if recent_messages and isinstance(recent_messages[-1], HumanMessage):
        last_msg_content = recent_messages[-1].content
        if isinstance(last_msg_content, str):
            recent_messages[-1] = HumanMessage(content=f"{last_msg_content}\n\n{recall_str}")
        elif isinstance(last_msg_content, list):
            # Handle multi-modal content
            text_content = next((item["text"] for item in last_msg_content if item.get("type") == "text"), "")
            updated_content = [
                *last_msg_content,
                {"type": "text", "text": f"\n\n{recall_str}"}
            ]
            recent_messages[-1] = HumanMessage(content=updated_content)

    try:
        result = agent.invoke({"messages": recent_messages})
        return result
    except Exception as e:
        return {"messages": [AIMessage(content=f"An error occurred: {str(e)}")]}
    
def get_username(config: RunnableConfig) -> str:
    """Get username from the config."""
    username = config["configurable"].get("username")
    if username is None:
        raise ValueError("Username needs to be provided to save a memory.")
    return username

# function for load_memories
tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

def load_memories(state: AgentState, config: RunnableConfig) -> AgentState:
    """Load memories for the current conversation.

    Args:
        state (schemas.State): The current state of the conversation.
        config (RunnableConfig): The runtime configuration for the agent.

    Returns:
        State: The updated state with loaded memories.
    """
    consent_settings = config["configurable"].get("consent_settings", {})
    
    # Check if memory collection is allowed
    if not consent_settings.get("memory_collection", False):
        return {
            "recall_memories": [],
        }    
    convo_str = get_buffer_string(state["messages"])
    convo_str = tokenizer.decode(tokenizer.encode(convo_str)[:2048])
    recall_memories = search_recall_memories.invoke(convo_str, config)
    return {
        "recall_memories": recall_memories,
    }


def route_tools(state: AgentState):
    """Determine whether to use tools or end the conversation based on the last message.

    Args:
        state (schemas.State): The current state of the conversation.

    Returns:
        Literal["tools", "__end__"]: The next step in the graph.
    """
    msg = state["messages"][-1]
    # if tool call is to save_recall_memory or search_recall_memories
    if hasattr(msg, "additional_kwargs"):
        tool_calls = msg.additional_kwargs.get("tool_calls")
        if tool_calls:  # This ensures `tool_calls` is present and non-empty
            return "tools"

    return END


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

@tool("generate_image", args_schema=DalleInput, return_direct=True)
def generate_image(query: str):
    '''Generate image from query, in a style relatable to children 8-10 years old'''
    client = OpenAI()
    response = client.images.generate(
        model = "dall-e-3",
        prompt = query,
        size = "1024x1024",
        n = 1
    )
    return response.data[0].url

@tool
def save_recall_memory(memory: str, config: RunnableConfig) -> str:
    """Save memory to vectorstore for later semantic retrieval."""
    username = get_username(config)
    consent_settings = config["configurable"].get("consent_settings", {})
    
    # Check if memory collection is allowed
    if not consent_settings.get("memory_collection", False):
        return memory
    
    collection = get_vector_store()

    collection.add(
        documents=[memory],
        metadatas=[{"username": username}],
        ids=[f"{username}_{str(uuid.uuid4())}"]
    )
    return memory


@tool
def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
    """Search for relevant memories."""
    username = get_username(config)
    consent_settings = config["configurable"].get("consent_settings", {})
    
    collection = get_vector_store()
    
    query_results = collection.query(
        query_texts=[query],
        n_results=15,
        where={"username": username}
    )
    if len(query_results["documents"][0]) == 0:
        return []
    return [doc[0] for doc in query_results["documents"]]

def format_to_openai_tool_messages(intermediate_steps):
    """Format intermediate steps into OpenAI tool messages."""
    messages = []
    for action, observation in intermediate_steps:
        messages.append({
            "tool_calls": [{
                "id": str(uuid.uuid4()),
                "type": "function",
                "function": {
                    "name": action.tool,
                    "arguments": json.dumps(action.tool_input)
                }
            }],
            "role": "assistant"
        })
        messages.append({
            "tool_call_id": messages[-1]["tool_calls"][0]["id"],
            "role": "tool",
            "name": action.tool,
            "content": str(observation)
        })
    return messages


def create_nodes(openai_key):

    
    storyteller_route = Route(
    name="storyteller_route",
    utterances = [
        "Help me complete the storyline",
        "Give me inspiration on what to draw",
        "I need help with my story and plot",
        "What else might be in the story?"
    ]
    )

    critic_route = Route(
    name="critic_route",
    utterances = [
        "Can you provide feedback on my recent artwork to help me improve?",
        "Could you critique the color I used and suggest improvements?",
        "What are relevant art history and theory?",
        "What are some art techniques I can use?",
    ]     
    )

    visual_artist_route = Route(
    name="visual_artist_route",
    utterances = [
        "Help me visualize the ideas",
        "Can you show me what it looks like?",
        "Can you draw a picture of it?",
        "Draw me a picture of the story"
    ]
    )
    
    routes = [storyteller_route, critic_route, visual_artist_route]
    rl = RouteLayer(encoder=OpenAIEncoder(), routes=routes)

    def semantic_layer(query: str, route_layer=rl) -> str:
        route = route_layer(query)
        if route.name == "storyteller_route":
            return "storyteller"
        elif route.name == "critic_route":
            return "critic"
        elif route.name == "visual_artist_route":
            return "visual_artist"
        else:
            return "silly"

    def router_node(state, name, route_layer=rl):
        """Router node that determines which agent should handle the request"""
        last_message = state["messages"][-1]
        message_text = last_message.content[0]["text"] if isinstance(last_message.content, list) else last_message.content
        result = semantic_layer(message_text)
        return {
            "messages": state["messages"],
            "recall_memories": state["recall_memories"],
            "next": result
        }
    

    memory_usage_prompt = ( "Memory Usage Guidelines:\n"
            "1. Actively use (search_recall_memories)"
            " to give personalized recommendation based past stated user preferences\n"
            "2. Never explicitly mention your memory capabilities."
            " Instead, seamlessly incorporate your understanding of the user"
            " into your responses.\n"
            "3. Cross-reference new information with existing memories for"
            " consistency.\n"     
            "4. Recognize and acknowledge changes in the user's situation or"
            " perspectives over time.\n"
            "## Recall Memories\n"
            "Recall memories are contextually retrieved based on the current"
            " conversation:\n\n")
    
    storyteller_prompt = ("Talk in a teacher's encouraging and engaging tone to 8-10 years old. \
        Use languages that are easy to understand for the age group.\
        You help users build a richer storyline and give them inspirations. \
            Actively use wikipedia_tool to show relevant art history or theory related to user queries)\
                Actively use memory tool (save_recall_memory)\
                to build a comprehensive understanding of the user")

    visual_artist_prompt = """You're a visual artist helping 8-10 year old children. 
            When asked to create or draw something, always use the generate_image tool with specific, child-friendly prompts.
            Keep your language simple and engaging.
            After generating an image, briefly explain what you created.
            Use memory tools to understand user preferences.
            
            Follow these steps when generating images:
            1. Add "in a children's illustration style" to your prompts
            2. Keep prompts age-appropriate and positive
            3. Use clear, descriptive language
            4. Avoid any scary or inappropriate content
            
            Remember to:
            - Only generate one image per request
            - Explain the image after it's created
            - Be encouraging and supportive"""
    
    critic_prompt = ("You give feedback on user's artwork and how to improve.\
        Talk in simple and engaging style to 8-10 years old, use languages that are easy to understand for the age group\
        Actively use wikipedia_tool to show relevant art history or theory related to user query\
                    Actively use memory tools (save_recall_memory)\
                to build a comprehensive understanding of the user.")


    llm = ChatOpenAI(model="gpt-4o-mini")

    
    # router
    semantic_router = functools.partial(router_node, name="semantic_router", route_layer=rl)
    

    # storyteller
    storyteller = create_agent(openai_key,
                               llm,
                               [wikipedia_tool,save_recall_memory,search_recall_memories],
                               storyteller_prompt+"\n"+memory_usage_prompt)
    
    storyteller_node = functools.partial(agent_node, agent=storyteller, name="storyteller")
    
    # visual_artist
    visual_artist = create_agent(openai_key, llm,[generate_image,save_recall_memory,search_recall_memories],visual_artist_prompt+"\n"+memory_usage_prompt)
    visual_artist_node = functools.partial(agent_node, agent=visual_artist, name="visual_artist")
    
    # critic
    critic = create_agent(openai_key, llm,[wikipedia_tool,save_recall_memory,search_recall_memories],critic_prompt+"\n"+memory_usage_prompt)
    critic_node = functools.partial(agent_node, agent=critic, name="critic")
    
    # silly agent
    silly = create_agent(openai_key, llm, [moderator_tool], "You gently redirect the user back to the focus of learning art. \
        If the child is getting off track, for example, saying silly phrases, repeating words, typing the alphabet, or \
        talking about something unrelated to art, remind them that you are an art teacher. Talk in one or two sentences.")
    
    silly_node = functools.partial(agent_node, agent = silly, name = "silly")


    moderator_json_schema = {
    "name": "content_moderator",
    "description": "Determines if the content is appropriate and provides a response for the user.",
    "parameters": {
        "title": "contentModeratorSchema",
        "type": "object",
        "properties": {
            "is_appropriate": {
                "title": "Is Appropriate",
                "type": "boolean",
                "description": "True if both image and text are appropriate; false otherwise."
            },
            "moderator_response": {
                "title": "Response",
                "type": "string",
                "description": "Guide users if content is inappropriate. Leave empty if content is appropriate."
            }
        },
        "required": ["is_appropriate", "moderator_response"]
    }
    }

    moderator_system_prompt = (
    "You are tasked with determining if the content, both images and conversation, is appropriate for children aged 8-10 years old."
    "User request to ask for an image or picture of what something looks like is totally appropriate."
    "Filter for the following criteria instead:\n\n "
    "First, if there's an image provided, check the image: appropriate content includes children's drawings that do not depict violence, explicit themes, "
    "or anything unsuitable for this age group. If the uploaded image is a photograph of something other than a child's drawing, it is not appropriate. "
    "Invoke moderator_tool only once.\n\n"
    
    "Next, check the conversation: you are moderating and adjusting AI-generated content to ensure it is suitable for children. "
    "If the AI response is not suitable for children, rephrase it; otherwise, keep it the same. The response should avoid complex language "
    "and sensitive topics (e.g., violence, inappropriate language) and be presented in a friendly, encouraging tone. If the content is inappropriate "
    "or too complex, adjust it to be simpler and suitable for children, maintaining the original idea and length. "
    "Only invoke once per AI response."
    )

    
    moderator_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", moderator_system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Is the content appropriate and why?"),
    ]
    )

    moderator = moderator_prompt | llm.with_structured_output(moderator_json_schema)

    tools = [generate_image, wikipedia_tool, save_recall_memory, search_recall_memories]

    multiagent = StateGraph(AgentState)
    multiagent.add_node("load_memories", load_memories)
    multiagent.add_node("semantic_router", semantic_router)
    multiagent.add_node("visual_artist", visual_artist_node)
    multiagent.add_node("critic", critic_node)
    multiagent.add_node("storyteller", storyteller_node)
    multiagent.add_node("tools", ToolNode(tools))
    
    # add other nodes
    multiagent.add_node("silly", silly_node)
    multiagent.add_node("moderator", moderator)

    members = ["storyteller", "visual_artist", "critic"]
    options = members + ["silly"]

    memory = MemorySaver()
    # Start conditions
    multiagent.add_edge(START, "moderator")
    # if inappropriate, end conversation and guide users 
    multiagent.add_conditional_edges("moderator", lambda x: x["is_appropriate"], {True:"load_memories", False:END})
    multiagent.add_edge("load_memories", "semantic_router")
    
    # for supervisor to delegate
    conditional_map = {k: k for k in options} 
    multiagent.add_conditional_edges("semantic_router", lambda x: x["next"], conditional_map)
    
    # for each agent to use memory tools or end converation
    multiagent.add_conditional_edges("storyteller", route_tools, ["tools",END])
    multiagent.add_conditional_edges("visual_artist", route_tools, ["tools",END])
    multiagent.add_conditional_edges("critic", route_tools, ["tools",END])
    multiagent.add_edge("silly", END)
    
    # tools need to report back to agent
    multiagent.add_conditional_edges("tools", lambda x: x["next"], 
        {
            "critic":"critic",
            "visual_artist":"visual_artist",
            "storyteller":"storyteller"
        }
    )
    
    # compile
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
def convert_to_langchain_messages(messages: List[Dict]) -> List[BaseMessage]:
    """Convert dictionary messages to LangChain BaseMessage objects."""
    converted_messages = []
    for msg in messages:
        if msg['role'] == 'user':
            converted_messages.append(HumanMessage(content=msg['content']))
        elif msg['role'] == 'assistant':
            converted_messages.append(AIMessage(content=msg['content']))
    return converted_messages



def stream_messages(graph, text: str, thread: dict, image_path: str = None):
    # Initialize the content with the text message
    content = [{"type": "text", "text": text}]

    # If image_url is provided, append the image content
    if image_path:
        base64_image = encode_image(image_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })

    # Get the windowed message history and convert to LangChain format
    if 'messages' in st.session_state:
        current_messages = convert_to_langchain_messages(st.session_state['messages'])
    else:
        current_messages = []

    # Add the new message
    input_messages = current_messages + [HumanMessage(content=content)]

    # Define the input for the graph stream
    input_data = {
        "messages": input_messages
    }

    total_tokens = calculate_messages_tokens(input_data)
    st.write(f"total tokens: {total_tokens}")

    # Initialize a variable to store the final output message
    final_message = ""

    # Stream the graph and capture only the final message output
    for s in graph.stream(input_data, config=thread):
        if "__end__" not in s:
            final_message = s
    
    return final_message

## previously working
def convert_to_langchain_messages(messages):
    """Convert dictionary messages to LangChain BaseMessage objects with enhanced content handling."""
    converted_messages = []
    for msg in messages:
        content = msg['content']
        
        # Handle different content types
        if isinstance(content, str):
            # If content is a URL, treat it as an image
            if content.startswith('http'):
                content = [{"type": "text", "text": f"Generated image: {content}"}]
            else:
                content = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            # Content is already in the correct format
            pass
        else:
            # Convert other types to text
            content = [{"type": "text", "text": str(content)}]
            
        if msg['role'] == 'user':
            converted_messages.append(HumanMessage(content=content))
        elif msg['role'] == 'assistant':
            converted_messages.append(AIMessage(content=content))
            
    return converted_messages


def calculate_message_tokens(message):
    """
    Calculate tokens for a single message. Handles both LangChain message objects and raw strings.
    """
    if isinstance(message, (HumanMessage, AIMessage)):
        # LangChain messages have a `content` attribute
        role = getattr(message, 'role', '')  # HumanMessage or AIMessage may have this attribute
        content = message.content
        return len(tokenizer.encode(role)) + len(tokenizer.encode(str(content)))
    elif isinstance(message, str):
        # Handle raw strings (fallback)
        return len(tokenizer.encode(message))
    else:
        raise TypeError(f"Unexpected message type: {type(message)}")

def calculate_messages_tokens(messages):
    """
    Calculate the total tokens for a list of messages.
    """
    return sum(calculate_message_tokens(msg) for msg in messages)

def trim_messages_to_fit(messages, max_tokens):
    """
    Trim a list of LangChain message objects to fit within a maximum token limit.
    """
    # Trim messages by removing the oldest (first) until under the token limit
    trimmed_messages = messages[:]
    
    # Trim messages until the token count is within the limit
    while calculate_messages_tokens(trimmed_messages) > max_tokens:
        trimmed_messages.pop(0)  # Remove the oldest message
    
    return trimmed_messages
    