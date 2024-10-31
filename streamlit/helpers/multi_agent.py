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

# Third-party imports
import requests
import streamlit as st
import tiktoken
import urllib3
from IPython.display import Image, display
from openai import OpenAI

# Langchain core imports
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, get_buffer_string
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
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
            # MessagesPlaceholder(variable_name="agent_scratchpad")
        ]
    )

    llm_with_tools = llm.bind_tools(tools)
    agent_chain = (
        prompt |
        llm_with_tools
    )
    return agent_chain
    
    # agent = create_openai_tools_agent(llm, tools, prompt)
    # executor = AgentExecutor(agent=agent, tools=tools, return_intermediate_steps=True, verbose=True)
    # return executor

# # Define agent nodes and human feedback nodes
# def agent_node(state, agent, name):
#     result = agent.invoke(state)
#     return {"messages": [AIMessage(content=result["output"], name=name)]}

def get_username(config: RunnableConfig) -> str:
    """Get username from the config."""
    username = config["configurable"].get("username")
    if username is None:
        raise ValueError("Username needs to be provided to save a memory.")
    return username

tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

# def load_memories(state: AgentState, config: RunnableConfig) -> AgentState:
#     """Load memories for the current conversation.

#     Args:
#         state (schemas.State): The current state of the conversation.
#         config (RunnableConfig): The runtime configuration for the agent.

#     Returns:
#         State: The updated state with loaded memories.
#     """
#     convo_str = get_buffer_string(state["messages"])
#     convo_str = tokenizer.decode(tokenizer.encode(convo_str)[:2048])
#     recall_memories = search_recall_memories.invoke(convo_str, config)
#     return {
#         "recall_memories": recall_memories,
#     }


def load_memories(state: AgentState, config: RunnableConfig) -> AgentState:
    """Load memories for the current conversation.

    Args:
        state (schemas.State): The current state of the conversation.
        config (RunnableConfig): The runtime configuration for the agent.

    Returns:
        AgentState: The updated state with loaded memories.
    """
    # Get username for filtering memories
    username = get_username(config)
    
    # Get conversation string from messages
    convo_str = get_buffer_string(state["messages"])
    
    # Truncate to prevent token overflow
    convo_str = tokenizer.decode(tokenizer.encode(convo_str)[:2048])
    
    # Search for relevant memories
    recall_memories = []
    try:
        # Get the vector store instance
        collection = get_vector_store()
        
        # Query for relevant memories based on conversation context
        query_results = collection.query(
            query_texts=[convo_str],
            n_results=3,
            where={"username": username}
        )
        
        if query_results["documents"] and len(query_results["documents"][0]) > 0:
            recall_memories = query_results["documents"][0]
    except Exception as e:
        print(f"Error loading memories: {str(e)}")
    
    # Return state with loaded memories
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
        model = "dall-e-2",
        prompt = query,
        size = "512x512",
        style = "vivid",
        n = 1
    )
    return response.data[0].url

@tool
def save_recall_memory(memory: str, config: RunnableConfig) -> str:
    """Save memory to vectorstore for later semantic retrieval."""
    username = get_username(config)
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
    collection = get_vector_store()
    
    query_results = collection.query(
        query_texts=[query],
        n_results=3,
        where={"username": username}
    )
    if len(query_results["documents"][0]) == 0:
        return []
    return [doc[0] for doc in query_results["documents"]]
    

def get_vector_store():
    """Get or create vector store instance"""
    persist_directory = '/home/ubuntu/workspace/mids-capstone-fall2024-ai-art-tutor/streamlit'
    
    try:
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize the embedding function
        emb_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name="text-embedding-ada-002"
        )
        
        # Initialize the persistent client
        client = chromadb.PersistentClient(path=persist_directory)
        
        try:
            # Try to get existing collection
            collection = client.get_collection(
                name="recall_vector_store",
                embedding_function=emb_fn
            )
            print("Successfully connected to existing collection")
            
        except Exception as e:
            # If collection doesn't exist, create new collection
            print(f"Creating new collection due to: {str(e)}")
            collection = client.create_collection(
                name="recall_vector_store",
                embedding_function=emb_fn
            )
            print("Successfully created new collection")
        
        return collection
    
    except Exception as e:
        print(f"Critical error getting vector store: {str(e)}")
        raise


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

    # supervisor_chain = (
    #     prompt
    #     | gpt_4o_llm.bind_functions(functions=[function_def], function_call="route")
    #     | JsonOutputFunctionsParser()
    # )

    supervisor_chain = (
        prompt
        | gpt_4o_llm.with_structured_output(function_def)
    )



    """
    Create Other Agent Nodes
    """

    memory_usage_prompt =( "Memory Usage Guidelines:\n"
            "1. Actively use memory tools (save_recall_memory)"
            " to build a comprehensive understanding of the user.\n"
            "2. There's no need to explicitly mention your memory capabilities."
            " Instead, seamlessly incorporate your understanding of the user"
            " into your responses.\n"
            "3. Regularly reflect on past interactions to identify patterns and"
            " preferences.\n"
            "4. Cross-reference new information with existing memories for"
            " consistency.\n"     
            "5. Recognize and acknowledge changes in the user's situation or"
            " perspectives over time.\n"
            "## Recall Memories\n"
            "Recall memories are contextually retrieved based on the current"
            " conversation:\n\n")
    
    
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

    
    storyteller = create_agent(openai_key, gpt_4o_llm,[check_story_completion],
                               "Talk in a teacher's tone to 8-10 years old. You help user complete a storyline. Use check_story_completion tool to check completion.\
                                Only finish when complete otherwise keep building storyline with user. Return 'story_complete' when story is complete. Otherwise return 'story_incomplete' \
                                Actively use memory tools (save_recall_memory) to build a comprehensive understanding of the user" 
                              )
    storyteller_node = functools.partial(agent_node, agent=storyteller, name="storyteller")

    # visual_artist
    visual_artist = create_agent(openai_key, gpt_4o_llm,[generate_image],"You're a visual artist \
        You draw in a style that is similar to children's drawings from age 8 to 10, \
            Make the style as similar as possible to user's original drawings\
            Your primary job is to help users visualize ideas\
            Input to artist_tool should be a single image description. Actively use memory tools (save_recall_memory) to build a comprehensive understanding of the user")
    visual_artist_node = functools.partial(agent_node, agent=visual_artist, name="visual_artist")

    # critic
    critic = create_agent(openai_key, gpt_4o_llm,[wikipedia_tool],"You give feedback on user's artwork and how to improve.\
        Talk in an encouraging teacher's tone to 8-10 years old, be consice for each user query \
            say no more than 3-4 sentences. Use wikipedia to look up information when users asked for \
                detailed explanation of art concepts or theories. Actively use memory tools (save_recall_memory) to build a comprehensive understanding of the user")
    critic_node = functools.partial(agent_node, agent=critic, name="critic")


    silly = create_agent(openai_key, gpt_4o_llm, [moderator_tool], "You gently redirect the user back to the focus of learning art. \
    If the child is getting off track, for example, saying silly phrases, repeating words, typing the alphabet, or \
    talking about something unrelated to art, remind them that you are an art teacher. Talk in one or two sentences. Actively use memory tools (save_recall_memory) to build a comprehensive understanding of the user")

    silly_node = functools.partial(agent_node, agent = silly, name = "silly")
    
    multiagent = StateGraph(AgentState)
    tools = [generate_image, wikipedia_tool, check_story_completion, save_recall_memory, search_recall_memories, moderator_tool]
    
    # multiagent.add_node("image_moderator_node", image_moderator_node)
    multiagent.add_node("conversation_moderator_node", conversation_moderator_node)
    multiagent.add_node("load_memories", load_memories)
    multiagent.add_node("supervisor", supervisor_chain)
    multiagent.add_node("visual_artist", visual_artist_node)
    multiagent.add_node("critic", critic_node)
    multiagent.add_node("storyteller", storyteller_node)
    multiagent.add_node("silly", silly_node)
    multiagent.add_node("tools", ToolNode(tools))

    memory = MemorySaver()

    # Start conditions
    multiagent.add_edge(START, "load_memories")
    multiagent.add_edge("load_memories", "supervisor")
    
    # for supervisor to delegate
    conditional_map = {k: k for k in members} 
    multiagent.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
    
    # for each agent to use memory tools or end converation
    multiagent.add_conditional_edges("storyteller", route_tools, ["tools", "conversation_moderator_node"])
    multiagent.add_conditional_edges("critic", route_tools, ["tools","conversation_moderator_node"])
    multiagent.add_conditional_edges("silly", route_tools, ["tools","conversation_moderator_node"])
    multiagent.add_conditional_edges("visual_artist", route_tools, ["tools",END])

    # tools need to report back to agent
    multiagent.add_conditional_edges("tools", lambda x: x["next"], conditional_map)
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
   
        final_message_str = img_url
    else:
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



    