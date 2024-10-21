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




# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # human_feedback: Annotated[Sequence[HumanMessage], operator.add]
    #facts: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str

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
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key = openai_key) 
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

@tool("generate_image", args_schema=DalleInput, return_direct=True)
def generate_image(query: str):
    '''Generate image based on query'''
    return DallEAPIWrapper().run(query)


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

    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key = openai_key)

    supervisor_chain = (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )



    """
    Create Other Agent Nodes
    """
    image_moderator = (
        "You are tasked with determining if the image uploaded is appropriate for the application."
        "Appropriate content includes children's drawings that do not include violence, explicit themes, or anything not suitable for an 8-10 year old"
        "If the image is a photograph of something other than a child's drawing, it is not appropriate"
        "Return True if the image is appropriate, and False if it is not"
        "One word answers only. Invoke moderator_tool only once."
    )

    image_moderator_agent = create_agent(
        openai_key, 
        llm,
        tools=[moderator_tool],
        system_prompt=image_moderator
    )
    image_moderator_node = functools.partial(agent_node, agent=image_moderator_agent, name="image_moderator_agent")
    
    
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
        llm,
        tools=[moderator_tool],
        system_prompt=conversation_moderator
    )
    conversation_moderator_node = functools.partial(agent_node, agent=conversation_moderator_agent, name="conversation_moderator_agent")

    
    storyteller = create_agent(openai_key, llm,[check_story_completion],"Talk in a teacher's tone to 6-8 years old.\
    You help user complete a storyline. Use check_story_completion tool to check completion\
    Only finish when complete\
        Otherwise keep building storyline with user.\
            Return 'story_complete' when story is complete. Otherwise return 'story_incomplete'")
    storyteller_node = functools.partial(agent_node, agent=storyteller, name="storyteller")

    # visual_artist
    visual_artist = create_agent(openai_key, llm,[generate_image],"You're a visual artist \
        You draw in a style that is similar to children's drawings from age 6 to 8, \
            Make the style as similar as possible to user's original drawings\
            Your primary job is to help users visualize ideas\
            Input to artist_tool should be a single image description")
    visual_artist_node = functools.partial(agent_node, agent=visual_artist, name="visual_artist")

    # critic
    critic = create_agent(openai_key, llm,[wikipedia_tool],"You give feedback on user's artwork and how to improve.\
        Talk in an encouraging teacher's tone to 6-8 years old, be consice for each user query \
            say no more than 3-4 sentences. Use wikipedia to look up information when users asked for \
                detailed explanation of art concepts or theories")
    critic_node = functools.partial(agent_node, agent=critic, name="critic")


    silly = create_agent(openai_key, llm, [moderator_tool], "You gently redirect the user back to the focus of learning art. \
    If the child is getting off track, for example, saying silly phrases, repeating words, typing the alphabet, or \
    talking about something unrelated to art, remind them that you are an art teacher. Talk in one or two sentences.")

    silly_node = functools.partial(agent_node, agent = silly, name = "silly")
    
    multiagent = StateGraph(AgentState)

    multiagent.add_node("image_moderator_node", image_moderator_node)
    multiagent.add_node("conversation_moderator_node", conversation_moderator_node)

    multiagent.add_node("supervisor", supervisor_chain)
    multiagent.add_node("visual_artist", visual_artist_node)
    multiagent.add_node("critic", critic_node)
    multiagent.add_node("storyteller", storyteller_node)
    multiagent.add_node("silly", silly_node)

    memory = MemorySaver()

    # Start conditions
    multiagent.add_edge(START, "image_moderator_node")

    multiagent.add_edge("image_moderator_node", "supervisor")
    #for member in ["storyteller","visual_artist"]:
        # We want our workers to "report back" to the supervisor when done
    #   workflow.add_edge(member, "supervisor")

    # for supervisor to delegate
    conditional_map = {k: k for k in members} 
    multiagent.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

    multiagent.add_edge("storyteller", "conversation_moderator_node")
    multiagent.add_edge("visual_artist","conversation_moderator_node")
    multiagent.add_edge("critic","conversation_moderator_node")
    multiagent.add_edge("silly","conversation_moderator_node")

    # End conditions
    multiagent.add_edge("conversation_moderator_node",END)

    graph = multiagent.compile(checkpointer=memory)
    
    return graph 

def stream_messages(graph, text: str, thread: dict, image_url: str= None):
    # Initialize the content with the text message
    content = [{"type": "text", "text": text}]

    # If image_url is provided, append the image content
    if image_url:
        content.append({
            "type": "image_url",
            "image_url": {"url": image_url}
        })

    # Define the input for the graph stream
    input_data = {
        "messages": [
            HumanMessage(content=content)
        ]
    }

    # Stream the graph and print the output
    for s in graph.stream(input_data,config=thread):
        if "__end__" not in s:
            print(s)
            print("----")