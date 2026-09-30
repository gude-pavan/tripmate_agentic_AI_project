import os 
import certifi
from dotenv import load_dotenv

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import TypedDict, Annotated
import operator
import uuid
import asyncio
import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights
from tools.weather_tool import get_current_weather, get_forecast


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your Render PostgreSQL External Database URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=disable"

    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")


# =========================
# LLM
# =========================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY,
    max_tokens=2500
)


# =========================
# State
# =========================

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int
    weather_results: str


# =========================
# Destination extractor
# =========================

def extract_destination(query: str) -> str:
    """Ask the LLM to pull just the destination city/country from the query."""
    response = llm.invoke(
        f"Extract only the destination city or country from this query.\n"
        f"Query: {query}\n"
        f"Return only the destination name."
    )
    return response.content.strip()


# =========================
# Flight Agent
# =========================

# Flight Agent (async — runs in parallel with hotel & weather)
async def flight_agent_async(query: str) -> dict:
    print("\nINSIDE FLIGHT AGENT\n")

    try:
        # Direct REST call to AviationStack — no MCP subprocess needed
        flight_data = await asyncio.to_thread(search_flights, query)
        print("\nFLIGHT DATA:\n", str(flight_data)[:500])
    except Exception as e:
        flight_data = f"Flight information unavailable: {str(e)}"

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(content="Flight recommendations generated")
        ],
        "llm_calls": 1
    }


# =========================
# Hotel Agent
# =========================

# Hotel Agent (async — runs in parallel with flight & weather)
async def hotel_agent_async(query: str) -> dict:
    hotel_query = f"Best hotels for {query}"
    # Direct Tavily SDK call — no MCP HTTP stream needed
    hotel_results = await asyncio.to_thread(tavily_search, hotel_query)

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel information fetched.")
        ],
        "llm_calls": 1
    }


# =========================
# Weather Agent
# =========================

# Weather Agent (async — runs in parallel with flight & hotel)
async def weather_agent_async(query: str) -> dict:
    city = await asyncio.to_thread(extract_destination, query)

    try:
        # Fetch current weather AND forecast simultaneously via asyncio.to_thread
        # (direct HTTP calls — no MCP stdio subprocess needed)
        weather_data, forecast_data = await asyncio.gather(
            asyncio.to_thread(get_current_weather, city),
            asyncio.to_thread(get_forecast, city)
        )
    except Exception as e:
        weather_data = f"Weather unavailable: {e}"
        forecast_data = ""

    return {
        "weather_results": f"""
        Current Weather:
        {weather_data}

        Forecast:
        {forecast_data}
        """,
        "messages": [
            AIMessage(content="Weather information fetched")
        ]
    }


# =========================
# Parallel data-fetch node
# =========================

async def _gather_all_data(query: str):
    """Run flight, hotel, and weather agents concurrently."""
    flight_res, hotel_res, weather_res = await asyncio.gather(
        flight_agent_async(query),
        hotel_agent_async(query),
        weather_agent_async(query),
        return_exceptions=True
    )

    # Gracefully handle partial failures
    flight_results  = flight_res.get("flight_results",  "Flight data unavailable.")  if isinstance(flight_res,  dict) else str(flight_res)
    hotel_results   = hotel_res.get("hotel_results",    "Hotel data unavailable.")    if isinstance(hotel_res,   dict) else str(hotel_res)
    weather_results = weather_res.get("weather_results","Weather data unavailable.") if isinstance(weather_res, dict) else str(weather_res)

    return flight_results, hotel_results, weather_results


def parallel_data_agent(state: TravelState):
    """LangGraph node: runs flight, hotel, and weather in parallel."""
    query = state["user_query"]
    flight_results, hotel_results, weather_results = asyncio.run(
        _gather_all_data(query)
    )
    return {
        "flight_results":  flight_results,
        "hotel_results":   hotel_results,
        "weather_results": weather_results,
        "messages": [
            AIMessage(content="Flight, hotel, and weather data fetched in parallel.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 3
    }




# =========================
# Itinerary Agent
# =========================

def itinerary_agent(state: TravelState):
    prompt = f"""
Create a complete travel itinerary.

User Query:
{state['user_query']}

Flight Results:
{state['flight_results']}

Hotel Results:
{state['hotel_results']}

Weather Results:
{state['weather_results']}

Make the itinerary practical, budget-aware, and easy to follow.
"""

    response = llm.invoke([
        SystemMessage(content="You are an expert travel planner."),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }



# =========================
# Final Response Agent
# =========================

def final_agent(state: TravelState):
    final_prompt = f"""
Generate the final travel response for the user.

User Request:
{state['user_query']}

Flights:
{state['flight_results']}

Hotels:
{state['hotel_results']}

Weather:
{state['weather_results']}

Itinerary:
{state['itinerary']}

Format the final answer beautifully and concisely using these sections:

1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Weather Information
5. Day-by-Day Itinerary
6. Estimated Budget
7. Final Recommendations


Important:
- Be clear, crisp, and well-structured with bullet points. Avoid unnecessary filler or repetition.
- Mention that live flight API may not provide ticket prices if pricing is unavailable.
- Include weather-based travel advice.
- Keep the response useful for real travel planning.
"""

    response = llm.invoke([
        SystemMessage(content="You are a professional AI travel booking assistant."),
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Build Graph
# =========================
# Flight, hotel, and weather now run as a SINGLE parallel node,
# so the graph is: parallel_data → itinerary → final
# This replaces the old sequential chain and cuts wait time significantly.

graph = StateGraph(TravelState)

graph.add_node("parallel_data_agent", parallel_data_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "parallel_data_agent")
graph.add_edge("parallel_data_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)


# =========================
# PostgreSQL Checkpointer
# =========================
DATABASE_URL = get_database_url()

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row
)

checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)



# =========================
# Function for FastAPI
# =========================

def run_travel_agent(user_input: str, thread_id: str | None = None):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = travel_graph.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config
    )

    final_answer = result["messages"][-1].content

    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "itinerary": result.get("itinerary", ""),
        "llm_calls": result.get("llm_calls", 0),
    }