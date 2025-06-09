import os
import json
from dotenv import load_dotenv
import requests

load_dotenv('api_keys.env')

api_key = os.getenv("TOGETHER_API_KEY")

if not api_key:
    raise ValueError("API key not found. Please check your api_keys.env file and ensure TOGETHER_API_KEY is set.")
else:
    print("✅ API key loaded successfully!")

TOGETHER_API_URL = "https://api.together.ai/v1/chat/completions"

flight_data = {
    "AI123": {"flight_number": "AI123", "departure_time": "08:00 AM", "destination": "Delhi", "status": "Delayed"},
    "AI456": {"flight_number": "AI456", "departure_time": "10:30 AM", "destination": "Mumbai", "status": "On Time"},
}

def get_flight_info(flight_number: str) -> dict:
    return flight_data.get(flight_number, {"error": "Flight not found"})

def info_agent_request(flight_number: str) -> str:
    return json.dumps(get_flight_info(flight_number))

def qa_agent_respond(user_query: str) -> str:
    try:
        payload = {
            "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            "messages": [
                {"role": "system", "content": "You are a flight assistant. Extract the flight number from the user query. Respond ONLY with the flight number (e.g., 'AI123'). If no valid flight number is detected, respond with 'Not Found'."},
                {"role": "user", "content": user_query},
            ],
            "temperature": 0.3,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(TOGETHER_API_URL, headers=headers, json=payload)
        if response.status_code != 200:
            return json.dumps({"error": f"Together AI API Error: {response.json()}"})

        flight_number = response.json()['choices'][0]['message']['content'].strip()

        if flight_number == "Not Found" or flight_number not in flight_data:

            return json.dumps({"answer": "Flight number not found in the database."})

        flight_info = get_flight_info(flight_number)

        return json.dumps({
            "answer": f"Flight {flight_info['flight_number']} departs at {flight_info['departure_time']} "
                      f"to {flight_info['destination']}. Current status: {flight_info['status']}."
        })

    except Exception as e:
        return json.dumps({"error": f"Unexpected Error: {str(e)}"})

if __name__ == "__main__":
    print("Test 1:", get_flight_info("AI123"))
    print("Test 2:", info_agent_request("AI123"))
    print("Test 3:", qa_agent_respond("When does Flight AI123 depart?"))
    print("Test 4:", qa_agent_respond("What is the status of Flight AI999?"))
    print("Test 5:", qa_agent_respond("Tell me about Flight AI456."))
