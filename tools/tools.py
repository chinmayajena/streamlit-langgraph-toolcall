from typing_extensions import Dict, Any
from langchain_core.tools import tool
import json
import os
from pathlib import Path


def load_weather_data():
    # Get the path to the current script and go to the data file
    script_dir = Path(__file__).parent.parent
    json_path = script_dir / "data" / "mock_india_weather_data.json"

    with open(json_path, 'r') as file:
        return json.load(file)


@tool
def get_weather(location: str) -> Dict[str, Any]:
    """
    Fetches current weather information for a given location.
    This is a mocked version as no external tool key is available at this time.

    Args:
        location (str): The name of the city of region for which to get the weather.

    Returns:
        Dict[str, Any]: A dictionary containing mock weather information.
                        Returns an error message if location is not recognized.

    """

    # Open and read the JSON file
    with open('data/mock_india_weather_data.json', 'r') as file:
        weather_data = json.load(file)

    # Example usage: print weather summary for Bangalore
    print(weather_data['bangalore']['summary'])


if __name__ == "__main__":
    weather_data = load_weather_data()
    print(weather_data["bangalore"]["summary"])
