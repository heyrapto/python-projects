import os

import requests
from dotenv import load_dotenv
from geopy.geocoders import Nominatim

load_dotenv()

api_key = os.getenv("OPENWEATHERMAP_API_KEY")

if not api_key:
    raise ValueError("API key not found please check your env file.")

print(f"Successfully loaded API key {api_key[:4]}")
geolocator = Nominatim(user_agent="weather_application")


def weather_app(location: str):
    geocoded_location = geolocator.geocode(location)
    if geocoded_location:
        print(f"Logitude: {geocoded_location.longitude}")
        print(f"Latitude: {geocoded_location.latitude}")
    url = f"https://api.openweathermap.org/data/4.0/onecall/current?lat={geocoded_location.longitude}&lon={geocoded_location.latitude}&appid={api_key}"
    response = requests.get(url)

    if response.status_code == 200:
        global data
        data = response.json()
        print(data)
    else:
        print("Failed to make API call")
        return data


location = input("Enter a location:")
weather_app(location)
