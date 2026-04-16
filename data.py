from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Coordinates:
    lon: float
    lat: float

@dataclass
class WeatherCondition:
    id: int
    main: str
    description: str
    icon: str

@dataclass
class MainStats:
    temp: float
    feels_like: float
    temp_min: float
    temp_max: float
    pressure: int
    humidity: int
    sea_level: Optional[int] = None
    grnd_level: Optional[int] = None

@dataclass
class Wind:
    speed: float
    deg: int
    gust: Optional[float] = None

@dataclass
class Rain:
    h1: float = field(metadata={"json_key": "1h"}, default=0.0)

@dataclass
class Clouds:
    all: int

@dataclass
class SystemInfo:
    type: int
    id: int
    country: str
    sunrise: int
    sunset: int

@dataclass
class WeatherData:
    coord: Coordinates
    weather: List[WeatherCondition]
    base: str
    main: MainStats
    visibility: int
    wind: Wind
    clouds: Clouds
    dt: int
    sys: SystemInfo
    timezone: int
    id: int
    name: str
    cod: int
    rain: Optional[Rain] = None