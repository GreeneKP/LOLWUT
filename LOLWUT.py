import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from io import StringIO

st.set_page_config(page_title="Live, Out of Lagrange 1, Weather Update Tool! (LOLWUT?)", page_icon="☀️", layout="wide")

st.title("☀️ Live, Out of Lagrange 1, Weather Update Tool! (LOLWUT?)")
st.markdown("Real-time solar flares and coronal mass ejections tracking")

# API Configuration
SWPC_BASE_URL = "https://services.swpc.noaa.gov"
NASA_DONKI_CME_API = "https://api.nasa.gov/DONKI/CME"
NASA_DONKI_FLR_API = "https://api.nasa.gov/DONKI/FLR"

# NASA API Key - Get your FREE key at: https://api.nasa.gov
# DEMO_KEY is limited to 30 requests per hour per IP
# Replace with your personal key for 1000 requests per hour
NASA_API_KEY = "W8g4oXP8B1XPG2kO291e0zfmZk6ol4h0gCwAsjO5"

@st.cache_data(ttl=3600)
def fetch_historical_cmes(days=90):
    """Fetch historical CME data from NASA DONKI API"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    params = {
        'startDate': start_date.strftime('%Y-%m-%d'),
        'endDate': end_date.strftime('%Y-%m-%d'),
        'api_key': NASA_API_KEY
    }
    
    try:
        response = requests.get(NASA_DONKI_CME_API, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            st.error("⚠️ **NASA API Rate Limit Exceeded!**")
            st.warning("""
            The DEMO_KEY is limited to 30 requests per hour.
            
            **To fix this:**
            1. Get a FREE NASA API key at: https://api.nasa.gov
            2. Replace 'DEMO_KEY' on line 20 of this file with your key
            3. Your personal key allows 1,000 requests per hour!
            
            **Temporary workaround:** Wait an hour and refresh the page.
            """)
            return None
        else:
            st.error(f"HTTP Error fetching CME data: {str(e)}")
            return None
    except Exception as e:
        st.error(f"Error fetching CME data: {str(e)}")
        return None
@st.cache_data(ttl=3600)
def fetch_historical_flares(days=90):
    """Fetch historical solar flare data from NASA DONKI API"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    params = {
        'startDate': start_date.strftime('%Y-%m-%d'),
        'endDate': end_date.strftime('%Y-%m-%d'),
        'api_key': NASA_API_KEY
    }
    
    try:
        response = requests.get(NASA_DONKI_FLR_API, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            st.info("⚠️ Solar flare data rate limited. Using CME data only.")
            return None
        else:
            st.error(f"HTTP Error fetching flare data: {str(e)}")
            return None
    except Exception as e:
        st.error(f"Error fetching flare data: {str(e)}")
        return None
def classify_cme_by_speed_scorer(speed):
    """Classify CME by speed using NOAA SCORER system
    S (Slow): < 500 km/s
    C (Common): 500-999 km/s
    O (Outstanding): 1000-1499 km/s
    R (Rare): 1500-1999 km/s
    ER (Extremely Rare): >= 2000 km/s
    """
    if speed >= 2000:
        return "ER"
    elif speed >= 1500:
        return "R"
    elif speed >= 1000:
        return "O"
    elif speed >= 500:
        return "C"
    else:
        return "S"

def get_speed_class_number_scorer(speed, scorer_class):
    """Get the numeric classification (0.1-9.9) based on speed within SCORER class"""
    if scorer_class == "ER":
        # ER: 2000+ km/s, scale by 100s
        number = min(9.9, (speed - 2000) / 100 + 0.1)
        return round(number, 1)
    elif scorer_class == "R":
        # R: 1500-1999 km/s, scale by 50s
        number = min(9.9, (speed - 1500) / 50 + 0.1)
        return round(number, 1)
    elif scorer_class == "O":
        # O: 1000-1499 km/s, scale by 50s
        number = min(9.9, (speed - 1000) / 50 + 0.1)
        return round(number, 1)
    elif scorer_class == "C":
        # C: 500-999 km/s, scale by 50s
        number = min(9.9, (speed - 500) / 50 + 0.1)
        return round(number, 1)
    else:
        # S: < 500 km/s, scale by 50s
        number = min(9.9, speed / 50 + 0.1)
        return round(number, 1)

def get_most_severe_linked_flare(linked_events, flare_data_list):
    """Extract the most severe flare from linked events
    Returns tuple: (flare_class, flare_number) or (None, None)
    """
    if not linked_events:
        return None, None
    
    flare_severity_order = {'X': 4, 'M': 3, 'C': 2, 'B': 1, 'A': 0}
    most_severe_flare = None
    highest_severity = -1
    
    for linked_event in linked_events:
        activity_id = linked_event.get('activityID', '')
        # Check if this is a flare event (FLR in activity ID)
        if 'FLR' in activity_id:
            # Try to extract flare class from the linked event or look it up
            # The activity ID format is typically like "2024-01-15T12:34:00-FLR-001"
            # We need to find the actual flare class from flare_data_list if available
            if flare_data_list:
                for flare in flare_data_list:
                    if flare.get('flrID') == activity_id or flare.get('id') == activity_id:
                        class_type = flare.get('classType', flare.get('class_type', ''))
                        if class_type:
                            # Extract class letter and number (e.g., "X3.2" -> 'X', 3.2)
                            flare_class = class_type[0].upper()
                            try:
                                flare_number = float(class_type[1:])
                            except:
                                flare_number = 0.0
                            
                            severity = flare_severity_order.get(flare_class, -1)
                            
                            # Compare severity (class letter first, then number)
                            if severity > highest_severity or (severity == highest_severity and flare_number > (most_severe_flare[1] if most_severe_flare else 0)):
                                highest_severity = severity
                                most_severe_flare = (flare_class, flare_number)
    
    return most_severe_flare if most_severe_flare else (None, None)

def is_earth_directed(cme):
    """Check if CME is Earth-directed based on analysis data"""
    analyses = cme.get('cmeAnalyses', [])
    
    # Check if any analysis indicates Earth direction
    for analysis in analyses:
        # Check for isMostAccurate flag
        if analysis.get('isMostAccurate', False):
            # Check if note contains Earth-directed indicators
            note = analysis.get('note', '').lower()
            if 'earth' in note or 'geo' in note:
                return True
        
        # Check latitude and longitude - Earth-directed CMEs are typically near Sun center
        latitude = analysis.get('latitude', None)
        longitude = analysis.get('longitude', None)
        half_angle = analysis.get('halfAngle', 0)
        
        if latitude is not None and longitude is not None:
            # CMEs from near Sun center (±60° longitude, any latitude) could hit Earth
            # Or wide-angle CMEs (halo CMEs > 120° half-angle)
            if abs(longitude) <= 60 or half_angle >= 120:
                return True
    
    # Check for linked GST (Geomagnetic Storm) events - indicates Earth impact
    linked_events = cme.get('linkedEvents', [])
    if linked_events:
        for event in linked_events:
            activity_id = event.get('activityID', '')
            if 'GST' in activity_id:  # Geomagnetic Storm
                return True
    
    return False

def parse_cme_data(cme_list, flare_data_list=None):
    """Parse CME data into a structured format, filtering for Earth-directed events
    
    Args:
        cme_list: List of CME data from API
        flare_data_list: Optional list of flare data to resolve linked flare classifications
    """
    parsed_cmes = []
    
    for cme in cme_list:
        try:
            # Filter for Earth-directed CMEs only
            if not is_earth_directed(cme):
                continue
            
            cme_id = cme.get('activityID', 'Unknown')
            start_time = cme.get('startTime', 'Unknown')
            
            # Get CME analysis data
            analyses = cme.get('cmeAnalyses', [])
            if analyses:
                analysis = analyses[0]  # Use first analysis
                speed = analysis.get('speed', 0)
                half_angle = analysis.get('halfAngle', 0)
                latitude = analysis.get('latitude', 0)
                longitude = analysis.get('longitude', 0)
                
                # Calculate full width
                width = half_angle * 2 if half_angle else 0
                
                # Get linked events
                linked_events = cme.get('linkedEvents', [])
                
                # Classify CME using SCORER system
                scorer_class = classify_cme_by_speed_scorer(speed)
                class_number = get_speed_class_number_scorer(speed, scorer_class)
                
                # Check for linked flares and get most severe
                flare_class, flare_number = get_most_severe_linked_flare(linked_events, flare_data_list)
                
                # Format classification with optional flare info
                if flare_class and flare_number:
                    classification = f"{scorer_class}{class_number}/{flare_class}{flare_number}F"
                else:
                    classification = f"{scorer_class}{class_number}"
                
                # Format display string
                display = f"{classification} CME - {start_time[:10]} {start_time[11:16]} UTC - {speed} km/s"
                
                parsed_cmes.append({
                    'id': cme_id,
                    'time': start_time,
                    'speed': speed,
                    'width': width,
                    'latitude': latitude,
                    'longitude': longitude,
                    'classification': classification,
                    'scorer_class': scorer_class,
                    'scorer_number': class_number,
                    'linked_flare_class': flare_class,
                    'linked_flare_number': flare_number,
                    'display': display,
                    'datetime': datetime.fromisoformat(start_time.replace('Z', '+00:00')),
                    'linked_events': linked_events
                })
        except Exception as e:
            continue
    
    # Sort by most recent first
    parsed_cmes.sort(key=lambda x: x['datetime'], reverse=True)
    
    return parsed_cmes

def fetch_noaa_data(endpoint):
    """Fetch data from NOAA SWPC API"""
    try:
        response = requests.get(f"{SWPC_BASE_URL}{endpoint}", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        # Silently fail - errors will be handled by calling code
        return None

def fetch_text_data(url):
    """Fetch text-based data products"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def calculate_cme_arrival(cme_speed, distance_au=1.0):
    """Calculate approximate CME arrival time at Earth/geostationary orbit"""
    if cme_speed <= 0:
        return None
    
    distance_km = distance_au * 149597870.7
    arrival_hours = distance_km / (cme_speed * 3600)
    
    return arrival_hours

def estimate_impact_duration(cme_speed, cme_width):
    """Estimate impact duration based on CME characteristics"""
    base_duration = 18  # hours
    width_factor = cme_width / 360.0
    speed_factor = 1000.0 / max(cme_speed, 300)
    duration = base_duration * width_factor * speed_factor
    
    return max(6, min(duration, 48))

def calculate_affected_longitudes(cme_longitude, cme_width):
    """Calculate which geostationary longitudes will be affected"""
    # Handle None or invalid values
    if cme_longitude is None or cme_width is None:
        return (0, 360)  # Full coverage if unknown
    
    # Convert solar longitude to geostationary longitude
    geo_center = -cme_longitude
    half_width = cme_width / 2.0
    
    start_long = (geo_center - half_width) % 360
    end_long = (geo_center + half_width) % 360
    
    return (start_long, end_long)

def plot_earth_3d(affected_start, affected_end, cme_time=None, title="Geostationary Impact Zone"):
    """Create 3D Earth visualization with affected longitude zones
    
    Args:
        affected_start: Start longitude of impact zone
        affected_end: End longitude of impact zone
        cme_time: datetime object for CME arrival time (used to calculate sun position)
        title: Plot title
    """
    fig = plt.figure(figsize=(12, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    
    # Create sphere for Earth
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Plot Earth (light blue sphere for better Earth representation)
    ax.plot_surface(x, y, z, color='lightskyblue', alpha=0.8, edgecolor='none')
    
    # Geostationary orbit radius (in Earth radii, ~6.6 RE)
    geo_radius = 6.6
    
    # Create geostationary orbit ring
    theta = np.linspace(0, 2*np.pi, 360)
    orbit_x = geo_radius * np.cos(theta)
    orbit_y = geo_radius * np.sin(theta)
    orbit_z = np.zeros_like(theta)
    
    # Plot full orbit ring
    ax.plot(orbit_x, orbit_y, orbit_z, 'white', alpha=0.5, linewidth=2, label='Geostationary Orbit')
    
    # Highlight affected region
    affected_mask = np.zeros(360, dtype=bool)
    if affected_start < affected_end:
        affected_mask[int(affected_start):int(affected_end)] = True
    else:
        affected_mask[int(affected_start):] = True
        affected_mask[:int(affected_end)] = True
    
    # Plot affected zone in red
    affected_indices = np.where(affected_mask)[0]
    if len(affected_indices) > 0:
        affected_theta = np.radians(affected_indices)
        affected_x = geo_radius * np.cos(affected_theta)
        affected_y = geo_radius * np.sin(affected_theta)
        affected_z = np.zeros_like(affected_theta)
        
        ax.scatter(affected_x, affected_y, affected_z, c='red', s=20, alpha=0.8, label='Impact Zone')
    
    # Add cardinal direction lines from center through orbit
    cardinal_angles = [0, 90, 180, 270]  # degrees
    line_length = geo_radius * 1.1
    for angle in cardinal_angles:
        rad = np.radians(angle)
        line_x = [0, line_length * np.cos(rad)]
        line_y = [0, line_length * np.sin(rad)]
        line_z = [0, 0]
        ax.plot(line_x, line_y, line_z, 'k--', alpha=0.5, linewidth=1)
    
    # Add Sun direction indicator based on actual time
    # Calculate sun longitude based on time (sun appears to move ~15°/hour)
    if cme_time is not None:
        # Get hour angle: 360° per 24 hours = 15°/hour
        # At noon UTC, sun is at 0° longitude (Prime Meridian)
        # Sun moves westward (negative direction in our coordinate system)
        hour_of_day = cme_time.hour + cme_time.minute / 60.0
        # Sun longitude: 0° at noon UTC, moving westward 15°/hour
        sun_longitude = -(hour_of_day - 12.0) * 15.0  # degrees from Prime Meridian
    else:
        # Default: point toward negative X (historical behavior)
        sun_longitude = 180.0
    
    sun_distance = 10
    sun_rad = np.radians(sun_longitude)
    sun_x = -sun_distance * np.cos(sun_rad)
    sun_y = -sun_distance * np.sin(sun_rad)
    ax.quiver(0, 0, 0, sun_x, sun_y, 0, color='yellow', arrow_length_ratio=0.1, linewidth=3, label='Sun Direction')
    
    # Add cardinal direction labels on orbit
    label_positions = [0, 90, 180, 270]
    labels = ['0° (Prime Meridian)', '90°E', '180° (Intl Date Line)', '90°W']
    
    for pos, label in zip(label_positions, labels):
        rad = np.radians(pos)
        lx = (geo_radius + 0.5) * np.cos(rad)
        ly = (geo_radius + 0.5) * np.sin(rad)
        ax.text(lx, ly, 0, label, fontsize=8, color='white')
    
    # Set labels and title with white color
    ax.set_xlabel('X (Earth Radii)', fontsize=10, color='white')
    ax.set_ylabel('Y (Earth Radii)', fontsize=10, color='white')
    ax.set_zlabel('Z (Earth Radii)', fontsize=10, color='white')
    ax.set_title(title, fontsize=14, pad=20, color='white')
    
    # Set tick colors to white
    ax.tick_params(colors='white', which='both')
    
    # Set pane colors
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    # Set equal aspect ratio
    max_range = geo_radius * 1.2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    # Add legend with white text
    legend = ax.legend(loc='upper right', facecolor='black', edgecolor='white')
    for text in legend.get_texts():
        text.set_color('white')
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    return fig


def plot_flare_geo_impact_3d(source_location, flare_class, flare_time=None, title="Solar Flare GEO Belt Impact"):
    """Create 3D Earth visualization showing solar flare impact on geostationary belt
    
    Args:
        source_location: Source location string (e.g., 'S15W30', 'N10E20')
        flare_class: Flare classification (e.g., 'X1.5', 'M3.2')
        flare_time: datetime object for flare occurrence
        title: Plot title
    """
    fig = plt.figure(figsize=(12, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    
    # Create sphere for Earth
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Plot Earth (light blue sphere for better Earth representation)
    ax.plot_surface(x, y, z, color='lightskyblue', alpha=0.8, edgecolor='none')
    
    # Geostationary orbit radius (in Earth radii, ~6.6 RE)
    geo_radius = 6.6
    
    # Create geostationary orbit ring
    theta = np.linspace(0, 2*np.pi, 360)
    orbit_x = geo_radius * np.cos(theta)
    orbit_y = geo_radius * np.sin(theta)
    orbit_z = np.zeros_like(theta)
    
    # Plot full orbit ring
    ax.plot(orbit_x, orbit_y, orbit_z, 'white', alpha=0.5, linewidth=2, label='Geostationary Orbit')
    
    # Parse source location to determine impact zone
    # Source location format: S15W30 means 15° South, 30° West
    # For flares, the sunward side (dayside) of Earth is affected
    flare_longitude = 0  # Default to Prime Meridian
    latitude = 0
    
    if source_location and source_location != 'Unknown':
        try:
            # Parse latitude (N/S)
            if 'N' in source_location:
                lat_str = source_location.split('N')[1].split('E')[0].split('W')[0]
                latitude = int(''.join(filter(str.isdigit, lat_str)))
            elif 'S' in source_location:
                lat_str = source_location.split('S')[1].split('E')[0].split('W')[0]
                latitude = -int(''.join(filter(str.isdigit, lat_str)))
        
            # Parse longitude (E/W)
            if 'E' in source_location:
                long_str = source_location.split('E')[-1]
                flare_longitude = int(''.join(filter(str.isdigit, long_str)))
            elif 'W' in source_location:
                long_str = source_location.split('W')[-1]
                flare_longitude = -int(''.join(filter(str.isdigit, long_str)))
        except:
            pass
    
    # Calculate sun-facing longitude on Earth at time of flare
    if flare_time is not None:
        hour_of_day = flare_time.hour + flare_time.minute / 60.0
        # Sun longitude: 0° at noon UTC, moving westward 15°/hour
        sun_longitude = -(hour_of_day - 12.0) * 15.0
    else:
        sun_longitude = 0.0
    
    # Adjust for solar longitude offset
    impact_center_longitude = sun_longitude - flare_longitude
    
    # Flare impact zone: sunward hemisphere (±90° from sub-solar point)
    # More intense flares have wider impact zones
    impact_width = 180  # Base width for dayside hemisphere
    
    # Adjust impact width based on flare class intensity
    if flare_class.startswith('X'):
        impact_width = 200  # X-class: wider impact, wraps around terminators
        impact_color = 'red'
        impact_intensity = 1.0
    elif flare_class.startswith('M'):
        impact_width = 180  # M-class: full dayside
        impact_color = 'orange'
        impact_intensity = 0.8
    elif flare_class.startswith('C'):
        impact_width = 140  # C-class: mostly central dayside
        impact_color = 'yellow'
        impact_intensity = 0.6
    else:
        impact_width = 120  # Lower classes: limited to sub-solar region
        impact_color = 'gold'
        impact_intensity = 0.4
    
    # Calculate affected longitude range
    half_width = impact_width / 2.0
    affected_start = (impact_center_longitude - half_width) % 360
    affected_end = (impact_center_longitude + half_width) % 360
    
    # Highlight affected region on GEO belt
    affected_mask = np.zeros(360, dtype=bool)
    if affected_start < affected_end:
        affected_mask[int(affected_start):int(affected_end)] = True
    else:  # Wraps around 360°/0°
        affected_mask[int(affected_start):] = True
        affected_mask[:int(affected_end)] = True
    
    # Plot affected zone
    affected_indices = np.where(affected_mask)[0]
    if len(affected_indices) > 0:
        affected_theta = np.radians(affected_indices)
        affected_x = geo_radius * np.cos(affected_theta)
        affected_y = geo_radius * np.sin(affected_theta)
        affected_z = np.zeros_like(affected_theta)
        
        ax.scatter(affected_x, affected_y, affected_z, c=impact_color, s=20, 
                  alpha=impact_intensity, label=f'{flare_class} Impact Zone')
    
    # Add intensity gradient visualization (optional: show peak impact zone)
    peak_width = impact_width * 0.4  # Peak intensity in central 40% of impact zone
    peak_start = (impact_center_longitude - peak_width / 2) % 360
    peak_end = (impact_center_longitude + peak_width / 2) % 360
    
    peak_mask = np.zeros(360, dtype=bool)
    if peak_start < peak_end:
        peak_mask[int(peak_start):int(peak_end)] = True
    else:
        peak_mask[int(peak_start):] = True
        peak_mask[:int(peak_end)] = True
    
    peak_indices = np.where(peak_mask)[0]
    if len(peak_indices) > 0:
        peak_theta = np.radians(peak_indices)
        peak_x = geo_radius * np.cos(peak_theta)
        peak_y = geo_radius * np.sin(peak_theta)
        peak_z = np.zeros_like(peak_theta)
        
        ax.scatter(peak_x, peak_y, peak_z, c='darkred', s=120, 
                  alpha=0.9, marker='*', label='Peak Intensity Zone')
    
    # Add cardinal direction lines from center through orbit
    cardinal_angles = [0, 90, 180, 270]  # degrees
    line_length = geo_radius * 1.1
    for angle in cardinal_angles:
        rad = np.radians(angle)
        line_x = [0, line_length * np.cos(rad)]
        line_y = [0, line_length * np.sin(rad)]
        line_z = [0, 0]
        ax.plot(line_x, line_y, line_z, 'white', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add Sun direction indicator with flare source location
    # The Sun arrow MUST point toward the Sun (where radiation comes from)
    # Affected satellites are at impact_center_longitude, Sun is in that direction
    sun_distance = 10
    sun_rad = np.radians(impact_center_longitude)
    sun_x = sun_distance * np.cos(sun_rad)  # REMOVED negative sign - now points toward Sun
    sun_y = sun_distance * np.sin(sun_rad)  # REMOVED negative sign - now points toward Sun
    ax.quiver(0, 0, 0, sun_x, sun_y, 0, color='yellow', arrow_length_ratio=0.1, 
             linewidth=3, label=f'Sun → Earth (Flare at {source_location})')
    
    # Add radiation cone visualization showing the path of X-rays from Sun to Earth
    # Orange color with better visibility
    radiation_radius = geo_radius * 1.5
    cone_angle = np.radians(impact_width / 2)  # Half-angle of radiation cone
    
    # Create a cone emanating from Sun direction toward Earth
    rad_u = np.linspace(sun_rad - cone_angle, sun_rad + cone_angle, 30)
    rad_v = np.linspace(geo_radius * 0.5, radiation_radius, 15)
    
    for v in rad_v[::3]:  # Draw several rings to show cone shape
        cone_x = v * np.cos(rad_u)
        cone_y = v * np.sin(rad_u)
        cone_z = np.zeros_like(rad_u)
        ax.plot(cone_x, cone_y, cone_z, color='darkorange', alpha=0.5, linewidth=2.5)
    
    # Add cardinal direction labels on orbit
    label_positions = [0, 90, 180, 270]
    labels = ['0° (Prime Meridian)', '90°E', '180° (Intl Date Line)', '90°W']
    
    for pos, label in zip(label_positions, labels):
        rad = np.radians(pos)
        lx = (geo_radius + 0.5) * np.cos(rad)
        ly = (geo_radius + 0.5) * np.sin(rad)
        ax.text(lx, ly, 0, label, fontsize=8, color='white')
    
    # Set labels and title with white color
    ax.set_xlabel('X (Earth Radii)', fontsize=10, color='white')
    ax.set_ylabel('Y (Earth Radii)', fontsize=10, color='white')
    ax.set_zlabel('Z (Earth Radii)', fontsize=10, color='white')
    ax.set_title(title, fontsize=14, pad=20, color='white')
    
    # Set tick colors to white
    ax.tick_params(colors='white', which='both')
    
    # Set pane colors
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    # Set equal aspect ratio
    max_range = geo_radius * 1.2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    # Add legend with white text
    legend = ax.legend(loc='upper right', fontsize=8, facecolor='black', edgecolor='white')
    for text in legend.get_texts():
        text.set_color('white')
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    return fig


# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["🌟 Current Conditions", "⚡ Solar Proton Events", "📊 Forecast & Timeline"])

with tab1:
    st.header("Current Space Weather Conditions")
    
    # Current metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.subheader("🧲 Magnetic Field")
        mag_data = fetch_noaa_data("/products/summary/solar-wind-mag-field.json")
        if mag_data:
            bt = mag_data.get('Bt', 'N/A')
            st.metric("Magnetic Field Bt (Current)", f"{bt} nT")
        else:
            st.info("Fetching magnetic field data...")
    
    with col2:
        st.subheader("🌀 Solar Wind")  
        sw_data = fetch_noaa_data("/products/solar-wind/plasma-1-day.json")
        if sw_data and isinstance(sw_data, list) and len(sw_data) > 1:
            # NOAA returns array: [headers, data_row1, data_row2, ...]
            # Most recent data is typically the last row
            try:
                headers = sw_data[0]
                latest_data = sw_data[-1]  # Get most recent reading
                
                # Find speed column index
                if 'speed' in headers:
                    speed_idx = headers.index('speed')
                    speed = latest_data[speed_idx]
                    
                    # Validate speed value
                    try:
                        speed_val = float(speed)
                        if speed_val > 0 and speed_val < 5000:  # Reasonable range
                            st.metric("Solar Wind Speed (Current)", f"{speed_val:.0f} km/s")
                        else:
                            st.metric("Solar Wind Speed (Current)", "N/A")
                            st.caption("⚠️ Invalid reading")
                    except (ValueError, TypeError):
                        st.metric("Solar Wind Speed (Current)", "N/A")
                        st.caption("⚠️ Data error")
                else:
                    st.info("Speed data not available")
            except Exception as e:
                st.info("Fetching solar wind data...")
        else:
            st.info("Fetching solar wind data...")
    
    with col3:
        st.subheader("☢️ X-Ray Flux")
        # Fetch latest GOES X-ray data - try multiple endpoints
        xray_data = fetch_noaa_data("/json/goes/primary/xrays-6-hour.json")
        if not xray_data:
            xray_data = fetch_noaa_data("/json/goes/secondary/xrays-6-hour.json")
        
        if xray_data and isinstance(xray_data, list) and len(xray_data) > 1:
            try:
                # NOAA X-ray format: array of measurements
                latest_xray = xray_data[-1]  # Most recent
                
                if isinstance(latest_xray, dict):
                    flux_val = latest_xray.get('flux') or latest_xray.get('observed_flux')
                    if flux_val:
                        try:
                            flux_num = float(flux_val)
                            # Determine flare class from flux value
                            if flux_num >= 1e-4:
                                flare_class = "X-class"
                            elif flux_num >= 1e-5:
                                flare_class = "M-class"
                            elif flux_num >= 1e-6:
                                flare_class = "C-class"
                            elif flux_num >= 1e-7:
                                flare_class = "B-class"
                            else:
                                flare_class = "Quiet"
                            st.metric("X-Ray Flux (GOES)", f"{flux_num:.2e} W/m²", delta=flare_class)
                        except:
                            st.metric("X-Ray Flux (GOES)", "N/A")
                    else:
                        st.metric("X-Ray Flux (GOES)", "N/A")
                else:
                    st.metric("X-Ray Flux (GOES)", "N/A")
            except Exception as e:
                st.info("Fetching X-ray data...")
        else:
            st.info("Fetching X-ray data...")
    
    with col4:
        st.subheader("📡 Radio Flux")
        radio_json = fetch_noaa_data("/products/summary/10cm-flux.json")
        if radio_json:
            radio_flux = radio_json.get('Flux', 'N/A')
            st.metric("Solar Radio Flux", f"{radio_flux} sfu")
            st.caption("📊 2800 MHz radio emissions")
        else:
            st.info("Fetching radio flux data...")
    
    with col5:
        st.subheader("🌡️ Proton Density")
        sw_data_density = fetch_noaa_data("/products/solar-wind/plasma-1-day.json")
        if sw_data_density and isinstance(sw_data_density, list) and len(sw_data_density) > 1:
            try:
                headers = sw_data_density[0]
                latest_data = sw_data_density[-1]
                if 'density' in headers:
                    density_idx = headers.index('density')
                    density = latest_data[density_idx]
                    try:
                        density_val = float(density)
                        if density_val > 0 and density_val < 1000:
                            st.metric("Proton Density (Current)", f"{density_val:.1f} p/cm³")
                        else:
                            st.metric("Proton Density (Current)", "N/A")
                    except (ValueError, TypeError):
                        st.metric("Proton Density (Current)", "N/A")
                else:
                    st.info("Density data not available")
            except Exception as e:
                st.info("Fetching proton density data...")
        else:
            st.info("Fetching proton density data...")
    
    st.divider()
    
    # Add explanation of the difference between X-Ray and Radio Flux
    with st.expander("ℹ️ Understanding X-Ray Flux vs Radio Flux"):
        st.markdown("""
        **These are COMPLETELY DIFFERENT measurements:**
        
        **☢️ X-Ray Flux (Column 1):**
        - **What**: Measures solar X-ray radiation (0.1-0.8 nm wavelength)
        - **Units**: W/m² (Watts per square meter)
        - **Purpose**: Detects solar flares in real-time
        - **Scale**: Logarithmic (10⁻⁸ to 10⁻³ W/m²)
        - **Indicates**: Flare activity (X, M, C, B, A classes)
        - **Measured by**: GOES satellites in geostationary orbit
        - **Changes**: Rapidly during flares (seconds to minutes)
        
        **📡 Radio Flux (Column 4):**
        - **What**: Measures radio emissions at 2800 MHz (10.7 cm wavelength)
        - **Units**: sfu (Solar Flux Units, 1 sfu = 10⁻²² W/m²/Hz)
        - **Purpose**: Tracks overall solar activity level
        - **Scale**: Linear (typically 60-300 sfu)
        - **Indicates**: Solar active regions, sunspot activity, solar cycle phase
        - **Measured by**: Ground-based radio telescopes (Penticton, Canada)
        - **Changes**: Gradually over days/weeks with sunspot evolution
        
        **Why Both Matter:**
        - X-Ray Flux: Immediate threat indicator (flares happening NOW)
        - Radio Flux: Background activity level (predicts future flare probability)
        - High radio flux + sudden X-ray spike = Major space weather event
        """)
    
    st.divider()
    
    # Historical trends using multiple data sources
    st.subheader("📈 Extended Historical Trends")
    
    with st.spinner("Loading historical data from multiple sources..."):
        # Fetch data for each metric (using available NOAA endpoints)
        # NOAA provides 1-day, 3-day, and 7-day JSON endpoints
        mag_1d = fetch_noaa_data("/products/solar-wind/mag-1-day.json")
        mag_3d = fetch_noaa_data("/products/solar-wind/mag-3-day.json")
        mag_7d = fetch_noaa_data("/products/solar-wind/mag-7-day.json")
        
        plasma_1d = fetch_noaa_data("/products/solar-wind/plasma-1-day.json")
        plasma_3d = fetch_noaa_data("/products/solar-wind/plasma-3-day.json")
        plasma_7d = fetch_noaa_data("/products/solar-wind/plasma-7-day.json")
        
        # Use correct GOES X-ray endpoints
        xray_1d = fetch_noaa_data("/json/goes/primary/xrays-6-hour.json")
        xray_3d = fetch_noaa_data("/json/goes/primary/xrays-3-day.json")
        xray_7d = fetch_noaa_data("/json/goes/primary/xrays-7-day.json")
        
        # Fetch radio flux data (solar radio flux)
        radio_combined_raw = fetch_noaa_data("/products/10cm-flux-30-day.json")
        
        # Extract headers and combine data (skip first row which is headers)
        mag_combined = []
        mag_headers = None
        for dataset in [mag_7d, mag_3d, mag_1d]:
            if dataset and isinstance(dataset, list) and len(dataset) > 1:
                if mag_headers is None:
                    mag_headers = dataset[0]  # Save header from first dataset
                mag_combined.extend(dataset[1:])  # Skip header row
        
        plasma_combined = []
        plasma_headers = None
        for dataset in [plasma_7d, plasma_3d, plasma_1d]:
            if dataset and isinstance(dataset, list) and len(dataset) > 1:
                if plasma_headers is None:
                    plasma_headers = dataset[0]  # Save header from first dataset
                plasma_combined.extend(dataset[1:])  # Skip header row
        
        xray_combined = []
        xray_headers = None
        # X-ray data from GOES is in JSON format (list of dicts), not CSV-like
        for dataset in [xray_7d, xray_3d, xray_1d]:
            if dataset and isinstance(dataset, list):
                xray_combined.extend(dataset)
        
        # Parse radio flux data (array format like mag/plasma data)
        radio_combined = []
        radio_headers = None
        if radio_combined_raw and isinstance(radio_combined_raw, list) and len(radio_combined_raw) > 1:
            radio_headers = radio_combined_raw[0]
            radio_combined = radio_combined_raw[1:]
    
    # Create combined multi-axis graph showing all five metrics
    st.markdown("### 📊 Combined Space Weather Metrics")
    if mag_combined and plasma_combined and xray_combined and radio_combined and mag_headers and plasma_headers and radio_headers:
        try:
            # Prepare all five datasets
            df_mag_combined = pd.DataFrame(mag_combined, columns=mag_headers)
            df_plasma_combined = pd.DataFrame(plasma_combined, columns=plasma_headers)
            # X-ray data is already in dict format (JSON), so we can create DataFrame directly
            df_xray_combined = pd.DataFrame(xray_combined)
            df_radio_combined = pd.DataFrame(radio_combined, columns=radio_headers)
            
            # Clean magnetic field data
            if 'time_tag' in df_mag_combined.columns and 'bt' in df_mag_combined.columns:
                df_mag_combined = df_mag_combined.drop_duplicates(subset=['time_tag'], keep='first')
                df_mag_combined['time_tag'] = pd.to_datetime(df_mag_combined['time_tag'], utc=True).dt.tz_localize(None)
                df_mag_combined['bt'] = pd.to_numeric(df_mag_combined['bt'], errors='coerce')
                df_mag_combined = df_mag_combined[df_mag_combined['bt'].notna() & (df_mag_combined['bt'] > -999)]
                df_mag_combined = df_mag_combined.sort_values('time_tag')
            
            # Clean plasma data
            if 'time_tag' in df_plasma_combined.columns:
                df_plasma_combined = df_plasma_combined.drop_duplicates(subset=['time_tag'], keep='first')
                df_plasma_combined['time_tag'] = pd.to_datetime(df_plasma_combined['time_tag'], utc=True).dt.tz_localize(None)
                if 'speed' in df_plasma_combined.columns:
                    df_plasma_combined['speed'] = pd.to_numeric(df_plasma_combined['speed'], errors='coerce')
                if 'density' in df_plasma_combined.columns:
                    df_plasma_combined['density'] = pd.to_numeric(df_plasma_combined['density'], errors='coerce')
                df_plasma_combined = df_plasma_combined.sort_values('time_tag')
            
            # Clean X-ray data
            if 'time_tag' in df_xray_combined.columns:
                df_xray_combined = df_xray_combined.drop_duplicates(subset=['time_tag'], keep='first')
                df_xray_combined['time_tag'] = pd.to_datetime(df_xray_combined['time_tag'], utc=True).dt.tz_localize(None)
                # X-ray flux typically in 'flux' column with values in W/m²
                if 'flux' in df_xray_combined.columns:
                    df_xray_combined['flux'] = pd.to_numeric(df_xray_combined['flux'], errors='coerce')
                    df_xray_combined = df_xray_combined[df_xray_combined['flux'].notna() & (df_xray_combined['flux'] > 0)]
                df_xray_combined = df_xray_combined.sort_values('time_tag')
            
            # Clean radio flux data
            if 'time_tag' in df_radio_combined.columns and 'flux' in df_radio_combined.columns:
                df_radio_combined = df_radio_combined.drop_duplicates(subset=['time_tag'], keep='first')
                df_radio_combined['time_tag'] = pd.to_datetime(df_radio_combined['time_tag'], utc=True).dt.tz_localize(None)
                df_radio_combined['flux'] = pd.to_numeric(df_radio_combined['flux'], errors='coerce')
                df_radio_combined = df_radio_combined[df_radio_combined['flux'].notna() & (df_radio_combined['flux'] > 0)]
                df_radio_combined = df_radio_combined.sort_values('time_tag')
            
            # Filter all datasets to last 7 days for consistent display
            cutoff_date = datetime.now() - timedelta(days=7)
            if 'time_tag' in df_mag_combined.columns:
                df_mag_combined = df_mag_combined[df_mag_combined['time_tag'] >= cutoff_date]
            if 'time_tag' in df_plasma_combined.columns:
                df_plasma_combined = df_plasma_combined[df_plasma_combined['time_tag'] >= cutoff_date]
            if 'time_tag' in df_xray_combined.columns:
                df_xray_combined = df_xray_combined[df_xray_combined['time_tag'] >= cutoff_date]
            if 'time_tag' in df_radio_combined.columns:
                df_radio_combined = df_radio_combined[df_radio_combined['time_tag'] >= cutoff_date]
            
            # Create figure with multiple y-axes
            fig, ax1 = plt.subplots(figsize=(16, 6), facecolor='black')
            ax1.set_facecolor('black')
            
            # Calculate threshold exceedances for time-based shading
            # Create a boolean mask for each metric exceeding its threshold
            threshold_masks = {}
            all_times = pd.concat([
                df_mag_combined[['time_tag']],
                df_plasma_combined[['time_tag']] if 'speed' in df_plasma_combined.columns or 'density' in df_plasma_combined.columns else pd.DataFrame(),
                df_xray_combined[['time_tag']] if 'flux' in df_xray_combined.columns else pd.DataFrame(),
                df_radio_combined[['time_tag']] if 'flux' in df_radio_combined.columns else pd.DataFrame()
            ]).drop_duplicates().sort_values('time_tag')
            
            # Magnetic field threshold (20 nT for storm conditions)
            mag_threshold = df_mag_combined.set_index('time_tag')['bt'] >= 20
            threshold_masks['mag'] = mag_threshold.reindex(all_times['time_tag'], method='nearest', fill_value=False)
            
            # Solar wind speed threshold (500 km/s)
            if 'speed' in df_plasma_combined.columns:
                speed_threshold = df_plasma_combined.set_index('time_tag')['speed'] >= 500
                threshold_masks['speed'] = speed_threshold.reindex(all_times['time_tag'], method='nearest', fill_value=False)
            
            # Proton density threshold (10 p/cm³)
            if 'density' in df_plasma_combined.columns:
                density_threshold = df_plasma_combined.set_index('time_tag')['density'] >= 10
                threshold_masks['density'] = density_threshold.reindex(all_times['time_tag'], method='nearest', fill_value=False)
            
            # X-ray flux threshold (M-class: 1e-5 W/m²)
            if 'flux' in df_xray_combined.columns:
                xray_threshold = df_xray_combined.set_index('time_tag')['flux'] >= 1e-5
                threshold_masks['xray'] = xray_threshold.reindex(all_times['time_tag'], method='nearest', fill_value=False)
            
            # Radio flux threshold (150 sfu for moderate activity)
            if 'flux' in df_radio_combined.columns:
                radio_threshold = df_radio_combined.set_index('time_tag')['flux'] >= 150
                threshold_masks['radio'] = radio_threshold.reindex(all_times['time_tag'], method='nearest', fill_value=False)
            
            # Count how many thresholds are exceeded at each time
            exceedance_count = pd.DataFrame(threshold_masks).sum(axis=1)
            
            # Plot shaded regions FIRST (behind everything else) based on exceedance count
            # Use white shading with alpha indicating number of overlaying factors
            for i in range(len(all_times) - 1):
                count = exceedance_count.iloc[i]
                if count >= 2:
                    # Shade from current time to next time
                    t_start = all_times.iloc[i]['time_tag']
                    t_end = all_times.iloc[i + 1]['time_tag']
                    # Alpha intensity based on number of factors
                    if count == 2:
                        ax1.axvspan(t_start, t_end, facecolor='white', alpha=0.25, zorder=0)
                    elif count == 3:
                        ax1.axvspan(t_start, t_end, facecolor='white', alpha=0.45, zorder=0)
                    elif count == 4:
                        ax1.axvspan(t_start, t_end, facecolor='white', alpha=0.65, zorder=0)
                    else:  # count >= 5
                        ax1.axvspan(t_start, t_end, facecolor='white', alpha=0.85, zorder=0)
            
            # NOW plot the data lines (on top of shading)
            # First axis: Magnetic Field (left)
            color1 = 'purple'
            ax1.set_xlabel('Date', fontsize=11, color='white')
            ax1.set_ylabel('Magnetic Field Bt (nT)', color=color1, fontsize=11)
            line1 = ax1.plot(df_mag_combined['time_tag'], df_mag_combined['bt'], 
                           color=color1, alpha=0.7, linewidth=1.5, label='Magnetic Field (Bt)', zorder=2)
            ax1.tick_params(axis='y', labelcolor=color1, colors='white')
            ax1.tick_params(axis='x', colors='white')
            ax1.grid(True, alpha=0.3, zorder=1, color='white')
            for spine in ax1.spines.values():
                spine.set_edgecolor('white')
            
            # Second axis: Solar Wind Speed (right)
            ax2 = ax1.twinx()
            color2 = 'orange'
            ax2.set_ylabel('Solar Wind Speed (km/s)', color=color2, fontsize=11)
            if 'speed' in df_plasma_combined.columns:
                df_plasma_speed = df_plasma_combined[df_plasma_combined['speed'].notna() & (df_plasma_combined['speed'] > -999)]
                line2 = ax2.plot(df_plasma_speed['time_tag'], df_plasma_speed['speed'], 
                               color=color2, alpha=0.7, linewidth=1.5, label='Solar Wind Speed', zorder=2)
                ax2.tick_params(axis='y', labelcolor=color2, colors='white')
            for spine in ax2.spines.values():
                spine.set_edgecolor('white')
            
            # Third axis: Proton Density (far right, offset)
            ax3 = ax1.twinx()
            ax3.spines['right'].set_position(('outward', 60))
            color3 = 'blue'
            ax3.set_ylabel('Proton Density (p/cm³)', color=color3, fontsize=11)
            if 'density' in df_plasma_combined.columns:
                df_plasma_density = df_plasma_combined[df_plasma_combined['density'].notna() & (df_plasma_combined['density'] > -999)]
                line3 = ax3.plot(df_plasma_density['time_tag'], df_plasma_density['density'], 
                               color=color3, alpha=0.7, linewidth=1.5, label='Proton Density', zorder=2)
                ax3.tick_params(axis='y', labelcolor=color3, colors='white')
            for spine in ax3.spines.values():
                spine.set_edgecolor('white')
            
            # Fourth axis: X-Ray Flux (far right, more offset) - logarithmic scale
            ax4 = ax1.twinx()
            ax4.spines['right'].set_position(('outward', 120))
            color4 = 'red'
            ax4.set_ylabel('X-Ray Flux (W/m²)', color=color4, fontsize=11)
            if 'flux' in df_xray_combined.columns:
                df_xray_flux = df_xray_combined[df_xray_combined['flux'].notna() & (df_xray_combined['flux'] > 0)]
                line4 = ax4.plot(df_xray_flux['time_tag'], df_xray_flux['flux'], 
                               color=color4, alpha=0.7, linewidth=1.5, label='X-Ray Flux', zorder=2)
                ax4.tick_params(axis='y', labelcolor=color4, colors='white')
                ax4.set_yscale('log')  # Use log scale for X-ray flux
            for spine in ax4.spines.values():
                spine.set_edgecolor('white')
            
            # Fifth axis: Radio Flux (far right, most offset)
            ax5 = ax1.twinx()
            ax5.spines['right'].set_position(('outward', 180))
            color5 = 'mediumseagreen'
            ax5.set_ylabel('Radio Flux (sfu)', color=color5, fontsize=11)
            if 'flux' in df_radio_combined.columns:
                df_radio_flux = df_radio_combined[df_radio_combined['flux'].notna() & (df_radio_combined['flux'] > 0)]
                line5 = ax5.plot(df_radio_flux['time_tag'], df_radio_flux['flux'], 
                               color=color5, alpha=0.7, linewidth=1.5, label='Radio Flux', zorder=2)
                ax5.tick_params(axis='y', labelcolor=color5, colors='white')
            for spine in ax5.spines.values():
                spine.set_edgecolor('white')
            
            # Add title
            ax1.set_title('Combined Space Weather Metrics - 7 Day History', 
                         fontsize=14, fontweight='bold', pad=20, color='white')
            
            # Create combined legend outside plot area
            lines = line1
            if 'speed' in df_plasma_combined.columns:
                lines += line2
            if 'density' in df_plasma_combined.columns:
                lines += line3
            if 'flux' in df_xray_combined.columns:
                lines += line4
            if 'flux' in df_radio_combined.columns:
                lines += line5
            labels = [l.get_label() for l in lines]
            legend = ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(0, -0.15), ncol=3, fontsize=9, frameon=True, facecolor='black', edgecolor='white')
            for text in legend.get_texts():
                text.set_color('white')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)  # Make room for legend below
            
            st.pyplot(fig)
            plt.close()
            
            st.caption("📊 **Combined view:** Shows how magnetic field strength, solar wind speed, proton density, and X-ray flux interact over time. **Shaded regions** indicate times when 2 or more metrics exceed geomagnetic storm thresholds simultaneously - darker shading means more factors are elevated, indicating higher geomagnetic storm likelihood.")
        except Exception as e:
            st.warning(f"Could not create combined graph: {str(e)}")
    
    st.divider()
    
    # Create five plots
    trend_col1, trend_col2, trend_col3, trend_col4, trend_col5 = st.columns(5)
    
    with trend_col1:
        st.markdown("**🧲 Solar Wind Magnetic Field**")
        if mag_combined and len(mag_combined) > 0 and mag_headers:
            try:
                # Create DataFrame with proper headers
                df_mag = pd.DataFrame(mag_combined, columns=mag_headers)
                
                # Remove duplicates based on time_tag
                if 'time_tag' in df_mag.columns:
                    df_mag = df_mag.drop_duplicates(subset=['time_tag'], keep='first')
                    df_mag = df_mag.sort_values('time_tag')
                
                if not df_mag.empty and 'time_tag' in df_mag.columns:
                    df_mag['time_tag'] = pd.to_datetime(df_mag['time_tag'])
                    
                    # Use 'bt' (total magnetic field) if available
                    if 'bt' in df_mag.columns:
                        # Convert to numeric and remove invalid values
                        df_mag['bt'] = pd.to_numeric(df_mag['bt'], errors='coerce')
                        df_mag = df_mag[df_mag['bt'].notna() & (df_mag['bt'] > -999)]
                        
                        if len(df_mag) > 10:
                            fig, ax = plt.subplots(figsize=(8, 4), facecolor='black')
                            ax.set_facecolor('black')
                            
                            # Plot actual data
                            ax.plot(df_mag['time_tag'], df_mag['bt'], 
                                   color='purple', alpha=0.6, linewidth=1, label='Actual')
                            
                            # Calculate statistics
                            if len(df_mag) >= 20:
                                window = max(10, len(df_mag) // 20)
                                rolling_mean = df_mag['bt'].rolling(window=window, center=True).mean()
                                
                                # Get absolute max and min for horizontal lines
                                max_value = df_mag['bt'].max()
                                min_value = df_mag['bt'].min()
                                
                                ax.plot(df_mag['time_tag'], rolling_mean, 
                                       color='darkviolet', linewidth=2, label='Trend')
                                ax.axhline(y=max_value, color='red', linestyle='--', linewidth=1.5, label=f'High ({max_value:.1f} nT)', alpha=0.7)                            
                            # Add geomagnetic storm thresholds
                            ax.axhline(y=20, color='darkviolet', linestyle='--', linewidth=1.5, alpha=0.6, label='Storm Threshold (20 nT)')
                            ax.axhline(y=40, color='darkviolet', linestyle=':', linewidth=2, alpha=0.7, label='Severe Storm (40 nT)')
                            
                            ax.set_xlabel('Date', fontsize=9, color='white')
                            ax.set_ylabel('Bt (nT)', fontsize=9, color='white')
                            ax.set_title(f'Magnetic Field ({len(df_mag)} readings)', fontsize=10, fontweight='bold', color='white')
                            legend = ax.legend(fontsize=7, facecolor='black', edgecolor='white')
                            for text in legend.get_texts():
                                text.set_color('white')
                            ax.grid(True, alpha=0.3, color='white')
                            ax.tick_params(colors='white')
                            for spine in ax.spines.values():
                                spine.set_edgecolor('white')
                            plt.xticks(rotation=45, fontsize=7)
                            plt.yticks(fontsize=7)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                            
                            # Add explanatory text
                            st.caption("""
                            **What is Bt (Total Magnetic Field)?** The total magnetic field strength in nanoteslas (nT) measured by spacecraft at the L1 Lagrange point.
                            
                            **How it's measured:** Magnetometers on satellites like DSCOVR detect the interplanetary magnetic field (IMF) carried by the solar wind.
                            
                            **What causes changes:** Solar wind fluctuations, coronal mass ejections (CMEs), and coronal holes can strengthen or weaken the IMF. Strong southward Bz components (negative values) are particularly geoeffective.
                            
                            **Effects at peak values:** **In geostationary orbit:** Enhanced charging of satellite surfaces, increased risk of electrostatic discharge, potential damage to solar panels and electronics. **On Earth:** Geomagnetic storms causing aurora at lower latitudes, disruption of power grids, degraded GPS accuracy, and HF radio communication blackouts.
                            """)
                        else:
                            st.info("Insufficient valid magnetic field data")
                    else:
                        st.info("Magnetic field data not available in expected format")
                else:
                    st.info("No time series data available")
            except Exception as e:
                st.warning(f"Error processing magnetic field data: {str(e)}")
        else:
            st.info("Magnetic field data unavailable")
    
    with trend_col2:
        st.markdown("**🌀 Solar Wind Speed**")
        if plasma_combined and len(plasma_combined) > 0 and plasma_headers:
            try:
                # Create DataFrame with proper headers
                df_plasma = pd.DataFrame(plasma_combined, columns=plasma_headers)
                
                # Remove duplicates based on time_tag
                if 'time_tag' in df_plasma.columns:
                    df_plasma = df_plasma.drop_duplicates(subset=['time_tag'], keep='first')
                    df_plasma = df_plasma.sort_values('time_tag')
                
                if not df_plasma.empty and 'time_tag' in df_plasma.columns:
                    df_plasma['time_tag'] = pd.to_datetime(df_plasma['time_tag'])
                    
                    # Use 'speed' if available
                    if 'speed' in df_plasma.columns:
                        # Convert to numeric and remove invalid values
                        df_plasma['speed'] = pd.to_numeric(df_plasma['speed'], errors='coerce')
                        df_plasma = df_plasma[df_plasma['speed'].notna() & (df_plasma['speed'] > -999)]
                        
                        if len(df_plasma) > 10:
                            fig, ax = plt.subplots(figsize=(8, 4), facecolor='black')
                            ax.set_facecolor('black')
                            
                            # Plot actual data
                            ax.plot(df_plasma['time_tag'], df_plasma['speed'], 
                                   color='orange', alpha=0.6, linewidth=1, label='Actual')
                            
                            # Calculate statistics
                            if len(df_plasma) >= 20:
                                window = max(10, len(df_plasma) // 20)
                                rolling_mean = df_plasma['speed'].rolling(window=window, center=True).mean()
                                
                                # Get absolute max and min for horizontal lines
                                max_value = df_plasma['speed'].max()
                                min_value = df_plasma['speed'].min()
                                
                                ax.plot(df_plasma['time_tag'], rolling_mean, 
                                       color='darkorange', linewidth=2, label='Trend')
                                ax.axhline(y=max_value, color='red', linestyle='--', linewidth=1.5, label=f'High ({max_value:.0f} km/s)', alpha=0.7)                            
                            # Add geomagnetic storm thresholds
                            ax.axhline(y=500, color='darkorange', linestyle='--', linewidth=1.5, alpha=0.6, label='High-Speed (500 km/s)')
                            ax.axhline(y=600, color='darkorange', linestyle='-.', linewidth=1.5, alpha=0.7, label='Storm (600 km/s)')
                            ax.axhline(y=800, color='darkorange', linestyle=':', linewidth=2, alpha=0.8, label='Severe (800 km/s)')
                            
                            ax.set_xlabel('Date', fontsize=9, color='white')
                            ax.set_ylabel('Speed (km/s)', fontsize=9, color='white')
                            ax.set_title(f'Solar Wind Speed ({len(df_plasma)} readings)', fontsize=10, fontweight='bold', color='white')
                            legend = ax.legend(fontsize=7, facecolor='black', edgecolor='white')
                            for text in legend.get_texts():
                                text.set_color('white')
                            ax.grid(True, alpha=0.3, color='white')
                            ax.tick_params(colors='white')
                            for spine in ax.spines.values():
                                spine.set_edgecolor('white')
                            plt.xticks(rotation=45, fontsize=7)
                            plt.yticks(fontsize=7)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                            
                            # Add explanatory text
                            st.caption("""
                            **What is Solar Wind Speed?** The velocity (in km/s) at which charged particles from the Sun flow past Earth, measured at the L1 point.
                            
                            **How it's measured:** Plasma analyzers on satellites like DSCOVR and ACE measure the bulk velocity of protons and electrons streaming from the Sun.
                            
                            **What causes changes:** Typical solar wind flows at 300-500 km/s from the Sun's corona. Coronal mass ejections (CMEs) can accelerate plasma to 1000-3000 km/s. High-speed streams from coronal holes reach 600-800 km/s.
                            
                            **Effects at peak values:** **In geostationary orbit:** Increased atmospheric drag on satellites, enhanced particle radiation exposure, elevated risk of single-event upsets in electronics, spacecraft charging events. **On Earth:** Compression of Earth's magnetosphere, auroral displays extending to mid-latitudes, induced currents in power transmission lines, disruption of radio communications, and degradation of satellite navigation systems.
                            """)
                        else:
                            st.info("Insufficient valid solar wind data")
                    else:
                        st.info("Solar wind speed data not available")
                else:
                    st.info("No time series data available")
            except Exception as e:
                st.warning(f"Error processing solar wind data: {str(e)}")
        else:
            st.info("Solar wind data unavailable")
    
    with trend_col3:
        st.markdown("**☢️ X-Ray Flux**")
        if xray_combined and len(xray_combined) > 0:
            try:
                # X-ray data is already in dict format (JSON), create DataFrame directly
                df_xray = pd.DataFrame(xray_combined)
                
                # Remove duplicates based on time_tag
                if 'time_tag' in df_xray.columns:
                    df_xray = df_xray.drop_duplicates(subset=['time_tag'], keep='first')
                    df_xray = df_xray.sort_values('time_tag')
                
                if not df_xray.empty and 'time_tag' in df_xray.columns:
                    df_xray['time_tag'] = pd.to_datetime(df_xray['time_tag'])
                    
                    # Use 'flux' if available
                    if 'flux' in df_xray.columns:
                        # Convert to numeric and remove invalid values
                        df_xray['flux'] = pd.to_numeric(df_xray['flux'], errors='coerce')
                        df_xray = df_xray[df_xray['flux'].notna() & (df_xray['flux'] > 0)]
                        
                        if len(df_xray) > 10:
                            fig, ax = plt.subplots(figsize=(8, 4), facecolor='black')
                            ax.set_facecolor('black')
                            
                            # Plot actual data on log scale
                            ax.plot(df_xray['time_tag'], df_xray['flux'], 
                                   color='red', alpha=0.6, linewidth=1, label='Actual')
                            
                            # Calculate statistics
                            if len(df_xray) >= 20:
                                window = max(10, len(df_xray) // 20)
                                rolling_mean = df_xray['flux'].rolling(window=window, center=True).mean()
                                
                                # Get absolute max and min for horizontal lines
                                max_value = df_xray['flux'].max()
                                min_value = df_xray['flux'].min()
                                
                                ax.plot(df_xray['time_tag'], rolling_mean, 
                                       color='darkred', linewidth=2, label='Trend')
                                ax.axhline(y=max_value, color='red', linestyle='--', linewidth=1.5, label=f'High ({max_value:.2e} W/m²)', alpha=0.7)                            
                            # Add flare class reference lines (these are the thresholds)
                            ax.axhline(y=1e-4, color='darkred', linestyle=':', linewidth=2, label='X-class Flare (10⁻⁴)', alpha=0.7)
                            ax.axhline(y=1e-5, color='darkred', linestyle='--', linewidth=1.5, label='M-class Flare (10⁻⁵)', alpha=0.6)
                            ax.axhline(y=1e-6, color='orange', linestyle=':', linewidth=1, label='C-class threshold', alpha=0.5)
                            
                            ax.set_xlabel('Date', fontsize=9, color='white')
                            ax.set_ylabel('Flux (W/m²)', fontsize=9, color='white')
                            ax.set_yscale('log')  # Log scale for X-ray flux
                            ax.set_title(f'X-Ray Flux ({len(df_xray)} readings)', fontsize=10, fontweight='bold', color='white')
                            legend = ax.legend(fontsize=6, loc='best', facecolor='black', edgecolor='white')
                            for text in legend.get_texts():
                                text.set_color('white')
                            ax.grid(True, alpha=0.3, which='both', color='white')
                            ax.tick_params(colors='white')
                            for spine in ax.spines.values():
                                spine.set_edgecolor('white')
                            plt.xticks(rotation=45, fontsize=7)
                            plt.yticks(fontsize=7)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                            
                            # Add explanatory text
                            st.caption("""
                            **What is X-Ray Flux?** The intensity of X-ray radiation (0.1-0.8 nm wavelength) emitted by the Sun, measured in watts per square meter (W/m²) at Earth.
                            
                            **How it's measured:** X-ray sensors on GOES satellites in geostationary orbit continuously monitor solar X-ray output in two bands (short and long wavelength).
                            
                            **What causes changes:** Solar flares - sudden explosions on the Sun's surface that release intense bursts of electromagnetic radiation. Background level is ~10⁻⁸ W/m², while major X-class flares can reach >10⁻⁴ W/m².
                            
                            **Effects at peak values:** **In geostationary orbit:** Elevated radiation dose to spacecraft electronics, increased risk of single-event effects (SEE), solar panel degradation, sensor interference. **On Earth:** Ionospheric disturbances causing radio blackouts (particularly HF communications), GPS/GNSS signal degradation, disruption of over-the-horizon radar, and increased radiation exposure for high-altitude aircraft.
                            """)
                        else:
                            st.info("Insufficient valid X-ray data")
                    else:
                        st.info("X-ray flux data not available")
                else:
                    st.info("No time series data available")
            except Exception as e:
                st.warning(f"Error processing X-ray data: {str(e)}")
        else:
            st.info("X-ray data unavailable")
    
    with trend_col4:
        st.markdown("**📡 Radio Flux**")
        # Use radio data already fetched for combined graph
        if radio_combined and radio_headers:
            try:
                # Create DataFrame
                df_radio = pd.DataFrame(radio_combined, columns=radio_headers)
                
                # Check for required columns
                if 'time_tag' in df_radio.columns and 'flux' in df_radio.columns:
                    # Convert time and flux
                    df_radio['time_tag'] = pd.to_datetime(df_radio['time_tag'], utc=True).dt.tz_localize(None)
                    df_radio['flux'] = pd.to_numeric(df_radio['flux'], errors='coerce')
                    
                    # Remove invalid values and sort
                    df_radio = df_radio[df_radio['flux'].notna() & (df_radio['flux'] > 0)]
                    df_radio = df_radio.sort_values('time_tag')
                    
                    # Filter to last 7 days
                    cutoff_date = datetime.now() - timedelta(days=7)
                    df_radio = df_radio[df_radio['time_tag'] >= cutoff_date]
                    
                    if not df_radio.empty and len(df_radio) > 1:
                        # Create plot
                        fig, ax = plt.subplots(figsize=(8, 4), facecolor='black')
                        ax.set_facecolor('black')
                        
                        # Plot actual data
                        ax.plot(df_radio['time_tag'], df_radio['flux'], 'mediumseagreen', linewidth=2, label='Radio Flux')
                        
                        # Calculate statistics
                        max_value = df_radio['flux'].max()
                        
                        # Add max value line (highest in 7-day period)
                        ax.axhline(y=max_value, color='red', linestyle='--', linewidth=1.5, label=f'High ({max_value:.1f} sfu)', alpha=0.7)
                        
                        # Add activity thresholds based on typical values
                        # Quiet Sun: 60-80 sfu | Moderate: 100-150 sfu | High: 200+ sfu
                        ax.axhline(y=150, color='orange', linestyle='--', linewidth=1.5, alpha=0.6, label='Moderate Activity (150 sfu)')
                        ax.axhline(y=200, color='darkred', linestyle='--', linewidth=1.5, alpha=0.6, label='High Activity (200 sfu)')
                        
                        ax.set_xlabel('Date', color='white')
                        ax.set_ylabel('Radio Flux (sfu)', color='white')
                        ax.set_title('Radio Flux - 7 Day', color='white')
                        legend = ax.legend(fontsize=8, facecolor='black', edgecolor='white')
                        for text in legend.get_texts():
                            text.set_color('white')
                        ax.grid(True, alpha=0.3, color='white')
                        ax.tick_params(colors='white')
                        for spine in ax.spines.values():
                            spine.set_edgecolor('white')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        # Stats
                        current_flux = df_radio['flux'].iloc[-1]
                        avg_flux = df_radio['flux'].mean()
                        max_flux = df_radio['flux'].max()
                        
                        st.markdown(f"""
                        **Current:** {current_flux:.1f} sfu  
                        **7-day Average:** {avg_flux:.1f} sfu  
                        **7-day Maximum:** {max_flux:.1f} sfu
                        """)
                        
                        st.caption("""
                        **What is Radio Flux?** Solar radio emissions at 2800 MHz (10.7 cm wavelength), measured in Solar Flux Units (sfu).
                        
                        **How it's measured:** Ground-based radio telescopes (primarily Penticton, Canada) measure the intensity of radio waves emitted by the Sun.
                        
                        **What causes changes:** Solar active regions and sunspot activity. Higher radio flux indicates more active regions on the solar disk.
                        
                        **Typical values:** Quiet Sun: 60-80 sfu | Moderate activity: 100-150 sfu | High activity: 200+ sfu
                        
                        **Why it matters:** Strong predictor of solar activity levels and correlates with increased flare/CME probability.
                        
                        **Effects at peak values:** **In geostationary orbit:** Increased radiation dose to spacecraft, solar panel degradation, heightened risk of electronic anomalies. **On Earth:** Indicator of heightened solar activity, correlates with increased probability of radio blackouts and geomagnetic disturbances.
                        """)
                    else:
                        st.info("Insufficient radio flux data")
                else:
                    st.info("Radio flux data format not recognized")
            except Exception as e:
                st.warning(f"Error processing radio flux data: {str(e)}")
        else:
            st.info("Radio flux data unavailable")
    
    with trend_col5:
        st.markdown("**🌡️ Proton Density**")
        if plasma_combined and len(plasma_combined) > 0 and plasma_headers:
            try:
                # Create DataFrame with proper headers
                df_plasma = pd.DataFrame(plasma_combined, columns=plasma_headers)
                
                # Remove duplicates based on time_tag
                if 'time_tag' in df_plasma.columns:
                    df_plasma = df_plasma.drop_duplicates(subset=['time_tag'], keep='first')
                    df_plasma = df_plasma.sort_values('time_tag')
                
                if not df_plasma.empty and 'time_tag' in df_plasma.columns:
                    df_plasma['time_tag'] = pd.to_datetime(df_plasma['time_tag'])
                    
                    # Use 'density' if available
                    if 'density' in df_plasma.columns:
                        # Convert to numeric and remove invalid values
                        df_plasma['density'] = pd.to_numeric(df_plasma['density'], errors='coerce')
                        df_plasma = df_plasma[df_plasma['density'].notna() & (df_plasma['density'] > -999)]
                        
                        if len(df_plasma) > 10:
                            fig, ax = plt.subplots(figsize=(8, 4), facecolor='black')
                            ax.set_facecolor('black')
                            
                            # Plot actual data
                            ax.plot(df_plasma['time_tag'], df_plasma['density'], 
                                   color='blue', alpha=0.6, linewidth=1, label='Actual')
                            
                            # Calculate statistics
                            if len(df_plasma) >= 20:
                                window = max(10, len(df_plasma) // 20)
                                rolling_mean = df_plasma['density'].rolling(window=window, center=True).mean()
                                
                                # Get absolute max and min for horizontal lines
                                max_value = df_plasma['density'].max()
                                min_value = df_plasma['density'].min()
                                
                                ax.plot(df_plasma['time_tag'], rolling_mean, 
                                       color='darkblue', linewidth=2, label='Trend')
                                ax.axhline(y=max_value, color='red', linestyle='--', linewidth=1.5, label=f'High ({max_value:.1f} p/cm³)', alpha=0.7)                            
                            # Add geomagnetic storm thresholds
                            ax.axhline(y=10, color='darkblue', linestyle='--', linewidth=1.5, alpha=0.6, label='Elevated Density (10 p/cm³)')
                            ax.axhline(y=20, color='darkblue', linestyle=':', linewidth=2, alpha=0.7, label='High Density (20 p/cm³)')
                            
                            ax.set_xlabel('Date', fontsize=9, color='white')
                            ax.set_ylabel('Density (p/cm³)', fontsize=9, color='white')
                            ax.set_title(f'Proton Density ({len(df_plasma)} readings)', fontsize=10, fontweight='bold', color='white')
                            legend = ax.legend(fontsize=7, facecolor='black', edgecolor='white')
                            for text in legend.get_texts():
                                text.set_color('white')
                            ax.grid(True, alpha=0.3, color='white')
                            ax.tick_params(colors='white')
                            for spine in ax.spines.values():
                                spine.set_edgecolor('white')
                            plt.xticks(rotation=45, fontsize=7)
                            plt.yticks(fontsize=7)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                            
                            # Add explanatory text
                            st.caption("""
                            **What is Proton Density?** The number of protons (hydrogen nuclei) per cubic centimeter in the solar wind, measured at Earth's orbit.
                            
                            **How it's measured:** Plasma instruments on spacecraft count individual particles passing through a detector, calculating particle density from flux measurements. Normal values range from 3-10 particles/cm³.
                            
                            **What causes changes:** Solar wind density varies with solar activity cycles and specific events. CMEs carry dense plasma blobs (20-100 p/cm³). Coronal hole streams have lower density (1-5 p/cm³). Slow, dense wind emanates from the Sun's streamer belt.
                            
                            **Effects at peak values:** **In geostationary orbit:** Increased spacecraft charging due to higher plasma flux, enhanced drag forces, greater risk of surface material erosion from particle bombardment, elevated background noise in particle detectors. **On Earth:** When combined with high solar wind speed, dense plasma causes stronger geomagnetic storms, more intense aurora, greater ionospheric disturbances affecting GPS signals, and increased risk to astronauts from radiation exposure.
                            """)
                        else:
                            st.info("Insufficient valid proton density data")
                    else:
                        st.info("Proton density data not available")
                else:
                    st.info("No time series data available")
            except Exception as e:
                st.warning(f"Error processing proton density data: {str(e)}")
        else:
            st.info("Proton density data unavailable")

with tab2:
    st.header("⚡ Solar Proton Events (CMEs & Solar Flares)")
    
    st.subheader("📜 Historical Solar Events Tracker (Last 90 Days)")
    
    # Fetch both CME and flare data
    with st.spinner("Loading historical solar event data from NASA DONKI..."):
        cme_data = fetch_historical_cmes(90)
        flare_data = fetch_historical_flares(90)
    
    # Check if we got rate limited
    if cme_data is None and flare_data is None:
        st.error("📡 Unable to load solar event data due to API rate limits or errors.")
        st.info("Please see the warning message above for instructions on how to get a free NASA API key.")
        st.stop()  # Stop execution here
    
    # Combine all solar proton events
    all_events = []
    event_by_id = {}  # Quick lookup by event ID
    
    # Parse CMEs (pass flare_data to resolve linked flares)
    if cme_data:
        parsed_cmes = parse_cme_data(cme_data, flare_data)
        if parsed_cmes:
            for cme in parsed_cmes:
                try:
                    # Check if this CME has linked events
                    linked_events = cme.get('linked_events', [])
                    has_linked = isinstance(linked_events, list) and len(linked_events) > 0
                    link_indicator = " 🔗" if has_linked else ""
                    
                    event_entry = {
                        'type': 'CME',
                        'classification': cme.get('classification', 'Unknown'),
                        'display': f"{cme.get('classification', 'Unknown')} CME - {cme['time'][:10]} {cme['time'][11:16]} UTC - {cme['speed']} km/s{link_indicator}",
                        'datetime': cme['datetime'],
                        'data': cme,
                        'id': cme['id']
                    }
                    all_events.append(event_entry)
                    event_by_id[cme['id']] = event_entry
                except Exception as e:
                    # Skip malformed CME entries
                    continue
    
    # Parse flares
    if flare_data:
        parsed_flares = []
        
        for flare in flare_data:
            try:
                flare_id = flare.get('flrID', 'Unknown')
                begin_time = flare.get('beginTime', 'Unknown')
                peak_time = flare.get('peakTime', 'Unknown')
                class_type = flare.get('classType', 'Unknown')
                source_location = flare.get('sourceLocation', 'Unknown')
                
                # Check if flare is Earth-directed (based on source location)
                # Earth-directed flares typically originate from Sun center (near S00E00 or S00W00)
                is_earth_impact = False
                if source_location and source_location != 'Unknown':
                    # Parse location like "S15W30" or "N10E20"
                    try:
                        # Extract longitude component
                        if 'E' in source_location or 'W' in source_location:
                            long_str = source_location.split('E')[-1] if 'E' in source_location else source_location.split('W')[-1]
                            longitude = int(''.join(filter(str.isdigit, long_str)))
                            # Consider Earth-directed if within ±60° longitude
                            if longitude <= 60:
                                is_earth_impact = True
                    except:
                        pass
                
                # Check for linked events (CME, GST, etc.) which indicates Earth impact
                linked_events = flare.get('linkedEvents', [])
                if linked_events:
                    is_earth_impact = True
                
                # Only include Earth-impacting flares
                if is_earth_impact:
                    flare_info = {
                        'id': flare_id,
                        'begin_time': begin_time,
                        'peak_time': peak_time,
                        'class_type': class_type,
                        'source_location': source_location,
                        'datetime': datetime.fromisoformat(begin_time.replace('Z', '+00:00')),
                        'linked_events': linked_events
                    }
                    
                    parsed_flares.append(flare_info)
                    
                    # Check if this flare has linked events
                    has_linked = len(linked_events) > 0
                    link_indicator = " 🔗" if has_linked else ""
                    
                    # Add to unified events list
                    event_entry = {
                        'type': 'Flare',
                        'classification': class_type,
                        'display': f"{class_type} Flare - {begin_time[:10]} {begin_time[11:16]} UTC - {source_location}{link_indicator}",
                        'datetime': flare_info['datetime'],
                        'data': flare_info,
                        'id': flare_id
                    }
                    all_events.append(event_entry)
                    event_by_id[flare_id] = event_entry
            except Exception as e:
                continue
    
    # Sort all events by most recent first
    all_events.sort(key=lambda x: x['datetime'], reverse=True)
    
    if all_events:
        st.success(f"✅ Loaded {len(all_events)} Earth-directed solar proton events from the last 90 days (🔗 indicates linked events)")
        
        # Initialize session state for selected event if not exists
        if 'selected_event_id' not in st.session_state:
            st.session_state.selected_event_id = all_events[0]['id']
        
        # Create dropdown for event selection
        event_options = {event['display']: event for event in all_events}
        event_displays = list(event_options.keys())
        
        # Find index of currently selected event
        current_selection = None
        for i, event in enumerate(all_events):
            if event['id'] == st.session_state.selected_event_id:
                current_selection = i
                break
        
        if current_selection is None:
            current_selection = 0
            st.session_state.selected_event_id = all_events[0]['id']
        
        selected_event_display = st.selectbox(
            "Select a solar proton event (CME or Flare):",
            options=event_displays,
            index=current_selection,
            help="Choose a historical Earth-directed solar event to view details. 🔗 indicates linked events.",
            key='event_selector'
        )
        
        # Update session state when selection changes
        selected_event = event_options[selected_event_display]
        st.session_state.selected_event_id = selected_event['id']
        
        if selected_event_display:
            event_data = selected_event['data']
            
            if selected_event['type'] == 'Flare':
                # Display flare details
                selected_flare = event_data
                
                st.info(f"**Selected Flare**: {selected_flare['id']}")
                
                detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)
                
                with detail_col1:
                    st.metric("Class", selected_flare['class_type'])
                
                with detail_col2:
                    st.metric("Begin Time", selected_flare['begin_time'][11:16] + " UTC")
                
                with detail_col3:
                    st.metric("Peak Time", selected_flare['peak_time'][11:16] + " UTC")
                
                with detail_col4:
                    st.metric("Source Location", selected_flare['source_location'])
                
                # Display linked events with clickable buttons
                if selected_flare['linked_events']:
                    st.subheader("🔗 Linked Events")
                    st.write("Click on an event ID to view its details:")
                    
                    for linked_event in selected_flare['linked_events']:
                        linked_id = linked_event.get('activityID', 'Unknown')
                        
                        # Create a button for each linked event
                        if linked_id in event_by_id:
                            col1, col2 = st.columns([1, 4])
                            with col1:
                                if st.button(f"📋 View", key=f"link_flare_{linked_id}"):
                                    st.session_state.selected_event_id = linked_id
                                    st.rerun()
                            with col2:
                                st.write(f"**{linked_id}**")
                        else:
                            st.write(f"- {linked_id} (not in current dataset)")
                    
                    st.divider()
                    st.caption("📊 Linked Events Table:")
                    linked_df = pd.DataFrame(selected_flare['linked_events'])
                    st.dataframe(linked_df, use_container_width=True)
                
                # Flare impact information
                st.divider()
                st.subheader("📋 Impact Details")
                
                impact_info = {
                    "Parameter": ["Flare ID", "Class Type", "Begin Time", "Peak Time", 
                                "Source Location", "Impact on Earth"],
                    "Value": [selected_flare['id'], selected_flare['class_type'],
                             selected_flare['begin_time'], selected_flare['peak_time'],
                             selected_flare['source_location'],
                             "Immediate (X-rays travel at speed of light, ~8 min)"]
                }
                
                st.dataframe(pd.DataFrame(impact_info), use_container_width=True)
                
                # 3D GEO Belt Impact Visualization for Flare
                st.subheader("🌍 3D Geostationary Belt Impact Visualization")
                
                with st.spinner("Rendering 3D flare impact model..."):
                    try:
                        flare_datetime = datetime.fromisoformat(selected_flare['begin_time'].replace('Z', '+00:00'))
                    except:
                        flare_datetime = None
                    
                    fig_flare_3d = plot_flare_geo_impact_3d(
                        selected_flare['source_location'],
                        selected_flare['class_type'],
                        flare_time=flare_datetime,
                        title=f"Solar Flare GEO Impact - {selected_flare['class_type']} at {selected_flare['source_location']}"
                    )
                    st.pyplot(fig_flare_3d)
                    plt.close()
                
                st.caption("🔴 Colored zone indicates affected geostationary satellites on sunward side | ⭐ Stars show peak intensity zone | 🟡 Yellow arrow shows radiation path FROM Sun TO Earth | 🌟 Yellow arc lines show the radiation cone spreading from the flare")
                
                # Impact explanation
                with st.expander("ℹ️ Understanding Solar Flare Impact on GEO Belt"):
                    st.markdown(f"""
                    **Solar Flare Characteristics:**
                    - **Class**: {selected_flare['class_type']} (X-ray intensity)
                    - **Source Location**: {selected_flare['source_location']} on Sun's surface
                    - **Travel Time**: ~8 minutes (speed of light: 299,792 km/s)
                    
                    **Why the Sun arrow ALWAYS points at the impact zone:**
                    - Solar flare X-rays travel in straight lines from Sun to Earth
                    - The arrow shows the direct path of radiation
                    - The impact zone (colored satellites) is ALWAYS on the side of Earth facing the Sun
                    - The arrow and impact zone MUST intersect because radiation can only hit what it "sees"
                    
                    **GEO Belt Impact:**
                    - Solar flares emit electromagnetic radiation (X-rays, UV, radio waves)
                    - Radiation travels at the speed of light, reaching Earth in ~8 minutes
                    - **Affected Region**: Satellites on the **sunward (day) side** of Earth
                    - X-class flares can affect satellites around the dawn/dusk terminators
                    - Impact duration: Minutes to hours depending on flare duration
                    
                    **The Radiation Cone (yellow arcs):**
                    - Shows how radiation spreads out from the flare location
                    - Wider cones = more powerful flares affecting larger areas
                    - The cone encompasses all affected satellites
                    
                    **Effects on Satellites:**
                    - Increased radiation dose to electronics
                    - Solar panel degradation from UV/X-ray exposure
                    - Communication disruptions on affected frequencies
                    - Potential for single-event upsets (SEU) in electronics
                    - GPS/navigation signal degradation
                    
                    **Color Coding:**
                    - 🔴 Red (X-class): Severe impact, wide area including terminators
                    - 🟠 Orange (M-class): Moderate impact, full dayside
                    - 🟡 Yellow (C-class): Minor impact, central dayside
                    - ⭐ Dark Red Stars: Peak intensity zone (sub-solar region)
                    """)
            
            elif selected_event['type'] == 'CME':
                # Display CME details
                selected_cme = event_data
                
                cme_speed = selected_cme['speed']
                cme_width = selected_cme['width']
                cme_longitude = selected_cme['longitude']
                cme_time = selected_cme['time']
                
                st.info(f"**Selected CME**: {selected_cme['id']}")
                
                detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)
                
                with detail_col1:
                    st.metric("CME Speed", f"{cme_speed} km/s")
                
                with detail_col2:
                    st.metric("Angular Width", f"{cme_width:.1f}°")
                
                with detail_col3:
                    st.metric("Solar Longitude", f"{cme_longitude:.1f}°")
                
                with detail_col4:
                    st.metric("Launch Time", cme_time[:10])
                
                # Display linked events with clickable buttons (before impact calculations)
                if selected_cme.get('linked_events'):
                    st.divider()
                    st.subheader("🔗 Linked Events")
                    st.write("Click on an event ID to view its details:")
                    
                    for linked_event in selected_cme['linked_events']:
                        linked_id = linked_event.get('activityID', 'Unknown')
                        
                        # Create a button for each linked event
                        if linked_id in event_by_id:
                            col1, col2 = st.columns([1, 4])
                            with col1:
                                if st.button(f"📋 View", key=f"link_cme_{linked_id}"):
                                    st.session_state.selected_event_id = linked_id
                                    st.rerun()
                            with col2:
                                st.write(f"**{linked_id}**")
                        else:
                            st.write(f"- {linked_id} (not in current dataset)")
                    
                    st.divider()
                    st.caption("📊 Linked Events Table:")
                    linked_df = pd.DataFrame(selected_cme['linked_events'])
                    st.dataframe(linked_df, use_container_width=True)
                    st.divider()
                
                # Calculate impact
                if cme_speed > 0:
                    arrival_hours = calculate_cme_arrival(cme_speed)
                    
                    if arrival_hours:
                        try:
                            launch_datetime = datetime.fromisoformat(cme_time.replace('Z', '+00:00'))
                        except:
                            launch_datetime = datetime.now()
                        
                        arrival_datetime = launch_datetime + timedelta(hours=arrival_hours)
                        duration_hours = estimate_impact_duration(cme_speed, cme_width)
                        long_start, long_end = calculate_affected_longitudes(cme_longitude, cme_width)
                        
                        st.divider()
                        st.success("✅ CME Impact Analysis Complete")
                        
                        result_col1, result_col2, result_col3 = st.columns(3)
                        
                        with result_col1:
                            st.metric("Arrival at Geostationary Belt", 
                                     arrival_datetime.strftime("%Y-%m-%d %H:%M UTC"),
                                     delta=f"{arrival_hours:.1f} hours after launch")
                        
                        with result_col2:
                            st.metric("Impact Duration", 
                                     f"{duration_hours:.1f} hours",
                                     delta=f"±{duration_hours*0.2:.1f} hrs uncertainty")
                        
                        with result_col3:
                            st.metric("Affected Longitude Range", 
                                     f"{long_start:.0f}° - {long_end:.0f}°")
                        
                        st.subheader("📋 Impact Details")
                        
                        impact_info = {
                            "Parameter": ["CME ID", "Launch Time", "CME Speed", "Angular Width", 
                                        "Solar Longitude", "Travel Distance", "Arrival Time", 
                                        "Impact Duration", "Geostationary Belt Entry", 
                                        "Geostationary Belt Exit", "Affected Longitude Start", 
                                        "Affected Longitude End"],
                            "Value": [selected_cme['id'], cme_time, f"{cme_speed} km/s", 
                                     f"{cme_width:.1f}°", f"{cme_longitude:.1f}°",
                                     "~1 AU (149.6 million km)", 
                                     arrival_datetime.strftime("%Y-%m-%d %H:%M UTC"),
                                     f"{duration_hours:.1f} hours",
                                     arrival_datetime.strftime("%Y-%m-%d %H:%M UTC"),
                                     (arrival_datetime + timedelta(hours=duration_hours)).strftime("%Y-%m-%d %H:%M UTC"),
                                     f"{long_start:.1f}° East", f"{long_end:.1f}° East"]
                        }
                        
                        st.dataframe(pd.DataFrame(impact_info), use_container_width=True)
                        
                        st.subheader("🌍 3D Geostationary Impact Visualization")
                        
                        with st.spinner("Rendering 3D Earth model..."):
                            fig_3d = plot_earth_3d(long_start, long_end, 
                                                   cme_time=arrival_datetime,
                                                   title=f"CME Impact Zone - {selected_cme['id']}")
                            st.pyplot(fig_3d)
                            plt.close()
                        
                        st.caption("🔴 Red points indicate affected geostationary longitudes | 🟡 Yellow arrow shows Sun direction")
    else:
        st.warning("No Earth-directed solar events found in the last 90 days")
    
    # Event classification explanation
    with st.expander("ℹ️ Solar Event Classification"):
        st.markdown("""
        **Solar Flare Classes** (by X-ray peak flux):
        - **X-class**: Major flares (≥10⁻⁴ W/m²) - Can cause planet-wide radio blackouts
        - **M-class**: Medium flares (10⁻⁵ - 10⁻⁴ W/m²) - Can cause brief radio blackouts
        - **C-class**: Small flares (10⁻⁶ - 10⁻⁵ W/m²) - Minor impacts
        - **B-class**: Very small (10⁻⁷ - 10⁻⁶ W/m²) - No significant effects
        - **A-class**: Minimal (< 10⁻⁷ W/m²) - Background level
        
        **CME Speed Classes (NOAA SCORER System)**:
        - **ER**: Extremely Rare (≥2000 km/s) - Major geomagnetic storms
        - **R**: Rare (1500-1999 km/s) - Moderate to strong geomagnetic activity
        - **O**: Outstanding (1000-1499 km/s) - Minor to moderate geomagnetic effects
        - **C**: Common (500-999 km/s) - Minimal to minor impact
        - **S**: Slow (<500 km/s) - Negligible effects
        
        **CME Naming Convention**:
        - Format: `SCORER_CLASS + NUMBER` (e.g., `ER2.3`, `O4.7`)
        - If linked to a solar flare: `SCORER/FLARE_CLASS + F` (e.g., `ER1.3/X3.2F`)
        - When multiple flares are linked, only the most severe flare is shown
        - The number indicates magnitude within the class (0.1-9.9)
        
        **Impact Timeline**:
        - **Flares**: Immediate (X-rays travel at light speed, ~8 minutes)
        - **CMEs**: Delayed (15 hours to 4 days depending on speed)
        """)

with tab3:
    st.header("⏱️ CME Impact Timeline - Geostationary Belt")
    
    # Fetch CME and flare data for timeline
    with st.spinner("Building CME impact timeline..."):
        cme_timeline_data = fetch_historical_cmes(90)
        flare_timeline_data = fetch_historical_flares(90)
    
    if cme_timeline_data:
        parsed_timeline_cmes = parse_cme_data(cme_timeline_data, flare_timeline_data)
        
        if parsed_timeline_cmes:
            # Calculate arrival times for all CMEs
            current_time = datetime.now()
            timeline_events = []
            
            for cme in parsed_timeline_cmes:
                if cme['speed'] > 0:
                    arrival_hours = calculate_cme_arrival(cme['speed'])
                    if arrival_hours:
                        try:
                            launch_time = datetime.fromisoformat(cme['time'].replace('Z', '+00:00'))
                            arrival_time = launch_time + timedelta(hours=arrival_hours)
                            duration = estimate_impact_duration(cme['speed'], cme['width'])
                            end_time = arrival_time + timedelta(hours=duration)
                            
                            timeline_events.append({
                                'cme_id': cme['id'],
                                'classification': cme['classification'],
                                'scorer_class': cme.get('scorer_class', cme['classification'].split('/')[0][:-1] if '/' not in cme['classification'] else cme['classification'][0:2]),
                                'launch_time': launch_time,
                                'arrival_time': arrival_time,
                                'end_time': end_time,
                                'duration': duration,
                                'speed': cme['speed']
                            })
                        except:
                            continue
            
            if timeline_events:
                # Sort by arrival time
                timeline_events.sort(key=lambda x: x['arrival_time'])
                
                # Find time range - ensure current_time is timezone-aware
                from datetime import timezone
                if current_time.tzinfo is None:
                    current_time = current_time.replace(tzinfo=timezone.utc)
                
                arrival_times = [e['arrival_time'] for e in timeline_events]
                min_time = min(current_time, min(arrival_times))
                max_time = max(e['end_time'] for e in timeline_events)
                
                # Add CME longitude data for 3D visualization
                for idx, event in enumerate(timeline_events):
                    # Find matching CME from parsed data
                    matching_cme = next((c for c in parsed_timeline_cmes if c['id'] == event['cme_id']), None)
                    if matching_cme:
                        long_start, long_end = calculate_affected_longitudes(
                            matching_cme['longitude'], matching_cme['width'])
                        event['long_start'] = long_start
                        event['long_end'] = long_end
                    else:
                        event['long_start'] = 0
                        event['long_end'] = 360
                
                # Filter for future events only (after current time)
                future_events = [e for e in timeline_events if e['arrival_time'] >= current_time]
                
                if not future_events:
                    st.info("No CME impacts forecasted in the next 7 days")
                else:
                    # Create 2D time-longitude visualization
                    st.markdown("### 📊 7-Day CME Impact Forecast - Time vs Longitude")
                    
                    fig, ax = plt.subplots(figsize=(18, 12))
                    
                    # Color mapping for SCORER system
                    color_map = {
                        'ER': '#FF0000',  # Red - Extremely Rare
                        'R': '#FF6600',   # Orange - Rare
                        'O': '#FFCC00',   # Yellow - Outstanding
                        'C': '#66CC00',   # Light green - Common
                        'S': '#00CC66'    # Green - Slow
                    }
                    
                    # Time reference is current time
                    reference_time = current_time
                    
                    # Calculate 7-day window (168 hours)
                    forecast_hours = 168  # 7 days
                    end_forecast_time = current_time + timedelta(hours=forecast_hours)
                    
                    # Add Mercator-style background (longitude reference grid)
                    # Create subtle background with longitude zones
                    for lon in range(-180, 181, 30):
                        ax.axvline(x=lon, color='lightgray', linestyle=':', linewidth=1, alpha=0.5, zorder=0)
                    
                    # Add horizontal time gridlines
                    for hour in range(0, forecast_hours + 1, 24):
                        ax.axhline(y=hour, color='lightgray', linestyle=':', linewidth=1, alpha=0.5, zorder=0)
                    
                    # Add continents/regions as shaded areas (simplified Mercator reference)
                    # Americas: -180 to -30
                    ax.axvspan(-180, -30, facecolor='wheat', alpha=0.1, zorder=0)
                    # Europe/Africa: -30 to 60
                    ax.axvspan(-30, 60, facecolor='lightgreen', alpha=0.1, zorder=0)
                    # Asia/Pacific: 60 to 180
                    ax.axvspan(60, 180, facecolor='lightblue', alpha=0.1, zorder=0)
                    
                    # Plot each CME event as a rectangle
                    for event in future_events:
                        # Only plot if event starts within 7-day window
                        if event['arrival_time'] > end_forecast_time:
                            continue
                        
                        # Get SCORER class for color (handle both simple and flare-linked formats)
                        scorer_class = event.get('scorer_class', event['classification'][:2] if len(event['classification']) >= 2 else event['classification'][0])
                        # Clean up scorer_class to just the letter(s) - remove any numbers
                        if scorer_class and scorer_class[0].isalpha():
                            if len(scorer_class) > 1 and scorer_class[1].isalpha():
                                scorer_class = scorer_class[:2]  # ER
                            else:
                                scorer_class = scorer_class[0]  # S, C, O, R
                        
                        color = color_map.get(scorer_class, '#999999')
                        
                        # Get time span (y-axis) in hours from current time
                        arrival_hours = (event['arrival_time'] - reference_time).total_seconds() / 3600
                        end_hours = min((event['end_time'] - reference_time).total_seconds() / 3600, forecast_hours)
                        
                        # Skip if event ends before current time or starts after 7 days
                        if end_hours <= 0 or arrival_hours > forecast_hours:
                            continue
                        
                        # Clip to forecast window
                        arrival_hours = max(0, arrival_hours)
                        duration_hours = end_hours - arrival_hours
                        
                        if duration_hours <= 0:
                            continue
                        
                        # Get longitude span (x-axis) - convert to -180 to 180 range
                        long_start = event.get('long_start', 0)
                        long_end = event.get('long_end', 360)
                        
                        # Convert from 0-360 to -180 to 180 (centered on Prime Meridian)
                        def convert_longitude(lon):
                            lon = lon % 360
                            if lon > 180:
                                return lon - 360
                            return lon
                        
                        long_start_merc = convert_longitude(long_start)
                        long_end_merc = convert_longitude(long_end)
                        
                        # Calculate the angular span in the original 0-360 space
                        # This correctly handles wraparound
                        angular_span = (long_end - long_start) % 360
                        
                        # Check if this wraps around in the -180/180 coordinate system
                        # Wraparound occurs if the converted end is less than start (when span < 180)
                        # OR if the angular span is very large (> 180 degrees)
                        needs_wraparound = (long_start_merc > long_end_merc and angular_span < 180) or angular_span > 180
                        
                        if not needs_wraparound:
                            # Normal case - single rectangle
                            # The width should be the converted difference
                            if long_start_merc <= long_end_merc:
                                width = long_end_merc - long_start_merc
                            else:
                                # This shouldn't happen if logic is correct, but handle it
                                width = (180 - long_start_merc) + (long_end_merc + 180)
                            
                            rect = plt.Rectangle((long_start_merc, arrival_hours), width, duration_hours,
                                                facecolor=color, edgecolor='black', alpha=0.75, linewidth=2, zorder=5)
                            ax.add_patch(rect)
                            
                            # Add label in center
                            center_x = (long_start_merc + long_end_merc) / 2
                            center_y = arrival_hours + duration_hours / 2
                            ax.text(center_x, center_y, f"{event['classification']}\n{event['speed']} km/s",
                                   ha='center', va='center', fontsize=9, fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='black'),
                                   zorder=10)
                        else:
                            # Wraparound case - two rectangles
                            # Draw from long_start_merc to +180
                            width1 = 180 - long_start_merc
                            rect1 = plt.Rectangle((long_start_merc, arrival_hours), width1, duration_hours,
                                                 facecolor=color, edgecolor='black', alpha=0.75, linewidth=2, zorder=5)
                            ax.add_patch(rect1)
                            
                            # Draw from -180 to long_end_merc
                            width2 = long_end_merc - (-180)
                            rect2 = plt.Rectangle((-180, arrival_hours), width2, duration_hours,
                                                 facecolor=color, edgecolor='black', alpha=0.75, linewidth=2, zorder=5)
                            ax.add_patch(rect2)
                            
                            # Add label at Prime Meridian (0)
                            center_y = arrival_hours + duration_hours / 2
                            ax.text(0, center_y, f"{event['classification']}\n{event['speed']} km/s",
                                   ha='center', va='center', fontsize=9, fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='black'),
                                   zorder=10)
                    
                    # Mark current time line (at y=0)
                    ax.axhline(y=0, color='blue', linestyle='-', linewidth=4, 
                              label='Current Time (Now)', alpha=0.95, zorder=15)
                    
                    # Set axis properties
                    ax.set_xlim(-180, 180)
                    ax.set_xlabel('Geostationary Longitude (degrees, Prime Meridian centered)', 
                                 fontsize=14, fontweight='bold')
                    
                    # Y-axis from 0 (now) to 168 hours (7 days)
                    ax.set_ylim(0, forecast_hours)
                    ax.set_ylabel('Hours from Now', fontsize=14, fontweight='bold')
                    
                    # Add date/time labels on right y-axis
                    ax2 = ax.twinx()
                    ax2.set_ylim(0, forecast_hours)
                    
                    # Create time tick labels every 24 hours (1 day)
                    num_days = 8  # 0 to 7 days
                    tick_hours = [i * 24 for i in range(num_days)]
                    tick_labels = [(reference_time + timedelta(hours=h)).strftime('%a %m/%d\n%H:%M') 
                                  for h in tick_hours]
                    ax2.set_yticks(tick_hours)
                    ax2.set_yticklabels(tick_labels, fontsize=10)
                    ax2.set_ylabel('Date & Time (UTC)', fontsize=14, fontweight='bold')
                    
                    # X-axis longitude labels (centered on Prime Meridian)
                    longitude_ticks = [-180, -135, -90, -45, 0, 45, 90, 135, 180]
                    longitude_labels = ['180°\n(Date Line)', '135°W', '90°W\n(Americas)', '45°W', 
                                       '0°\n(Prime)', '45°E', '90°E\n(Asia)', '135°E', 
                                       '180°\n(Date Line)']
                    ax.set_xticks(longitude_ticks)
                    ax.set_xticklabels(longitude_labels, fontsize=11)
                    
                    # Enhanced grid
                    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8, color='gray', zorder=1)
                    ax.set_axisbelow(True)
                    
                    # Title
                    ax.set_title('7-Day CME Impact Forecast: Time vs Geostationary Longitude\n' +
                               f'From NOW ({reference_time.strftime("%m/%d %H:%M UTC")}) through 7 days ahead',
                               fontsize=16, fontweight='bold', pad=20)
                    
                    # Add color legend (SCORER system)
                    from matplotlib.patches import Patch
                    legend_elements = [
                        Patch(facecolor='#FF0000', label='ER - Extremely Rare (≥2000 km/s)', edgecolor='black'),
                        Patch(facecolor='#FF6600', label='R - Rare (1500-1999 km/s)', edgecolor='black'),
                        Patch(facecolor='#FFCC00', label='O - Outstanding (1000-1499 km/s)', edgecolor='black'),
                        Patch(facecolor='#66CC00', label='C - Common (500-999 km/s)', edgecolor='black'),
                        Patch(facecolor='#00CC66', label='S - Slow (<500 km/s)', edgecolor='black')
                    ]
                    ax.legend(handles=legend_elements, loc='upper left', title='CME Classification (SCORER)',
                             fontsize=11, title_fontsize=12, framealpha=0.95, edgecolor='black', fancybox=True)
                    
                    # Add region labels at top
                    ax.text(-105, forecast_hours * 1.01, 'AMERICAS', ha='center', fontsize=11, 
                           fontweight='bold', color='darkgreen', alpha=0.7)
                    ax.text(15, forecast_hours * 1.01, 'EUROPE/AFRICA', ha='center', fontsize=11, 
                           fontweight='bold', color='darkgreen', alpha=0.7)
                    ax.text(120, forecast_hours * 1.01, 'ASIA/PACIFIC', ha='center', fontsize=11, 
                           fontweight='bold', color='darkblue', alpha=0.7)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    st.caption("📊 **How to read this chart:** Each colored box shows when (vertical span) and where (horizontal span) CMEs will impact the geostationary belt. " +
                              "| 💙 Blue line at bottom = NOW | " +
                              "📍 Prime Meridian (0°) centered | " +
                              "🌍 Background shading shows global regions | " +
                              "🔗 CME labels with `/XnF` or `/MnF` format indicate CMEs linked to solar flares (e.g., ER2.3/X5.1F means an Extremely Rare CME linked to an X5.1 flare)")
                
                # Summary statistics
                st.divider()
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                
                with summary_col1:
                    future_events = [e for e in timeline_events if e['arrival_time'] > current_time]
                    st.metric("Upcoming CME Impacts", len(future_events))
                
                with summary_col2:
                    if future_events:
                        next_event = min(future_events, key=lambda x: x['arrival_time'])
                        hours_until = (next_event['arrival_time'] - current_time).total_seconds() / 3600
                        st.metric("Next Impact", f"{hours_until:.1f} hours", 
                                 delta=next_event['classification'])
                    else:
                        st.metric("Next Impact", "None forecasted")
                
                with summary_col3:
                    # Define SCORER severity order (highest to lowest)
                    scorer_severity = {'E': 5, 'R': 4, 'O': 3, 'C': 2, 'S': 1}
                    # Only check future events that are actually shown on the graph (within 7 days)
                    max_severity = max(future_events, key=lambda x: scorer_severity.get(x['classification'][0], 0))
                    st.metric("Highest Severity Event", max_severity['classification'])
            else:
                st.info("No CME impact events calculated from available data")
        else:
            st.info("No Earth-directed CMEs found in timeline period")
    else:
        st.warning("Unable to fetch CME timeline data")
    
    st.divider()
    
    # NOAA Text Forecast
    st.subheader("📅 NOAA 3-Day Forecast")
    
    forecast_url = "https://services.swpc.noaa.gov/text/3-day-forecast.txt"
    forecast_text = fetch_text_data(forecast_url)
    
    if forecast_text:
        st.text(forecast_text)
    else:
        st.warning("Unable to fetch NOAA forecast data")
    
    # Additional forecast products
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Geomagnetic Storm Probability")
        
        probabilities = {
            "Day": ["Day 1", "Day 2", "Day 3"],
            "G1 (Minor)": ["15%", "10%", "5%"],
            "G2 (Moderate)": ["5%", "3%", "1%"],
            "G3 (Strong)": ["1%", "1%", "1%"]
        }
        st.dataframe(pd.DataFrame(probabilities), use_container_width=True)
    
    with col2:
        st.subheader("📻 Radio Blackout Probability")
        
        blackout_probs = {
            "Day": ["Day 1", "Day 2", "Day 3"],
            "R1 (Minor)": ["40%", "40%", "35%"],
            "R2 (Moderate)": ["15%", "15%", "10%"],
            "R3 (Strong)": ["5%", "5%", "5%"]
        }
        st.dataframe(pd.DataFrame(blackout_probs), use_container_width=True)

# Sidebar with additional info
with st.sidebar:
    st.header("ℹ️ About")
    
    # API Key Status
    if NASA_API_KEY == "DEMO_KEY":
        st.warning("⚠️ Using DEMO_KEY (30 req/hour limit)")
        st.info("""
        **Get unlimited access:**
        1. Visit [api.nasa.gov](https://api.nasa.gov)
        2. Get your FREE API key
        3. Update line 20 in the code
        4. Enjoy 1,000 requests/hour!
        """)
    else:
        st.success("✅ Using personal NASA API key")
    
    st.divider()
    
    st.markdown("""
    This app monitors space weather using data from:
    - **NASA DONKI** (CME Database)
    - **NOAA Space Weather Prediction Center (SWPC)**
    - **GOES Satellites**
    
    ### Quick Reference
    **Geostationary Belt**: 35,786 km altitude
    
    **Solar Flare Classes**:
    - X: Major (worst)
    - M: Moderate
    - C: Minor
    - B/A: Minimal
    
    **CME Speed Classes (NOAA SCORER System)**:
    - **ER**: Extremely Rare (≥2000 km/s) - Major geomagnetic storms
    - **R**: Rare (1500-1999 km/s) - Moderate to strong geomagnetic activity
    - **O**: Outstanding (1000-1499 km/s) - Minor to moderate geomagnetic effects
    - **C**: Common (500-999 km/s) - Minimal to minor impact
    - **S**: Slow (<500 km/s) - Negligible effects
    """)
    
    st.divider()
    
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")

# Footer
st.divider()
st.caption("Data sources: NASA DONKI & NOAA SWPC | ⚠️ For official forecasts visit [spaceweather.gov](https://www.spaceweather.gov)")
