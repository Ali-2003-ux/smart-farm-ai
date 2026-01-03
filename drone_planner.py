import math

def calculate_gps_coords(pixel_x, pixel_y, anchor_lat, anchor_lon, gsd_cm):
    """
    Converts pixel coordinates to GPS coordinates with high precision.
    
    Args:
        pixel_x (int): X coordinate in pixels (from left).
        pixel_y (int): Y coordinate in pixels (from top).
        anchor_lat (float): Latitude of the top-left corner (0,0).
        anchor_lon (float): Longitude of the top-left corner (0,0).
        gsd_cm (float): Ground Sampling Distance in cm/pixel.
        
    Returns:
        tuple: (lat, lon)
    """
    # 1. Calculate distance in meters from anchor
    dist_x_m = (pixel_x * gsd_cm) / 100.0
    dist_y_m = (pixel_y * gsd_cm) / 100.0
    
    # 2. Convert meters to degrees
    # Earth radius approximation? No, for "infinite precision" (local flat earth approximation is standard for small drone missions)
    # But precise conversion:
    # 1 deg latitude = ~111,132.954 meters (at equator) - 559.822 cos(2x) + 1.175 cos(4x)
    # Using standard WGS84 approx for short distances: 111,139 m per degree lat
    
    meters_per_deg_lat = 111139.0
    
    # Longitude depends on latitude
    meters_per_deg_lon = 111139.0 * math.cos(math.radians(anchor_lat))
    
    # Calculate delta
    # Image Y goes DOWN (South), so we subtract from Lat
    delta_lat = -(dist_y_m / meters_per_deg_lat) 
    
    # Image X goes RIGHT (East), so we add to Lon
    delta_lon = dist_x_m / meters_per_deg_lon
    
    new_lat = anchor_lat + delta_lat
    new_lon = anchor_lon + delta_lon
    
    return new_lat, new_lon

def generate_mavlink_mission(targets, anchor_lat, anchor_lon, gsd_cm, altitude=5.0, speed=5.0):
    """
    Generates a QGroundControl/Mission Planner compatible .waypoints file content.
    
    Args:
        targets (list): List of dicts {'x': int, 'y': int, ...}
        anchor_lat/lon: GPS of image top-left.
        gsd_cm: Ground Sampling Distance.
        altitude (float): Flight altitude in meters.
        speed (float): Flight speed in m/s.
        
    Returns:
        str: The file content string.
    """
    # Header for QGC WPL 110
    file_content = "QGC WPL 110\n"
    
    # Index 0: Home location (usually current location, but we can set it to start)
    # Seq, Current, Frame, Command, p1, p2, p3, p4, x(lat), y(lon), z(alt), autocontinue
    # Command 16 = MAV_CMD_NAV_WAYPOINT
    # Command 178 = MAV_CMD_DO_CHANGE_SPEED
    
    seq = 0
    # Add home (dummy line, typically ignored or set to first point)
    file_content += f"{seq}\t1\t0\t16\t0\t0\t0\t0\t{anchor_lat:.8f}\t{anchor_lon:.8f}\t{altitude:.2f}\t1\n"
    seq += 1
    
    # Set Speed
    file_content += f"{seq}\t0\t3\t178\t{speed:.1f}\t{speed:.1f}\t-1\t0\t0\t0\t0\t1\n"
    seq += 1
    
    for t in targets:
        lat, lon = calculate_gps_coords(t['x'], t['y'], anchor_lat, anchor_lon, gsd_cm)
        
        # Add Waypoint
        # Frame 3 = Global Relative Altitude (relative to home/takeoff)
        file_content += f"{seq}\t0\t3\t16\t0.0\t0.0\t0.0\t0.0\t{lat:.8f}\t{lon:.8f}\t{altitude:.2f}\t1\n"
        seq += 1
        
        # Optional: Add "Wait" or "Spray" command here if supported
        # e.g., MAV_CMD_NAV_DELAY (93)
        
    return file_content
