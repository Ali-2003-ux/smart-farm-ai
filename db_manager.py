import sqlite3
import pandas as pd
from datetime import datetime
import os

DB_FILE = "farm_data.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Tables
    # Surveys: Represents one upload event (Date)
    c.execute('''CREATE TABLE IF NOT EXISTS surveys
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  scan_date TEXT,
                  total_palms INTEGER,
                  avg_health REAL)''')
                  
    # Palms: Individual tree data per survey
    # palm_id_track: Consistent ID across weeks (1, 2, 3...)
    c.execute('''CREATE TABLE IF NOT EXISTS palms
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  survey_id INTEGER,
                  palm_id_track INTEGER, 
                  x_coord INTEGER,
                  y_coord INTEGER,
                  area_pixels INTEGER,
                  health_score REAL,
                  growth_rate REAL,
                  FOREIGN KEY(survey_id) REFERENCES surveys(id))''')
    
    conn.commit()
    conn.close()

def save_survey(date_str, palm_data_list):
    """
    palm_data_list: List of dicts {'x':, 'y':, 'area':, 'health':}
    Returns: survey_id
    """
    # 1. Logic to match previous palms (Tracking)
    # Get previous survey
    prev_palms = get_latest_palms()
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Save Survey
    avg_health = sum([p['health'] for p in palm_data_list]) / len(palm_data_list) if palm_data_list else 0
    c.execute("INSERT INTO surveys (scan_date, total_palms, avg_health) VALUES (?, ?, ?)",
              (date_str, len(palm_data_list), avg_health))
    survey_id = c.lastrowid
    
    # Save Palms with Tracking Logic
    current_palms_saved = []
    
    for p in palm_data_list:
        assigned_id = -1
        growth = 0.0
        
        # Simple Nearest Neighbor matching
        # Find closest palm from last week within 50 pixels
        if not prev_palms.empty:
            distances = ((prev_palms['x_coord'] - p['x'])**2 + (prev_palms['y_coord'] - p['y'])**2)**0.5
            closest_idx = distances.idxmin()
            min_dist = distances.min()
            
            if min_dist < 50: # Match found!
                assigned_id = prev_palms.loc[closest_idx, 'palm_id_track']
                # Calculate Growth
                prev_area = prev_palms.loc[closest_idx, 'area_pixels']
                growth = ((p['area'] - prev_area) / prev_area) * 100
                
        # If new palm
        if assigned_id == -1:
            # Generate new ID (Max + 1)
            # This is simplified; robust systems need better ID management
            # Get max ID from DB
            c.execute("SELECT MAX(palm_id_track) FROM palms")
            res = c.fetchone()[0]
            assigned_id = 1 if res is None else res + 1
            # Check if we already assigned this ID in this current batch to avoid duplicates
            while assigned_id in [cp['id'] for cp in current_palms_saved]:
                assigned_id += 1

        c.execute('''INSERT INTO palms (survey_id, palm_id_track, x_coord, y_coord, area_pixels, health_score, growth_rate)
                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (survey_id, int(assigned_id), p['x'], p['y'], p['area'], p['health'], growth))
        
        current_palms_saved.append({'id': int(assigned_id)})

    conn.commit()
    conn.close()
    return survey_id

def get_latest_palms():
    conn = sqlite3.connect(DB_FILE)
    # Get last survey ID
    c = conn.cursor()
    c.execute("SELECT MAX(id) FROM surveys")
    last_id = c.fetchone()[0]
    
    if last_id is None:
        conn.close()
        return pd.DataFrame()
        
    df = pd.read_sql_query(f"SELECT * FROM palms WHERE survey_id = {last_id}", conn)
    conn.close()
    return df

def get_palm_history(palm_id_track):
    conn = sqlite3.connect(DB_FILE)
    query = f"""
        SELECT s.scan_date, p.area_pixels, p.health_score, p.growth_rate 
        FROM palms p
        JOIN surveys s ON p.survey_id = s.id
        WHERE p.palm_id_track = {palm_id_track}
        ORDER BY s.id ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_all_surveys():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM surveys ORDER BY id ASC", conn)
    conn.close()
    return df
    
def reset_db():
    try:
        os.remove(DB_FILE)
    except FileNotFoundError:
        pass
    init_db()

# Initialize on import
init_db()
