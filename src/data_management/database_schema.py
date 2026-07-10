"""
Optimized SQLite Database Schema for Traffic Analysis.

Designed for Phase 1 as lightweight, scalable alternative to Hadoop.
Includes:
1. Normalized schema for tracks, detections, parameters
2. Indexes for fast queries
3. Partitioning strategy for multiple facilities
4. Batch insert optimization
"""

import sqlite3
from typing import Dict, List, Optional
import logging
import json
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseSchema:
    """SQLite schema definition for roundabout analysis."""
    
    # Core tables
    FACILITIES_TABLE = """
    CREATE TABLE IF NOT EXISTS facilities (
        facility_id INTEGER PRIMARY KEY,
        name TEXT UNIQUE NOT NULL,
        location TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    VIDEOS_TABLE = """
    CREATE TABLE IF NOT EXISTS videos (
        video_id INTEGER PRIMARY KEY,
        facility_id INTEGER NOT NULL,
        segment_number INTEGER NOT NULL,
        filename TEXT NOT NULL,
        duration_seconds FLOAT,
        frame_rate FLOAT,
        resolution_width INTEGER,
        resolution_height INTEGER,
        file_size_bytes INTEGER,
        import_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (facility_id) REFERENCES facilities(facility_id),
        UNIQUE(facility_id, segment_number)
    );
    """
    
    # Detection results
    DETECTIONS_TABLE = """
    CREATE TABLE IF NOT EXISTS detections (
        detection_id INTEGER PRIMARY KEY,
        video_id INTEGER NOT NULL,
        frame_number INTEGER NOT NULL,
        x1 REAL NOT NULL,
        y1 REAL NOT NULL,
        x2 REAL NOT NULL,
        y2 REAL NOT NULL,
        confidence REAL NOT NULL,
        class_id INTEGER NOT NULL,
        class_name TEXT NOT NULL,
        FOREIGN KEY (video_id) REFERENCES videos(video_id)
    );
    """
    
    # Tracking results
    TRACKS_TABLE = """
    CREATE TABLE IF NOT EXISTS tracks (
        track_id INTEGER PRIMARY KEY,
        facility_id INTEGER NOT NULL,
        video_id INTEGER NOT NULL,
        external_track_id INTEGER NOT NULL,
        class_id INTEGER NOT NULL,
        class_name TEXT NOT NULL,
        start_frame INTEGER NOT NULL,
        end_frame INTEGER NOT NULL,
        track_length INTEGER NOT NULL,
        confidence_mean REAL,
        confidence_min REAL,
        confidence_max REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (facility_id) REFERENCES facilities(facility_id),
        FOREIGN KEY (video_id) REFERENCES videos(video_id),
        UNIQUE(video_id, external_track_id)
    );
    """
    
    # Track trajectory points (normalized: only key points, not every frame)
    TRACK_POINTS_TABLE = """
    CREATE TABLE IF NOT EXISTS track_points (
        point_id INTEGER PRIMARY KEY,
        track_id INTEGER NOT NULL,
        frame_number INTEGER NOT NULL,
        x REAL NOT NULL,
        y REAL NOT NULL,
        vx REAL,
        vy REAL,
        FOREIGN KEY (track_id) REFERENCES tracks(track_id)
    );
    """
    
    # Extracted parameters per track
    TRACK_PARAMETERS_TABLE = """
    CREATE TABLE IF NOT EXISTS track_parameters (
        parameter_id INTEGER PRIMARY KEY,
        track_id INTEGER NOT NULL,
        
        -- Speed parameters (km/h)
        speed_avg REAL,
        speed_max REAL,
        speed_min REAL,
        
        -- Zone classification
        entry_frames INTEGER,
        circulatory_frames INTEGER,
        exit_frames INTEGER,
        
        -- Geometry
        turning_radius REAL,
        path_length REAL,
        curvature_avg REAL,
        
        -- TRL/AASHTO classification
        vehicle_class TEXT,
        vehicle_category TEXT,
        
        -- Trajectory analysis
        first_zone TEXT,
        last_zone TEXT,
        zone_sequence TEXT,
        
        -- Time-based metrics
        time_in_roundabout_seconds REAL,
        dwell_time_seconds REAL,
        
        -- Quality metrics
        track_quality REAL,
        confidence_score REAL,
        
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (track_id) REFERENCES tracks(track_id)
    );
    """
    
    # Aggregated metrics per facility/time period
    AGGREGATED_METRICS_TABLE = """
    CREATE TABLE IF NOT EXISTS aggregated_metrics (
        metric_id INTEGER PRIMARY KEY,
        facility_id INTEGER NOT NULL,
        time_period_start TIMESTAMP NOT NULL,
        time_period_end TIMESTAMP NOT NULL,
        period_duration_minutes INTEGER NOT NULL,
        
        -- Volume metrics
        vehicle_count INTEGER,
        vehicle_class_car INTEGER,
        vehicle_class_bus INTEGER,
        vehicle_class_truck INTEGER,
        vehicle_class_motorcycle INTEGER,
        vehicle_class_pedestrian INTEGER,
        
        -- Speed metrics
        speed_avg REAL,
        speed_85th_percentile REAL,
        speed_50th_percentile REAL,
        
        -- Delay metrics
        total_delay_hours REAL,
        avg_delay_per_vehicle REAL,
        
        -- Flow metrics
        entry_flow_vehicles_per_hour REAL,
        exit_flow_vehicles_per_hour REAL,
        circulating_flow_density REAL,
        
        -- Turning metrics
        left_turn_percentage REAL,
        straight_percentage REAL,
        right_turn_percentage REAL,
        
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (facility_id) REFERENCES facilities(facility_id)
    );
    """
    
    # Signal timing optimization recommendations
    OPTIMIZATION_RESULTS_TABLE = """
    CREATE TABLE IF NOT EXISTS optimization_results (
        optimization_id INTEGER PRIMARY KEY,
        facility_id INTEGER NOT NULL,
        analysis_period_start TIMESTAMP NOT NULL,
        analysis_period_end TIMESTAMP NOT NULL,
        
        -- Current signal timing
        current_cycle_length_seconds INTEGER,
        current_green_time_entry_seconds INTEGER,
        
        -- Recommendations
        recommended_cycle_length_seconds INTEGER,
        recommended_green_time_entry_seconds INTEGER,
        
        -- Expected improvements
        expected_delay_reduction_percent REAL,
        expected_throughput_increase_percent REAL,
        expected_emissions_reduction_percent REAL,
        
        -- Simulation results
        simulation_confidence REAL,
        notes TEXT,
        
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (facility_id) REFERENCES facilities(facility_id)
    );
    """
    
    # Data quality and validation logs
    VALIDATION_LOG_TABLE = """
    CREATE TABLE IF NOT EXISTS validation_log (
        log_id INTEGER PRIMARY KEY,
        facility_id INTEGER NOT NULL,
        video_id INTEGER,
        track_id INTEGER,
        validation_type TEXT NOT NULL,
        status TEXT NOT NULL,
        error_message TEXT,
        severity TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (facility_id) REFERENCES facilities(facility_id),
        FOREIGN KEY (video_id) REFERENCES videos(video_id),
        FOREIGN KEY (track_id) REFERENCES tracks(track_id)
    );
    """
    
    # Indexes for performance
    INDEXES = [
        # Detection indexes
        "CREATE INDEX IF NOT EXISTS idx_detections_video_frame ON detections(video_id, frame_number);",
        "CREATE INDEX IF NOT EXISTS idx_detections_class ON detections(class_id);",
        
        # Track indexes
        "CREATE INDEX IF NOT EXISTS idx_tracks_facility ON tracks(facility_id);",
        "CREATE INDEX IF NOT EXISTS idx_tracks_video ON tracks(video_id);",
        "CREATE INDEX IF NOT EXISTS idx_tracks_class ON tracks(class_id);",
        "CREATE INDEX IF NOT EXISTS idx_tracks_created ON tracks(created_at);",
        
        # Track points indexes
        "CREATE INDEX IF NOT EXISTS idx_track_points_track ON track_points(track_id);",
        "CREATE INDEX IF NOT EXISTS idx_track_points_frame ON track_points(frame_number);",
        
        # Parameter indexes
        "CREATE INDEX IF NOT EXISTS idx_track_params_track ON track_parameters(track_id);",
        "CREATE INDEX IF NOT EXISTS idx_track_params_class ON track_parameters(vehicle_class);",
        
        # Metrics indexes
        "CREATE INDEX IF NOT EXISTS idx_metrics_facility_time ON aggregated_metrics(facility_id, time_period_start);",
        
        # Validation indexes
        "CREATE INDEX IF NOT EXISTS idx_validation_facility ON validation_log(facility_id);",
        "CREATE INDEX IF NOT EXISTS idx_validation_type ON validation_log(validation_type, status);",
    ]
    
    @classmethod
    def get_all_create_statements(cls) -> List[str]:
        """Get all table creation statements."""
        return [
            cls.FACILITIES_TABLE,
            cls.VIDEOS_TABLE,
            cls.DETECTIONS_TABLE,
            cls.TRACKS_TABLE,
            cls.TRACK_POINTS_TABLE,
            cls.TRACK_PARAMETERS_TABLE,
            cls.AGGREGATED_METRICS_TABLE,
            cls.OPTIMIZATION_RESULTS_TABLE,
            cls.VALIDATION_LOG_TABLE,
        ]


class DatabaseManager:
    """Manager for SQLite database operations."""
    
    def __init__(self, db_path: str, auto_init: bool = True):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
            auto_init: Whether to auto-create schema
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = None
        self.cursor = None
        
        self.connect()
        
        if auto_init:
            self.initialize_schema()
    
    def connect(self):
        """Establish database connection."""
        try:
            self.connection = sqlite3.connect(
                str(self.db_path),
                timeout=10.0,
                check_same_thread=False
            )
            self.cursor = self.connection.cursor()
            
            # Enable performance optimizations
            self.cursor.execute("PRAGMA journal_mode = WAL;")  # Write-Ahead Logging
            self.cursor.execute("PRAGMA synchronous = NORMAL;")  # Better performance
            self.cursor.execute("PRAGMA cache_size = 10000;")  # Larger cache
            self.cursor.execute("PRAGMA temp_store = MEMORY;")  # In-memory temp
            
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def initialize_schema(self):
        """Create all tables and indexes."""
        try:
            # Create tables
            for statement in DatabaseSchema.get_all_create_statements():
                self.cursor.execute(statement)
            
            # Create indexes
            for index_statement in DatabaseSchema.INDEXES:
                self.cursor.execute(index_statement)
            
            self.connection.commit()
            logger.info("Database schema initialized")
        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            self.connection.rollback()
            raise
    
    def insert_facility(self, name: str, location: Optional[str] = None) -> int:
        """Insert facility and return facility_id."""
        self.cursor.execute(
            "INSERT INTO facilities (name, location) VALUES (?, ?)",
            (name, location)
        )
        self.connection.commit()
        return self.cursor.lastrowid
    
    def insert_video(self, facility_id: int, segment_number: int, 
                    filename: str, **kwargs) -> int:
        """Insert video metadata."""
        self.cursor.execute(
            """INSERT INTO videos 
               (facility_id, segment_number, filename, duration_seconds, 
                frame_rate, resolution_width, resolution_height, file_size_bytes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (facility_id, segment_number, filename,
             kwargs.get('duration_seconds'),
             kwargs.get('frame_rate'),
             kwargs.get('resolution_width'),
             kwargs.get('resolution_height'),
             kwargs.get('file_size_bytes'))
        )
        self.connection.commit()
        return self.cursor.lastrowid
    
    def insert_track(self, facility_id: int, video_id: int, 
                    external_track_id: int, class_id: int, class_name: str,
                    start_frame: int, end_frame: int, **kwargs) -> int:
        """Insert track metadata."""
        track_length = end_frame - start_frame + 1
        
        self.cursor.execute(
            """INSERT INTO tracks 
               (facility_id, video_id, external_track_id, class_id, class_name,
                start_frame, end_frame, track_length, confidence_mean, 
                confidence_min, confidence_max)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (facility_id, video_id, external_track_id, class_id, class_name,
             start_frame, end_frame, track_length,
             kwargs.get('confidence_mean'),
             kwargs.get('confidence_min'),
             kwargs.get('confidence_max'))
        )
        self.connection.commit()
        return self.cursor.lastrowid
    
    def batch_insert_track_points(self, track_id: int, 
                                 points: List[Dict]) -> int:
        """Batch insert track points (optimized for large inserts)."""
        data = [
            (track_id, p['frame_number'], p['x'], p['y'], 
             p.get('vx'), p.get('vy'))
            for p in points
        ]
        
        self.cursor.executemany(
            """INSERT INTO track_points 
               (track_id, frame_number, x, y, vx, vy)
               VALUES (?, ?, ?, ?, ?, ?)""",
            data
        )
        self.connection.commit()
        return len(data)
    
    def insert_track_parameters(self, track_id: int, parameters: Dict) -> int:
        """Insert extracted parameters for a track."""
        self.cursor.execute(
            """INSERT INTO track_parameters 
               (track_id, speed_avg, speed_max, speed_min, entry_frames,
                circulatory_frames, exit_frames, turning_radius, path_length,
                curvature_avg, vehicle_class, vehicle_category, first_zone,
                last_zone, zone_sequence, time_in_roundabout_seconds,
                dwell_time_seconds, track_quality, confidence_score)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (track_id,
             parameters.get('speed_avg'),
             parameters.get('speed_max'),
             parameters.get('speed_min'),
             parameters.get('entry_frames'),
             parameters.get('circulatory_frames'),
             parameters.get('exit_frames'),
             parameters.get('turning_radius'),
             parameters.get('path_length'),
             parameters.get('curvature_avg'),
             parameters.get('vehicle_class'),
             parameters.get('vehicle_category'),
             parameters.get('first_zone'),
             parameters.get('last_zone'),
             parameters.get('zone_sequence'),
             parameters.get('time_in_roundabout_seconds'),
             parameters.get('dwell_time_seconds'),
             parameters.get('track_quality'),
             parameters.get('confidence_score'))
        )
        self.connection.commit()
        return self.cursor.lastrowid
    
    def query_tracks_by_facility(self, facility_id: int, 
                                limit: int = 1000) -> List[Dict]:
        """Query tracks for a facility."""
        self.cursor.execute(
            """SELECT t.*, tp.speed_avg, tp.vehicle_class
               FROM tracks t
               LEFT JOIN track_parameters tp ON t.track_id = tp.track_id
               WHERE t.facility_id = ?
               LIMIT ?""",
            (facility_id, limit)
        )
        
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
    
    def get_aggregated_metrics(self, facility_id: int, 
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> List[Dict]:
        """Query aggregated metrics."""
        query = "SELECT * FROM aggregated_metrics WHERE facility_id = ?"
        params = [facility_id]
        
        if start_date:
            query += " AND time_period_start >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND time_period_end <= ?"
            params.append(end_date)
        
        self.cursor.execute(query, params)
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class DataExporter:
    """Export data from SQLite to various formats."""
    
    @staticmethod
    def export_to_csv(db_manager: DatabaseManager, facility_id: int, 
                     output_path: str):
        """Export facility data to CSV."""
        import csv
        
        # Get all tracks with parameters
        db_manager.cursor.execute(
            """SELECT t.track_id, t.external_track_id, t.class_name,
                      t.start_frame, t.end_frame, tp.speed_avg, tp.speed_max,
                      tp.turning_radius, tp.vehicle_class, tp.first_zone,
                      tp.last_zone, tp.time_in_roundabout_seconds
               FROM tracks t
               LEFT JOIN track_parameters tp ON t.track_id = tp.track_id
               WHERE t.facility_id = ?""",
            (facility_id,)
        )
        
        rows = db_manager.cursor.fetchall()
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'track_id', 'external_id', 'class', 'start_frame', 'end_frame',
                'speed_avg', 'speed_max', 'turning_radius', 'vehicle_class',
                'first_zone', 'last_zone', 'time_seconds'
            ])
            writer.writerows(rows)
        
        logger.info(f"Exported {len(rows)} tracks to {output_path}")
    
    @staticmethod
    def export_to_json(db_manager: DatabaseManager, facility_id: int,
                      output_path: str):
        """Export facility data to JSON."""
        db_manager.cursor.execute(
            """SELECT t.*, tp.* FROM tracks t
               LEFT JOIN track_parameters tp ON t.track_id = tp.track_id
               WHERE t.facility_id = ?""",
            (facility_id,)
        )
        
        columns = [desc[0] for desc in db_manager.cursor.description]
        rows = [dict(zip(columns, row)) for row in db_manager.cursor.fetchall()]
        
        with open(output_path, 'w') as f:
            json.dump(rows, f, indent=2, default=str)
        
        logger.info(f"Exported {len(rows)} tracks to {output_path}")
