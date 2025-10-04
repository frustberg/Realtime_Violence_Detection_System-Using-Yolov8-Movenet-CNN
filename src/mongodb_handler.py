from pymongo import MongoClient
from datetime import datetime
import traceback

class MongoHandler:
    def __init__(self, conn_str, db_name="violence_detection", collection_name="recorded_incidents"):
        self.conn_str = conn_str
        self.db = None
        self.collection = None
        self.client = None
        try:
            self.client = MongoClient(self.conn_str)
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            print("Successfully connected to MongoDB")
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            self.client = None
            self.db = None
            self.collection = None

    def insert_incident(self, incident_record):
        if self.collection is None:
            return False
        try:
            self.collection.insert_one(incident_record)
            return True
        except Exception as e:
            print(f"Error inserting incident: {e}")
            print(traceback.format_exc())
            return False

    def update_incident(self, incident_id, update_fields):
        if self.collection is None:
            return False
        try:
            self.collection.update_one({"incident_id": incident_id}, {"$set": update_fields})
            return True
        except Exception as e:
            print(f"Error updating incident: {e}")
            print(traceback.format_exc())
            return False

    def close(self):
        if self.client is not None:
            self.client.close()
            print("MongoDB connection closed")
