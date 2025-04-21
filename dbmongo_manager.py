from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime

class MongoDBManager:
    def __init__(self, mongo_uri=None):
        # Use provided URI or default to localhost
        self.mongo_uri = mongo_uri or "mongodb://localhost:27017/"
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client['question_papers_db']
        self.papers_collection = self.db['papers']
        self.patterns_collection = self.db['patterns']
        
    def save_paper(self, subject, syllabus, question_types, difficulty, pattern_id=None, content=None):
        """Save a question paper to MongoDB"""
        paper_data = {
            "subject": subject,
            "syllabus": syllabus,
            "question_types": question_types,
            "difficulty": difficulty,
            "pattern_id": pattern_id,
            "content": content,
            "created_at": datetime.utcnow()
        }
        
        result = self.papers_collection.insert_one(paper_data)
        return str(result.inserted_id)
        
    def get_paper(self, paper_id):
        """Retrieve a paper by its ID"""
        return self.papers_collection.find_one({"_id": ObjectId(paper_id)})
        
    def get_all_papers(self):
        """Get all papers from the database"""
        return list(self.papers_collection.find())
        
    def delete_paper(self, paper_id):
        """Delete a paper by ID"""
        result = self.papers_collection.delete_one({"_id": ObjectId(paper_id)})
        return result.deleted_count
        
    def save_pattern(self, name, description, structure):
        """Save a question paper pattern"""
        pattern_data = {
            "name": name,
            "description": description,
            "structure": structure,
            "created_at": datetime.utcnow()
        }
        result = self.patterns_collection.insert_one(pattern_data)
        return str(result.inserted_id)
        
    def get_pattern(self, pattern_id):
        """Get a pattern by ID"""
        return self.patterns_collection.find_one({"_id": ObjectId(pattern_id)})
        
    def get_all_patterns(self):
        """Get all patterns"""
        return list(self.patterns_collection.find())
