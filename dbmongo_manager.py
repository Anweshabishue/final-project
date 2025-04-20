import os
import hashlib
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# Third-party imports - ensure these are installed
from pymongo import MongoClient
from bson.objectid import ObjectId
from bson.binary import Binary
from rich.console import Console

# Constants
MODEL_NAME = "gemini-2.0-flash"  # Updated to use gemini-2.0-flash
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = "question_paper_generator"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("question_paper_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# Rich console for better terminal output
console = Console()

class MongoDBManager:
    def __init__(self, mongodb_uri: str = MONGODB_URI, db_name: str = DB_NAME):
        """Initialize MongoDB connection and ensure collections exist"""
        try:
            self.client = MongoClient(mongodb_uri)
            self.db = self.client[db_name]
            # Define collections
            self.patterns = self.db["patterns"]
            self.papers = self.db["papers"]
            self.feedback = self.db["feedback"]
            
            # Create indexes
            self.patterns.create_index("file_hash", unique=True)
            self.papers.create_index("created_at")
            self.feedback.create_index("paper_id")
            
            logger.info(f"Connected to MongoDB: {db_name}")
            console.print(f"[green]Connected to MongoDB: {db_name}[/green]")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            console.print(f"[bold red]Failed to connect to MongoDB: {str(e)}[/bold red]")
            raise ConnectionError(f"Failed to connect to MongoDB: {str(e)}")
    
    def save_pattern(self, name: str, file_content: bytes, text_content: str, analysis: Optional[str] = None) -> str:
        """Save a question paper template pattern to the database"""
        file_hash = hashlib.md5(file_content).hexdigest()
        
        # Check if pattern already exists
        existing = self.patterns.find_one({"file_hash": file_hash})
        
        if existing:
            pattern_id = str(existing["_id"])
            # Update the analysis if provided
            if analysis:
                self.patterns.update_one(
                    {"_id": ObjectId(pattern_id)},
                    {"$set": {"analysis": analysis}}
                )
            return pattern_id
        
        # Insert new pattern
        pattern_doc = {
            "name": name,
            "file_hash": file_hash,
            "file_content": Binary(file_content),
            "text_content": text_content,
            "analysis": analysis,
            "created_at": datetime.now()
        }
        
        result = self.patterns.insert_one(pattern_doc)
        return str(result.inserted_id)
    
    def get_patterns(self) -> List[Dict[str, Any]]:
        """Get all template patterns from the database"""
        patterns = self.patterns.find({}, {
            "_id": 1, 
            "name": 1, 
            "created_at": 1
        }).sort("created_at", -1)
        
        return [{
            "id": str(pattern["_id"]),
            "name": pattern["name"],
            "created_at": pattern["created_at"]
        } for pattern in patterns]
    
    def get_pattern(self, pattern_id: str) -> Dict[str, Any]:
        """Get a specific pattern by ID"""
        pattern = self.patterns.find_one({"_id": ObjectId(pattern_id)})
        if pattern:
            pattern["id"] = str(pattern["_id"])
            del pattern["_id"]
            return pattern
        return None
    
    def save_paper(self, subject: str, syllabus: str, question_types: Dict, 
                  difficulty: Dict, pattern_id: Optional[str], content: str) -> str:
        """Save a generated question paper to the database"""
        paper_doc = {
            "subject": subject,
            "syllabus": syllabus,
            "question_types": question_types,
            "difficulty": difficulty,
            "pattern_id": ObjectId(pattern_id) if pattern_id else None,
            "content": content,
            "created_at": datetime.now()
        }
        
        result = self.papers.insert_one(paper_doc)
        return str(result.inserted_id)
    
    def get_recent_papers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recently generated papers"""
        pipeline = [
            {"$sort": {"created_at": -1}},
            {"$limit": limit},
            {
                "$lookup": {
                    "from": "patterns",
                    "localField": "pattern_id",
                    "foreignField": "_id",
                    "as": "pattern"
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "id": {"$toString": "$_id"},
                    "subject": 1,
                    "created_at": 1,
                    "template_name": {"$arrayElemAt": ["$pattern.name", 0]}
                }
            }
        ]
        
        papers = list(self.papers.aggregate(pipeline))
        return [{
            "id": paper["id"],
            "subject": paper["subject"],
            "created_at": paper["created_at"],
            "template_name": paper.get("template_name", "Custom")
        } for paper in papers]
    
    def get_paper(self, paper_id: str) -> Dict[str, Any]:
        """Get a specific paper by ID"""
        try:
            paper = self.papers.find_one({"_id": ObjectId(paper_id)})
            if paper:
                paper["id"] = str(paper["_id"])
                del paper["_id"]
                if paper.get("pattern_id"):
                    paper["pattern_id"] = str(paper["pattern_id"])
                return paper
            return None
        except Exception as e:
            logger.error(f"Error retrieving paper: {str(e)}")
            return None
    
    def save_feedback(self, paper_id: str, rating: int, comments: Optional[str] = None) -> str:
        """Save user feedback for a generated paper"""
        feedback_doc = {
            "paper_id": ObjectId(paper_id),
            "rating": rating,
            "comments": comments,
            "created_at": datetime.now()
        }
        
        result = self.feedback.insert_one(feedback_doc)
        return str(result.inserted_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = {
            "total_patterns": self.patterns.count_documents({}),
            "total_papers": self.papers.count_documents({}),
            "total_feedback": self.feedback.count_documents({}),
            "avg_rating": 0
        }
        
        # Calculate average rating if there are feedback entries
        if stats["total_feedback"] > 0:
            ratings = list(self.feedback.aggregate([
                {"$group": {"_id": None, "avg": {"$avg": "$rating"}}}
            ]))
            if ratings and len(ratings) > 0:
                stats["avg_rating"] = round(ratings[0]["avg"], 1)
        
        return stats
    
    def delete_paper(self, paper_id: str) -> bool:
        """Delete a paper and its associated feedback"""
        try:
            # Delete feedback first
            self.feedback.delete_many({"paper_id": ObjectId(paper_id)})
            
            # Delete the paper
            result = self.papers.delete_one({"_id": ObjectId(paper_id)})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting paper: {str(e)}")
            return False
    
    def delete_pattern(self, pattern_id: str) -> bool:
        """Delete a pattern and update associated papers"""
        try:
            # Set pattern_id to None for associated papers
            self.papers.update_many(
                {"pattern_id": ObjectId(pattern_id)},
                {"$set": {"pattern_id": None}}
            )
            
            # Delete the pattern
            result = self.patterns.delete_one({"_id": ObjectId(pattern_id)})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting pattern: {str(e)}")
            return False

