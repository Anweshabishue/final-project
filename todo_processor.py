import json
import re
from datetime import datetime, timezone
from bson.objectid import ObjectId
import groq

class TodoProcessor:
    def __init__(self, groq_api_key, collection=None):
        self.groq_api_key = groq_api_key
        self.collection = collection
        
    def process_text_with_llm(self, text):
        """Process the input text with GROQ LLM and return structured data"""
        client = groq.Client(api_key=self.groq_api_key)
        
        prompt = f"""
        Extract the following information from this text:
        - Person: Who is involved or responsible
        - Time: When this task needs to be done
        - Topic: What the task is about
        
        Input text: {text}
        
        Return the data in JSON format with keys 'person', 'time', and 'topic'.
        """
        
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract and parse the JSON from the LLM response
        try:
            response_text = response.choices[0].message.content
            # Try to extract JSON if it's wrapped in backticks or other text
            json_match = re.search(r'{.*}', response_text, re.DOTALL)
            if json_match:
                structured_data = json.loads(json_match.group(0))
            else:
                structured_data = json.loads(response_text)
                
            return structured_data
        except Exception as e:
            raise Exception(f"Failed to parse LLM response: {str(e)}")
            
    def process_and_save(self, text):
        """Process text with LLM and save structured data to MongoDB"""
        # First, process text with LLM
        structured_data = self.process_text_with_llm(text)
        
        # Add timestamp and original text
        structured_data["created_at"] = datetime.now(timezone.utc)
        structured_data["original_text"] = text  # Keep original text for reference
        
        # Save to MongoDB
        result = self.collection.insert_one(structured_data)
        
        # Return the ID and structured data
        return result.inserted_id, structured_data
        
    def delete_todo(self, todo_id):
        """Delete a todo by ID"""
        result = self.collection.delete_one({"_id": ObjectId(todo_id)})
        if result.deleted_count == 0:
            raise Exception("Todo not found")
        return True
        
    def modify_todo(self, todo_id, new_text):
        """Modify a todo with new text"""
        # Process the new text with LLM
        structured_data = self.process_text_with_llm(new_text)
        structured_data["updated_at"] = datetime.now(timezone.utc)
        structured_data["original_text"] = new_text
        
        # Update in MongoDB
        result = self.collection.update_one(
            {"_id": ObjectId(todo_id)},
            {"$set": structured_data}
        )
        
        if result.modified_count == 0:
            raise Exception("Todo not found or no changes made")
            
        # Return the updated document
        updated_doc = self.collection.find_one({"_id": ObjectId(todo_id)})
        return updated_doc