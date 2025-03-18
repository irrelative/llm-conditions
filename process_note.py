import json
from openai import OpenAI
from pydantic import BaseModel, Field, create_model
from enum import Enum, auto
from typing import Dict, Type, List, Optional

client = OpenAI()

class SentimentType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"

class Condition(BaseModel):
    condition: str
    sentiment: SentimentType
    explanation: str

class ConditionDetail(BaseModel):
    sentiment: SentimentType
    explanation: str

class MedicalAnalysis(BaseModel):
    conditions: list[Condition]

def create_patient_condition_model(conditions: List[str]) -> Type[BaseModel]:
    """
    Factory function that creates a Pydantic model with condition names as fields.
    
    Args:
        conditions: List of condition strings to use as field names
        
    Returns:
        A Pydantic model class with each condition as a field mapped to a ConditionDetail
    """
    fields = {}
    for condition in conditions:
        # Create a field for each condition, with Optional[ConditionDetail] as the type
        fields[condition] = (Optional[ConditionDetail], None)
    
    # Create and return the model class
    return create_model('PatientConditions', **fields)


def read_note_content(file_path):
    """Read the content of the note file."""
    with open(file_path, 'r') as file:
        return file.read()

def process_medical_note(note_content) -> MedicalAnalysis:
    """Process the medical note using OpenAI API and return a structured MedicalAnalysis."""
    
    response_format = {
        "type": "json_object"
    }
    
    # Send the request to OpenAI
    response = client.chat.completions.create(
        model="gpt-4o",  # Use an appropriate model
        messages=[
            {"role": "system", "content": "You are a medical assistant that analyzes patient notes and extracts structured information. Extract conditions mentioned in the note with a sentiment value ('positive' for resolved/improving, 'negative' for ongoing/worsening, 'neutral' for stable, 'unknown' for unclear). Return as JSON with a 'conditions' array containing objects with 'condition', 'sentiment', and 'explanation' fields."},
            {"role": "user", "content": note_content}
        ],
        response_format=response_format,
    )
    
    # Parse the response into our Pydantic model
    response_data = json.loads(response.choices[0].message.content)
    return MedicalAnalysis.model_validate(response_data)

def main():
    note_path = "note.txt"
    note_content = read_note_content(note_path)
    
    try:
        result = process_medical_note(note_content)
        
        # Pretty print the result
        print(result.model_dump_json(indent=2))
        
        # Optionally save the result to a file
        with open("medical_analysis.json", "w") as f:
            f.write(result.model_dump_json(indent=2))
        
        print("\nAnalysis complete! Results saved to medical_analysis.json")
        
    except Exception as e:
        print(f"Error processing the medical note: {e}")

if __name__ == "__main__":
    main()
