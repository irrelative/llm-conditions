import os
import json
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(
    # This will use the OPENAI_API_KEY environment variable
    # Make sure to set this in your environment
)

def read_note_content(file_path):
    """Read the content of the note file."""
    with open(file_path, 'r') as file:
        return file.read()

def process_medical_note(note_content):
    """Process the medical note using OpenAI API."""
    
    # Define the response format as JSON
    response_format = {
        "type": "json_object"
    }
    
    # Send the request to OpenAI
    response = client.chat.completions.create(
        model="gpt-4o",  # Use an appropriate model
        messages=[
            {"role": "system", "content": "You are a medical assistant that analyzes patient notes and extracts structured information. Extract conditions mentioned in the note with a boolean sentiment (true if positive/resolved, false if negative/ongoing) and a brief explanation. Return as JSON with a 'conditions' array containing objects with 'condition', 'sentiment', and 'explanation' fields."},
            {"role": "user", "content": note_content}
        ],
        response_format=response_format,
    )
    
    # Extract and return the JSON content
    return json.loads(response.choices[0].message.content)

def main():
    note_path = "note.txt"
    note_content = read_note_content(note_path)
    
    try:
        result = process_medical_note(note_content)
        
        # Pretty print the result
        print(json.dumps(result, indent=2))
        
        # Optionally save the result to a file
        with open("medical_analysis.json", "w") as f:
            json.dump(result, indent=2, fp=f)
        
        print("\nAnalysis complete! Results saved to medical_analysis.json")
        
    except Exception as e:
        print(f"Error processing the medical note: {e}")

if __name__ == "__main__":
    main()
