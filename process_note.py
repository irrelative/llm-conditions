from openai import OpenAI
from pydantic import BaseModel, create_model
from typing import Type, List, Optional

client = OpenAI()

class ConditionDetail(BaseModel):
    present: bool
    explanation: str

def create_patient_condition_model(conditions: List[str]) -> Type[BaseModel]:
    fields = {}
    for condition in conditions:
        fields[condition] = (Optional[ConditionDetail], None)
    
    return create_model('PatientConditions', **fields)

def process_medical_note(note_content):
    resp_format = create_patient_condition_model([
        "chest pain",
        "lightheadedness",
        "nausea",
        "diaphoresis",
        "palpitations",
        "stroke",
        "diabetes"
    ])
    
    response = client.beta.chat.completions.parse(
        model="gpt-4o",  # Use an appropriate model
        messages=[
            {"role": "system", "content": "You are a medical assistant that analyzes patient notes and extracts structured information. Extract conditions mentioned in the note with a sentiment value ('positive' for resolved/improving, 'negative' for ongoing/worsening, 'neutral' for stable, 'unknown' for unclear). Return as JSON with a 'conditions' array containing objects with 'condition', 'sentiment', and 'explanation' fields."},
            {"role": "user", "content": note_content}
        ],
        response_format=resp_format,
    )
    return response.choices[0].message.parsed


if __name__ == "__main__":
    note_content = open("note.txt", "r").read()
    result = process_medical_note(note_content)
    print(result.model_dump_json(indent=2))

