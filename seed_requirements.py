from db_manager import insert_requirement, create_database

create_database() # يجب أن يتم استدعاء هذه الدالة أولاً

file_name = "smart_lock_requirements.txt"

requirements = [
    {
        "text": "The system shall allow the owner to update the door lock code through a secure mobile interface.",
        "functional": True,
        "ambiguous": False,
        "measurable": True,
        "notes": "Clear and measurable through app interaction."
    },
    {
        "text": "The tenant can open the smart lock using the assigned code.",
        "functional": True,
        "ambiguous": False,
        "measurable": True,
        "notes": "Straightforward functionality."
    },
    {
        "text": "The system must notify the owner when the door is opened.",
        "functional": True,
        "ambiguous": False,
        "measurable": True,
        "notes": "Can be measured by delivery of notification."
    },
    {
        "text": "The smart lock should open quickly.",
        "functional": True,
        "ambiguous": True,
        "measurable": False,
        "notes": "Ambiguous — 'quickly' needs definition (e.g., within 2 seconds)."
    },
    {
        "text": "The system shall record every access attempt.",
        "functional": True,
        "ambiguous": False,
        "measurable": True,
        "notes": "Log entries are countable and reviewable."
    },
    {
        "text": "The system must be secure.",
        "functional": False,
        "ambiguous": True,
        "measurable": False,
        "notes": "Very general — needs measurable criteria (e.g., encryption standard)."
    },
    {
        "text": "Only the owner can view access logs.",
        "functional": True,
        "ambiguous": False,
        "measurable": True,
        "notes": "Access control can be verified via roles."
    },
    {
        "text": "The system shall allow temporary access codes for guests.",
        "functional": True,
        "ambiguous": False,
        "measurable": True,
        "notes": "Measurable by code generation and expiration."
    },
    {
        "text": "The user experience should be smooth.",
        "functional": False,
        "ambiguous": True,
        "measurable": False,
        "notes": "Too subjective — needs user feedback metrics or clearer definition."
    }
]

for r in requirements:
    insert_requirement(
        requirement_text=r["text"],
        is_functional=r["functional"],
        is_ambiguous=r["ambiguous"],
        is_measurable=r["measurable"],
        notes=r["notes"],
        file_name=file_name
    )