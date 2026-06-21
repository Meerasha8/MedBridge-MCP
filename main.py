import os
import math
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
MCP_SECRET_KEY = os.getenv("MCP_SECRET_KEY", "")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

app = FastAPI(
    title="MedBridge MCP Tool Server",
    description="FastAPI MCP tool server for MedBridge — multilingual AI healthcare platform",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

def verify_mcp_key(x_mcp_key: str = Header(..., alias="X-MCP-Key")):
    if x_mcp_key != MCP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid MCP secret key")
    return x_mcp_key


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def write_audit_log(actor_id: str, action: str, target_table: str, target_id: str, metadata: dict = {}):
    try:
        supabase.table("audit_logs").insert({
            "actor_id": actor_id,
            "action": action,
            "target_table": target_table,
            "target_id": target_id,
            "metadata": metadata,
            "created_at": now_utc().isoformat(),
        }).execute()
    except Exception as e:
        logger.warning(f"audit_log write failed: {e}")


def send_notification(user_id: str, notif_type: str, title: str, message: str):
    try:
        supabase.table("notifications").insert({
            "user_id": user_id,
            "type": notif_type,
            "title": title,
            "message": message,
            "is_read": False,
            "created_at": now_utc().isoformat(),
        }).execute()
    except Exception as e:
        logger.warning(f"notification write failed: {e}")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SearchDoctorsInput(BaseModel):
    symptoms: str
    language: str
    lat: float
    lng: float
    radius_km: int = 20
    specialization: Optional[str] = None


class GetDoctorProfileInput(BaseModel):
    doctor_id: str


class CheckAvailabilityInput(BaseModel):
    doctor_id: str
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")


class BookAppointmentInput(BaseModel):
    patient_id: str
    doctor_id: str
    scheduled_at: str
    symptoms_text: str
    symptoms_language: str
    is_urgent: bool = False


class GetPatientReportsInput(BaseModel):
    patient_id: str
    requesting_doctor_id: str


class GetDoctorRatingsInput(BaseModel):
    doctor_id: str


class SubmitReviewInput(BaseModel):
    appointment_id: str
    patient_id: str
    rating: int = Field(..., ge=1, le=5)
    feedback_text: str
    feedback_language: str


class CompleteAppointmentInput(BaseModel):
    appointment_id: str
    doctor_id: str
    notes: str


class UploadDocumentMetaInput(BaseModel):
    patient_id: str
    uploaded_by: str
    appointment_id: Optional[str] = None
    doc_type: str
    file_url: str
    file_name: str
    description: str


# ---------------------------------------------------------------------------
# Health + Tools discovery
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "MedBridge MCP Server", "timestamp": now_utc().isoformat()}


TOOL_SCHEMAS = {
    "search_doctors": {
        "description": "Search for verified doctors by location, symptoms and optional specialization.",
        "input": {
            "symptoms": "str", "language": "str", "lat": "float",
            "lng": "float", "radius_km": "int (default 20)",
            "specialization": "str (optional)"
        },
        "output": "list of doctor summaries with distance_km"
    },
    "get_doctor_profile": {
        "description": "Get full profile of a doctor by doctor_id.",
        "input": {"doctor_id": "str"},
        "output": "doctor profile dict"
    },
    "check_availability": {
        "description": "Return available 30-min slots for a doctor on a given date (9am-5pm).",
        "input": {"doctor_id": "str", "date": "YYYY-MM-DD"},
        "output": {"available_slots": "list[str]"}
    },
    "book_appointment": {
        "description": "Book an appointment, insert shared data and notifications.",
        "input": {
            "patient_id": "str", "doctor_id": "str",
            "scheduled_at": "ISO datetime str", "symptoms_text": "str",
            "symptoms_language": "str", "is_urgent": "bool"
        },
        "output": {"appointment_id": "str", "status": "str", "scheduled_at": "str", "doctor_name": "str"}
    },
    "get_patient_reports": {
        "description": "Return patient documents accessible by the requesting doctor (consent-gated).",
        "input": {"patient_id": "str", "requesting_doctor_id": "str"},
        "output": "list of document metadata"
    },
    "get_doctor_ratings": {
        "description": "Get average rating, total reviews, and 5 recent review summaries for a doctor.",
        "input": {"doctor_id": "str"},
        "output": {"avg_rating": "float", "total_reviews": "int", "recent_reviews": "list"}
    },
    "submit_review": {
        "description": "Submit a patient review for a completed appointment.",
        "input": {
            "appointment_id": "str", "patient_id": "str",
            "rating": "int 1-5", "feedback_text": "str", "feedback_language": "str"
        },
        "output": {"success": "bool"}
    },
    "complete_appointment": {
        "description": "Mark appointment as completed, update shared data TTL, notify patient.",
        "input": {"appointment_id": "str", "doctor_id": "str", "notes": "str"},
        "output": {"success": "bool"}
    },
    "cleanup_expired_shared_data": {
        "description": "Delete expired appointment_shared_data rows (cron job trigger).",
        "input": {},
        "output": {"deleted_count": "int"}
    },
    "upload_document_meta": {
        "description": "Insert document metadata record and audit log.",
        "input": {
            "patient_id": "str", "uploaded_by": "str",
            "appointment_id": "str (optional)", "doc_type": "str",
            "file_url": "str", "file_name": "str", "description": "str"
        },
        "output": {"document_id": "str"}
    },
}


@app.get("/tools")
def list_tools():
    return {
        "tools": [
            {"name": name, "endpoint": f"/tools/{name}", **schema}
            for name, schema in TOOL_SCHEMAS.items()
        ]
    }


# ---------------------------------------------------------------------------
# Tool 1 — search_doctors
# ---------------------------------------------------------------------------

@app.post("/tools/search_doctors")
def search_doctors(body: SearchDoctorsInput, _=Depends(verify_mcp_key)):
    query = supabase.table("doctors").select(
        "id, profile_id, specialization, clinic_name, clinic_address, "
        "lat, lng, languages_spoken, rating, consultation_fee, experience_years, "
        "profiles(full_name)"
    ).eq("is_verified", True)

    if body.specialization:
        query = query.ilike("specialization", f"%{body.specialization}%")

    res = query.execute()
    if not res.data:
        return []

    results = []
    for doc in res.data:
        if doc.get("lat") is None or doc.get("lng") is None:
            continue
        dist = haversine_km(body.lat, body.lng, doc["lat"], doc["lng"])
        if dist > body.radius_km:
            continue
        name = (doc.get("profiles") or {}).get("full_name", "Unknown")
        results.append({
            "doctor_id": doc["id"],
            "name": name,
            "specialization": doc.get("specialization"),
            "clinic_name": doc.get("clinic_name"),
            "clinic_address": doc.get("clinic_address"),
            "languages_spoken": doc.get("languages_spoken", []),
            "rating": doc.get("rating"),
            "consultation_fee": doc.get("consultation_fee"),
            "distance_km": round(dist, 2),
        })

    results.sort(key=lambda d: (d["distance_km"], -(d["rating"] or 0)))
    return results[:5]


# ---------------------------------------------------------------------------
# Tool 2 — get_doctor_profile
# ---------------------------------------------------------------------------

@app.post("/tools/get_doctor_profile")
def get_doctor_profile(body: GetDoctorProfileInput, _=Depends(verify_mcp_key)):
    res = supabase.table("doctors").select(
        "*, profiles(full_name, email, phone, avatar_url, language_preference)"
    ).eq("id", body.doctor_id).single().execute()

    if not res.data:
        raise HTTPException(status_code=404, detail="Doctor not found")

    doc = res.data
    profile = doc.pop("profiles", {}) or {}

    return {
        **doc,
        "full_name": profile.get("full_name"),
        "email": profile.get("email"),
        "phone": profile.get("phone"),
        "avatar_url": profile.get("avatar_url"),
        "language_preference": profile.get("language_preference"),
        "availability_hint": "Typically available Mon–Sat, 9 AM – 5 PM. Use check_availability for exact slots.",
    }


# ---------------------------------------------------------------------------
# Tool 3 — check_availability
# ---------------------------------------------------------------------------

@app.post("/tools/check_availability")
def check_availability(body: CheckAvailabilityInput, _=Depends(verify_mcp_key)):
    date_start = f"{body.date}T00:00:00+00:00"
    date_end = f"{body.date}T23:59:59+00:00"

    res = supabase.table("appointments").select("scheduled_at").eq(
        "doctor_id", body.doctor_id
    ).neq("status", "cancelled").gte("scheduled_at", date_start).lte(
        "scheduled_at", date_end
    ).execute()

    booked_times = set()
    for row in res.data or []:
        dt = datetime.fromisoformat(row["scheduled_at"].replace("Z", "+00:00"))
        booked_times.add(f"{dt.hour:02d}:{dt.minute:02d}")

    all_slots = []
    hour, minute = 9, 0
    while (hour, minute) < (17, 0):
        slot = f"{hour:02d}:{minute:02d}"
        if slot not in booked_times:
            all_slots.append(slot)
        minute += 30
        if minute == 60:
            minute = 0
            hour += 1

    return {"available_slots": all_slots}


# ---------------------------------------------------------------------------
# Tool 4 — book_appointment
# ---------------------------------------------------------------------------

@app.post("/tools/book_appointment")
def book_appointment(body: BookAppointmentInput, _=Depends(verify_mcp_key)):
    # Fetch doctor's profile_id for notification
    doc_res = supabase.table("doctors").select("profile_id, profiles(full_name)").eq(
        "id", body.doctor_id
    ).single().execute()
    if not doc_res.data:
        raise HTTPException(status_code=404, detail="Doctor not found")

    doctor_profile_id = doc_res.data["profile_id"]
    doctor_name = (doc_res.data.get("profiles") or {}).get("full_name", "Doctor")

    appt_res = supabase.table("appointments").insert({
        "patient_id": body.patient_id,
        "doctor_id": body.doctor_id,
        "scheduled_at": body.scheduled_at,
        "status": "pending",
        "symptoms_text": body.symptoms_text,
        "symptoms_language": body.symptoms_language,
        "is_urgent": body.is_urgent,
        "created_at": now_utc().isoformat(),
    }).execute()

    if not appt_res.data:
        raise HTTPException(status_code=500, detail="Failed to create appointment")

    appt = appt_res.data[0]
    appt_id = appt["id"]

    # Fetch patient contact info
    pat_res = supabase.table("patients").select(
        "emergency_contact_name, emergency_contact_phone, profiles(full_name, phone, email)"
    ).eq("id", body.patient_id).single().execute()
    patient_contact = {}
    patient_profile_id = None
    if pat_res.data:
        p = pat_res.data
        profile = p.get("profiles") or {}
        patient_contact = {
            "name": profile.get("full_name"),
            "phone": profile.get("phone"),
            "email": profile.get("email"),
        }
        # get patient's profile_id for audit
        prof_res = supabase.table("profiles").select("id").eq(
            "id", profile.get("id", "")
        ).execute()

    supabase.table("appointment_shared_data").insert({
        "appointment_id": appt_id,
        "patient_contact": patient_contact,
        "symptom_summary": body.symptoms_text,
        "shared_at": now_utc().isoformat(),
        "delete_after": None,  # set when appointment completes
    }).execute()

    send_notification(
        doctor_profile_id,
        "new_appointment",
        "New Appointment Request",
        f"You have a new {'URGENT ' if body.is_urgent else ''}appointment request scheduled at {body.scheduled_at}.",
    )

    write_audit_log(
        actor_id=body.patient_id,
        action="book_appointment",
        target_table="appointments",
        target_id=appt_id,
        metadata={"doctor_id": body.doctor_id, "is_urgent": body.is_urgent},
    )

    return {
        "appointment_id": appt_id,
        "status": "pending",
        "scheduled_at": body.scheduled_at,
        "doctor_name": doctor_name,
    }


# ---------------------------------------------------------------------------
# Tool 5 — get_patient_reports
# ---------------------------------------------------------------------------

@app.post("/tools/get_patient_reports")
def get_patient_reports(body: GetPatientReportsInput, _=Depends(verify_mcp_key)):
    # Documents via explicit access grant
    access_res = supabase.table("document_access").select(
        "document_id"
    ).eq("granted_to_doctor_id", body.requesting_doctor_id).is_("revoked_at", "null").execute()

    granted_doc_ids = {row["document_id"] for row in (access_res.data or [])}

    # Appointments this doctor has with this patient
    appt_res = supabase.table("appointments").select("id").eq(
        "doctor_id", body.requesting_doctor_id
    ).eq("patient_id", body.patient_id).execute()
    doctor_appt_ids = {row["id"] for row in (appt_res.data or [])}

    # All documents for patient
    docs_res = supabase.table("documents").select("*").eq(
        "patient_id", body.patient_id
    ).execute()

    accessible = []
    for doc in docs_res.data or []:
        if doc["id"] in granted_doc_ids:
            accessible.append(doc)
        elif doc.get("appointment_id") and doc["appointment_id"] in doctor_appt_ids:
            accessible.append(doc)

    return [
        {
            "doc_id": d["id"],
            "doc_type": d.get("doc_type"),
            "file_name": d.get("file_name"),
            "file_url": d.get("file_url"),
            "description": d.get("description"),
            "created_at": d.get("created_at"),
        }
        for d in accessible
    ]


# ---------------------------------------------------------------------------
# Tool 6 — get_doctor_ratings
# ---------------------------------------------------------------------------

@app.post("/tools/get_doctor_ratings")
def get_doctor_ratings(body: GetDoctorRatingsInput, _=Depends(verify_mcp_key)):
    res = supabase.table("reviews").select("*").eq(
        "doctor_id", body.doctor_id
    ).order("created_at", desc=True).execute()

    reviews = res.data or []
    total = len(reviews)
    avg = round(sum(r["rating"] for r in reviews) / total, 2) if total else 0.0

    recent = [
        {
            "rating": r["rating"],
            "feedback": r.get("feedback_text"),
            "language": r.get("feedback_language"),
            "created_at": r.get("created_at"),
        }
        for r in reviews[:5]
    ]

    return {"avg_rating": avg, "total_reviews": total, "recent_reviews": recent}


# ---------------------------------------------------------------------------
# Tool 7 — submit_review
# ---------------------------------------------------------------------------

@app.post("/tools/submit_review")
def submit_review(body: SubmitReviewInput, _=Depends(verify_mcp_key)):
    appt_res = supabase.table("appointments").select("status, doctor_id").eq(
        "id", body.appointment_id
    ).single().execute()

    if not appt_res.data:
        raise HTTPException(status_code=404, detail="Appointment not found")
    if appt_res.data["status"] != "completed":
        raise HTTPException(status_code=400, detail="Can only review completed appointments")

    doctor_id = appt_res.data["doctor_id"]

    supabase.table("reviews").insert({
        "appointment_id": body.appointment_id,
        "patient_id": body.patient_id,
        "doctor_id": doctor_id,
        "rating": body.rating,
        "feedback_text": body.feedback_text,
        "feedback_language": body.feedback_language,
        "created_at": now_utc().isoformat(),
    }).execute()

    # Recalculate doctor rating
    all_reviews = supabase.table("reviews").select("rating").eq(
        "doctor_id", doctor_id
    ).execute()
    ratings = [r["rating"] for r in (all_reviews.data or [])]
    new_avg = round(sum(ratings) / len(ratings), 2) if ratings else body.rating

    supabase.table("doctors").update({
        "rating": new_avg,
        "total_reviews": len(ratings),
    }).eq("id", doctor_id).execute()

    write_audit_log(
        actor_id=body.patient_id,
        action="submit_review",
        target_table="reviews",
        target_id=body.appointment_id,
        metadata={"rating": body.rating, "doctor_id": doctor_id},
    )

    return {"success": True}


# ---------------------------------------------------------------------------
# Tool 8 — complete_appointment
# ---------------------------------------------------------------------------

@app.post("/tools/complete_appointment")
def complete_appointment(body: CompleteAppointmentInput, _=Depends(verify_mcp_key)):
    appt_res = supabase.table("appointments").select(
        "id, patient_id, status"
    ).eq("id", body.appointment_id).eq("doctor_id", body.doctor_id).single().execute()

    if not appt_res.data:
        raise HTTPException(status_code=404, detail="Appointment not found or doctor mismatch")

    completed_at = now_utc()
    delete_after = completed_at + timedelta(hours=24)

    supabase.table("appointments").update({
        "status": "completed",
        "completed_at": completed_at.isoformat(),
        "notes": body.notes,
    }).eq("id", body.appointment_id).execute()

    supabase.table("appointment_shared_data").update({
        "delete_after": delete_after.isoformat(),
    }).eq("appointment_id", body.appointment_id).execute()

    patient_id = appt_res.data["patient_id"]
    pat_res = supabase.table("patients").select("profile_id").eq(
        "id", patient_id
    ).single().execute()
    if pat_res.data:
        send_notification(
            pat_res.data["profile_id"],
            "appointment_completed",
            "Appointment Completed",
            "Your appointment has been marked as completed. You can now submit a review.",
        )

    write_audit_log(
        actor_id=body.doctor_id,
        action="complete_appointment",
        target_table="appointments",
        target_id=body.appointment_id,
        metadata={"notes": body.notes},
    )

    return {"success": True}


# ---------------------------------------------------------------------------
# Tool 9 — cleanup_expired_shared_data
# ---------------------------------------------------------------------------

@app.post("/tools/cleanup_expired_shared_data")
def cleanup_expired_shared_data(_=Depends(verify_mcp_key)):
    now = now_utc().isoformat()

    expired_res = supabase.table("appointment_shared_data").select("id").lt(
        "delete_after", now
    ).execute()

    expired_ids = [row["id"] for row in (expired_res.data or [])]
    deleted_count = 0

    for eid in expired_ids:
        supabase.table("appointment_shared_data").delete().eq("id", eid).execute()
        deleted_count += 1
        write_audit_log(
            actor_id="system",
            action="cleanup_shared_data",
            target_table="appointment_shared_data",
            target_id=eid,
            metadata={"reason": "expired delete_after"},
        )

    return {"deleted_count": deleted_count}


# ---------------------------------------------------------------------------
# Tool 10 — upload_document_meta
# ---------------------------------------------------------------------------

@app.post("/tools/upload_document_meta")
def upload_document_meta(body: UploadDocumentMetaInput, _=Depends(verify_mcp_key)):
    payload = {
        "patient_id": body.patient_id,
        "uploaded_by": body.uploaded_by,
        "doc_type": body.doc_type,
        "file_url": body.file_url,
        "file_name": body.file_name,
        "description": body.description,
        "created_at": now_utc().isoformat(),
    }
    if body.appointment_id:
        payload["appointment_id"] = body.appointment_id

    res = supabase.table("documents").insert(payload).execute()
    if not res.data:
        raise HTTPException(status_code=500, detail="Failed to insert document")

    doc_id = res.data[0]["id"]

    write_audit_log(
        actor_id=body.uploaded_by,
        action="upload_document",
        target_table="documents",
        target_id=doc_id,
        metadata={"doc_type": body.doc_type, "file_name": body.file_name},
    )

    return {"document_id": doc_id}