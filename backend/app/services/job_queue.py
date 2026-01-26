"""Job queue service for async generation jobs."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.services.supabase_client import get_supabase

logger = logging.getLogger(__name__)


class JobNotFoundError(Exception):
    """Raised when a job is not found."""

    pass


class JobQueueService:
    """Service for managing generation jobs in the database."""

    TABLE_NAME = "generation_jobs"

    def create_job(
        self,
        mode: str,
        base_image_path: str,
        target_image_path: str,
        parts: Optional[List[str]] = None,
        strength: float = 0.5,
        seed: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new generation job.

        Args:
            mode: 'morph' or 'parts'
            base_image_path: Base64 or storage path for base image
            target_image_path: Base64 or storage path for target image
            parts: List of parts to blend (for mode='parts')
            strength: Blend strength 0-1
            seed: Random seed
            user_id: Optional user ID

        Returns:
            Created job record
        """
        supabase = get_supabase()
        job_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        job_data = {
            "id": job_id,
            "mode": mode,
            "parts": parts or [],
            "strength": strength,
            "seed": seed,
            "base_image_path": base_image_path,
            "target_image_path": target_image_path,
            "status": "queued",
            "progress": 0,
            "user_id": user_id,
            "created_at": now,
            "updated_at": now,
        }

        result = supabase.table(self.TABLE_NAME).insert(job_data).execute()

        if not result.data:
            raise RuntimeError("Failed to create job")

        logger.info(f"Created job {job_id} with mode={mode}")
        return result.data[0]

    def get_job(self, job_id: str) -> Dict[str, Any]:
        """
        Get a job by ID.

        Args:
            job_id: Job UUID

        Returns:
            Job record

        Raises:
            JobNotFoundError: If job doesn't exist
        """
        supabase = get_supabase()

        result = (
            supabase.table(self.TABLE_NAME).select("*").eq("id", job_id).execute()
        )

        if not result.data:
            raise JobNotFoundError(f"Job {job_id} not found")

        return result.data[0]

    def update_job_status(
        self,
        job_id: str,
        status: str,
        progress: Optional[int] = None,
        error: Optional[str] = None,
        result_image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Update job status.

        Args:
            job_id: Job UUID
            status: New status
            progress: Progress percentage
            error: Error message (for failed status)
            result_image_path: Result image path (for succeeded status)

        Returns:
            Updated job record
        """
        supabase = get_supabase()

        update_data: Dict[str, Any] = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat(),
        }

        if progress is not None:
            update_data["progress"] = progress

        if error is not None:
            update_data["error"] = error

        if result_image_path is not None:
            update_data["result_image_path"] = result_image_path

        # Set timestamps based on status
        if status == "running":
            update_data["started_at"] = datetime.utcnow().isoformat()
        elif status in ("succeeded", "failed"):
            update_data["completed_at"] = datetime.utcnow().isoformat()
            if status == "succeeded":
                update_data["progress"] = 100

        result = (
            supabase.table(self.TABLE_NAME)
            .update(update_data)
            .eq("id", job_id)
            .execute()
        )

        if not result.data:
            raise JobNotFoundError(f"Job {job_id} not found")

        logger.info(f"Updated job {job_id}: status={status}")
        return result.data[0]

    def claim_next_job(self) -> Optional[Dict[str, Any]]:
        """
        Atomically claim the next queued job for processing.

        Uses optimistic locking to prevent race conditions.

        Returns:
            Job record if found, None otherwise
        """
        supabase = get_supabase()

        # Find oldest queued job
        result = (
            supabase.table(self.TABLE_NAME)
            .select("*")
            .eq("status", "queued")
            .order("created_at", desc=False)
            .limit(1)
            .execute()
        )

        if not result.data:
            return None

        job = result.data[0]
        job_id = job["id"]

        # Try to claim it (optimistic lock)
        update_result = (
            supabase.table(self.TABLE_NAME)
            .update(
                {
                    "status": "running",
                    "started_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat(),
                }
            )
            .eq("id", job_id)
            .eq("status", "queued")  # Only if still queued
            .execute()
        )

        if not update_result.data:
            # Another worker claimed it, try again
            logger.debug(f"Job {job_id} was claimed by another worker")
            return None

        logger.info(f"Claimed job {job_id}")
        return update_result.data[0]

    def get_user_jobs(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Get jobs for a user.

        Args:
            user_id: User ID
            limit: Max results
            offset: Pagination offset

        Returns:
            List of job records
        """
        supabase = get_supabase()

        result = (
            supabase.table(self.TABLE_NAME)
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )

        return result.data or []

    def get_pending_jobs_count(self) -> int:
        """Get count of queued/running jobs."""
        supabase = get_supabase()

        result = (
            supabase.table(self.TABLE_NAME)
            .select("id", count="exact")
            .in_("status", ["queued", "running"])
            .execute()
        )

        return result.count or 0


# Global instance
_job_queue_service: Optional[JobQueueService] = None


def get_job_queue_service() -> JobQueueService:
    """Get or create job queue service instance."""
    global _job_queue_service
    if _job_queue_service is None:
        _job_queue_service = JobQueueService()
    return _job_queue_service
