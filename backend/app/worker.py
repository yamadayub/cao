"""
Generation worker - processes jobs from the queue.

This worker polls the database for queued jobs and processes them
using the generation pipeline.

Usage:
    python -m app.worker

Or run in the background:
    python -m app.worker &
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from typing import Any, Dict, Optional

# Configure logging before imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

# Reduce noise from libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class Worker:
    """Generation job worker."""

    def __init__(
        self,
        poll_interval: float = 2.0,
        max_errors: int = 5,
    ):
        """
        Initialize worker.

        Args:
            poll_interval: Seconds between queue polls
            max_errors: Max consecutive errors before stopping
        """
        self.poll_interval = poll_interval
        self.max_errors = max_errors
        self.running = False
        self.error_count = 0

        # Lazy imports to avoid circular deps
        self._job_service = None
        self._pipeline = None

    @property
    def job_service(self):
        """Lazy load job service."""
        if self._job_service is None:
            from app.services.job_queue import get_job_queue_service

            self._job_service = get_job_queue_service()
        return self._job_service

    @property
    def pipeline(self):
        """Lazy load pipeline."""
        if self._pipeline is None:
            from app.services.pipeline import get_generation_pipeline

            self._pipeline = get_generation_pipeline()
        return self._pipeline

    def start(self):
        """Start the worker loop."""
        logger.info("Starting generation worker")
        logger.info(f"Poll interval: {self.poll_interval}s")

        self.running = True
        self.error_count = 0

        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        while self.running:
            try:
                self._process_next_job()
                self.error_count = 0  # Reset on success
            except Exception as e:
                self.error_count += 1
                logger.error(f"Worker error ({self.error_count}/{self.max_errors}): {e}")

                if self.error_count >= self.max_errors:
                    logger.critical("Max errors reached, stopping worker")
                    self.running = False
                    break

            if self.running:
                time.sleep(self.poll_interval)

        logger.info("Worker stopped")

    def stop(self):
        """Stop the worker loop."""
        logger.info("Stopping worker...")
        self.running = False

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    def _process_next_job(self):
        """Claim and process the next available job."""
        job = self.job_service.claim_next_job()

        if job is None:
            logger.debug("No jobs in queue")
            return

        job_id = job["id"]
        logger.info(f"Processing job {job_id} (mode={job['mode']})")

        try:
            self._execute_job(job)
        except Exception as e:
            logger.exception(f"Job {job_id} failed: {e}")
            self._fail_job(job_id, str(e))

    def _execute_job(self, job: Dict[str, Any]):
        """Execute a single job."""
        job_id = job["id"]

        def progress_callback(progress: int, message: str):
            """Update job progress."""
            logger.debug(f"Job {job_id}: {progress}% - {message}")
            try:
                self.job_service.update_job_status(
                    job_id,
                    status="running",
                    progress=progress,
                )
            except Exception as e:
                logger.warning(f"Failed to update progress: {e}")

        # Run pipeline
        result = self.pipeline.generate(
            base_image_data=job["base_image_path"],
            target_image_data=job["target_image_path"],
            mode=job["mode"],
            parts=job.get("parts"),
            strength=job.get("strength", 0.5),
            seed=job.get("seed"),
            progress_callback=progress_callback,
        )

        if result.success:
            logger.info(f"Job {job_id} succeeded")
            self.job_service.update_job_status(
                job_id,
                status="succeeded",
                progress=100,
                result_image_path=result.image_base64,
            )
        else:
            logger.error(f"Job {job_id} failed: {result.error}")
            self._fail_job(job_id, result.error or "Unknown error")

    def _fail_job(self, job_id: str, error: str):
        """Mark a job as failed."""
        # Truncate error message if too long
        max_error_len = 500
        if len(error) > max_error_len:
            error = error[:max_error_len] + "..."

        try:
            self.job_service.update_job_status(
                job_id,
                status="failed",
                error=error,
            )
        except Exception as e:
            logger.error(f"Failed to update job status: {e}")


def main():
    """Entry point for worker CLI."""
    parser = argparse.ArgumentParser(description="Generation job worker")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between queue polls (default: 2.0)",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=5,
        help="Max consecutive errors before stopping (default: 5)",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Process one job and exit (for testing)",
    )

    args = parser.parse_args()

    worker = Worker(
        poll_interval=args.poll_interval,
        max_errors=args.max_errors,
    )

    if args.single:
        # Process one job and exit
        logger.info("Single-job mode")
        try:
            worker._process_next_job()
        except Exception as e:
            logger.error(f"Job failed: {e}")
            sys.exit(1)
    else:
        worker.start()


if __name__ == "__main__":
    main()
