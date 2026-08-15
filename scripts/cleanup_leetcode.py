"""Entrypoint forwarding to cron.cleanup_leetcode."""

from cron.cleanup_leetcode import cleanup_expired_leetcode_tasks

if __name__ == "__main__":
    cleanup_expired_leetcode_tasks()
