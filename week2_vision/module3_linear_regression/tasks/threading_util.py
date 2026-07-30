import threading


class PausableThread(threading.Thread):
    def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None):
        # Forward all standard arguments to the base Thread constructor
        super().__init__(target=target, args=args, kwargs=kwargs, daemon=daemon, name=name)
        self._can_run = threading.Event()
        self._can_run.set()  # Start in unpaused state
        self._stop_event = threading.Event()

    def run(self):
        """Overrides the thread's execution target wrapper."""
        # If no target function was passed, fall back to standard behavior
        if self._target is None:
            return

        # Execute the target loop while checking for pause/stop states
        while not self._stop_event.is_set():
            # Blocks here if pause() was called
            self._can_run.wait()

            # Safely check stop event again immediately after unpausing
            if self._stop_event.is_set():
                break

            # Execute one iteration of the target function
            # Note: The target function must execute quickly so it can check the pause flag frequently
            self._target(*self._args, **self._kwargs)

    def pause(self):
        self._can_run.clear()

    def resume(self):
        self._can_run.set()

    def is_paused(self):
        return not self._can_run.is_set()

    def stop(self):
        self._stop_event.set()
