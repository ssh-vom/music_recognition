"""Cluster notes that share a horizontal attack (same vertical slice)."""

from schema import Note

EVENT_X_TOLERANCE_PX = 5


def group_notes_into_events(notes: list[Note], x_tol: int) -> list[list[Note]]:
    ordered = sorted(notes, key=lambda note: (note.center_x, note.center_y))
    events: list[list[Note]] = []

    for note in ordered:
        if not events:
            events.append([note])
            continue
        last_event = events[-1]
        anchor_x = round(sum(n.center_x for n in last_event) / float(len(last_event)))
        if abs(note.center_x - anchor_x) <= x_tol:
            last_event.append(note)
        else:
            events.append([note])

    return events
