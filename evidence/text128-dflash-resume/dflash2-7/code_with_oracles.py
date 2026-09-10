def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    """Merge overlapping or touching closed integer intervals.

    Given a list of closed intervals [start, end] with start <= end, return a
    new list of intervals sorted by start, where any intervals that overlap
    or touch (e.g., [1, 3] and [3, 5]) are merged into a single interval.

    The input list is not mutated, and the returned intervals are new list
    objects (not aliases of the input's inner lists).

    Args:
        intervals: A list of [start, end] pairs, each with start <= end.

    Returns:
        A new list of merged, non-overlapping intervals sorted by start.
        Returns [] for empty input.
    """
    # Handle the empty input case up front.
    if not intervals:
        return []

    # Sort a shallow copy of the input by start value so the original list
    # is never mutated. Sorting by start guarantees that any interval that
    # overlaps a previous one must be adjacent in this order.
    sorted_intervals = sorted(intervals, key=lambda interval: interval[0])

    # Build the result with fresh list objects so the output never shares
    # inner lists with the input.
    merged: list[list[int]] = [[start, end] for start, end in sorted_intervals]

    # Walk through the sorted intervals, extending the last merged interval
    # whenever the current one overlaps or touches it.
    write_index = 0
    for read_index in range(1, len(merged)):
        current_start, current_end = merged[read_index]
        last_end = merged[write_index][1]

        # Overlap or touch: current interval starts at or before the end of
        # the last merged interval, so extend that interval's end if needed.
        if current_start <= last_end:
            if current_end > last_end:
                merged[write_index][1] = current_end
        else:
            # No overlap: start a new merged interval at this position.
            write_index += 1
            merged[write_index] = merged[read_index]

    # Keep only the first write_index + 1 slots, which hold the merged result.
    return merged[: write_index + 1]

import random,copy
assert merge_intervals([])==[]
assert merge_intervals([[1,3],[2,6],[8,10],[10,18]])==[[1,6],[8,18]]
assert merge_intervals([[5,5],[-4,-1],[-2,3],[4,6]])==[[-4,3],[4,6]]
def oracle(rows):
    answer=[]
    for a,b in sorted(rows):
        if answer and a<=answer[-1][1]:answer[-1][1]=max(answer[-1][1],b)
        else:answer.append([a,b])
    return answer
rng=random.Random(827)
for _ in range(200):
    values=[sorted([rng.randint(-100,100),rng.randint(-100,100)]) for _ in range(rng.randrange(30))]
    original=copy.deepcopy(values)
    assert merge_intervals(values)==oracle(values)
    assert values==original
print('PASS: fixed cases, 200 generated cases, nonmutation')
