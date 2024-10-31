import argparse

from .misc import load_json


def find_frame_indices(json_file, start_time: float = None, duration: float = None):
    """
    Calculate the start and end frame indices based on start time and duration.

    Parameters:
    json_file (str): Path to json file.
    start_time (float): The start time in minutes.
    duration (float): The total duration in minutes.

    Returns:
    tuple: Start frame index and end frame index.

    Raises:
    ValueError: If the start time or duration is invalid.
    """

    # Load JSON data
    data = load_json(json_file)
    start_times = [float(round(st)) for st in data["FrameTimesStart"]]
    durations = data["FrameDuration"]
    total_duration = float(sum(durations))
    end_times = [float(s + d) for s, d in zip(start_times, durations)]
    if start_time is None:
        start_time = 0.0
    if duration is None:
        duration = total_duration

    if start_time < start_times[0] or start_time > start_times[-1]:
        raise ValueError("Start time is out of bounds.")

    if duration < durations[0] or duration > total_duration:
        raise ValueError("Duration is out of bounds.")

    # Handle the case where there is only one frame
    if len(start_times) == 1:
        if start_time == start_times[0] and duration == total_duration:
            return 0, 0  # Start and end frame are the same for single-frame data
        else:
            raise ValueError(
                "The specified time range does not fit within the single frame."
            )

    # If no specific frames are found, return first and last frame
    if start_time:
        try:
            start_frame = start_times.index(start_time)
        except ValueError:
            print(f"Start time of {start_time} is not valid")
    else:
        start_frame = 0
        start_time = start_times[0]

    if duration:
        try:
            end_frame = end_times.index(start_time + duration)
        except ValueError:
            print(f"Duration {duration} is not valid")
    else:
        end_frame = len(end_times) - 1

    return start_frame, end_frame


def main():
    """
    Main function to handle argument parsing and computation of start/end frames.
    """
    parser = argparse.ArgumentParser(
        description="Compute the start and end frame indices based on input time and duration."
    )

    parser.add_argument(
        "--json_file",
        type=str,
        help="Path to the JSON file containing frame times.",
        required=True,
    )
    parser.add_argument(
        "--start_time",
        type=float,
        default=None,
        help="Start time in minutes.",
        required=False,
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration in minutes.",
        required=False,
    )

    args = parser.parse_args()

    try:
        # Calculate start and end frame
        start_frame, end_frame = find_frame_indices(
            args.json_file, args.start_time, args.duration
        )

        print(f"Start Frame: {start_frame}")
        print(f"End Frame: {end_frame}")

    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
