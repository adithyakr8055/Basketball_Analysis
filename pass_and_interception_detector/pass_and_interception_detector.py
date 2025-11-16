# pass_and_interception_detector.py
from copy import deepcopy
from typing import List, Dict, Optional

class PassAndInterceptionDetector:
    """
    Detects passes between teammates and interceptions by opponents.

    This class analyzes a timeline of ball possession (player IDs or -1 for no possession)
    and a timeline of player team assignments (list of dicts mapping player_id -> team_id)
    and returns per-frame labels indicating whether a pass or interception occurred at that frame.

    Return encoding:
        -1 : no event
         1 : Team 1 (pass/interception)
         2 : Team 2 (pass/interception)
    """

    def __init__(self, debug: bool = False):
        """
        Args:
            debug: If True, prints helpful debugging info while processing.
        """
        self.debug = debug

    def _safe_get_team(self, player_assignment: List[Optional[Dict[int, int]]], frame_idx: int, player_id: int) -> int:
        """
        Helper: safely get the team for a player at a frame; returns -1 if unknown.
        """
        if player_assignment is None:
            return -1
        if frame_idx < 0 or frame_idx >= len(player_assignment):
            return -1
        frame_assign = player_assignment[frame_idx]
        if not isinstance(frame_assign, dict):
            return -1
        return frame_assign.get(player_id, -1)

    def detect_passes(self, ball_acquisition: List[int], player_assignment: List[Optional[Dict[int, int]]]) -> List[int]:
        """
        Detect successful passes between players of the same team.

        A pass is recorded on the frame where possession changes from one player to
        another player on the *same* team.

        Args:
            ball_acquisition: list of length N where each element is player_id (int) or -1 for no possession.
            player_assignment: list of length N (or less) where each element is dict mapping player_id -> team_id.

        Returns:
            List[int] of length N where each element is -1 (no pass) or team id (1 or 2) for a pass at that frame.
        """
        n = len(ball_acquisition)
        passes = [-1] * n

        # We will track the "last seen holder" and the frame index where they were last seen.
        last_holder = -1
        last_holder_frame = -1

        for frame in range(n):
            current_holder = ball_acquisition[frame]

            # Update last_holder if the previous frame had a holder
            if frame > 0 and ball_acquisition[frame - 1] != -1:
                last_holder = ball_acquisition[frame - 1]
                last_holder_frame = frame - 1

            # If possession changed and both holders valid, check teams
            if last_holder != -1 and current_holder != -1 and last_holder != current_holder:
                # get teams safely
                prev_team = self._safe_get_team(player_assignment, last_holder_frame, last_holder)
                curr_team = self._safe_get_team(player_assignment, frame, current_holder)

                if self.debug:
                    print(f"[DBG Pass] frame={frame} prev_holder={last_holder} (frame {last_holder_frame}) prev_team={prev_team} -> current_holder={current_holder} curr_team={curr_team}")

                # Pass occurs when both players belong to same team and team known
                if prev_team != -1 and prev_team == curr_team:
                    passes[frame] = prev_team
                # otherwise leave as -1

        return passes

    def detect_interceptions(self, ball_acquisition: List[int], player_assignment: List[Optional[Dict[int, int]]]) -> List[int]:
        """
        Detect interceptions where possession changes between opposing teams.

        An interception is recorded on the frame where possession changes from a player
        on one team to a player on the other team.

        Args and return are same shape/meaning as detect_passes.
        """
        n = len(ball_acquisition)
        interceptions = [-1] * n

        last_holder = -1
        last_holder_frame = -1

        for frame in range(n):
            current_holder = ball_acquisition[frame]

            # Update last_holder from previous frame when available
            if frame > 0 and ball_acquisition[frame - 1] != -1:
                last_holder = ball_acquisition[frame - 1]
                last_holder_frame = frame - 1

            if last_holder != -1 and current_holder != -1 and last_holder != current_holder:
                prev_team = self._safe_get_team(player_assignment, last_holder_frame, last_holder)
                curr_team = self._safe_get_team(player_assignment, frame, current_holder)

                if self.debug:
                    print(f"[DBG Intercept] frame={frame} prev_holder={last_holder} (frame {last_holder_frame}) prev_team={prev_team} -> current_holder={current_holder} curr_team={curr_team}")

                # Interception occurs when teams are different and both known
                if prev_team != -1 and curr_team != -1 and prev_team != curr_team:
                    interceptions[frame] = curr_team

        return interceptions


# quick local test when run directly
if __name__ == "__main__":
    # small smoke test
    detector = PassAndInterceptionDetector(debug=True)
    # frames: -1 means no possession, numbers are player ids
    ba = [-1, 10, 10, 11, 11, -1, 12, 13, 13]
    # player_assignment: for each frame, map player id -> team_id
    pa = [
        {},  # frame 0
        {10:1},  # 1
        {10:1},  # 2
        {11:1},  # 3 - same team as 10 => pass expected at frame 3
        {11:1},  # 4
        {},      # 5
        {12:2},  # 6
        {13:1},  # 7 - different team => interception expected at frame 7
        {13:1}   # 8
    ]

    print("passes:", detector.detect_passes(ba, pa))
    print("interceptions:", detector.detect_interceptions(ba, pa))