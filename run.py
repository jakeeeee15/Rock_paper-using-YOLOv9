import cv2
import random
import time
from ultralytics import YOLO


def determine_winner(player, computer):
    if player == computer:
        return "It's a Tie! 🤝"

    # Fully unified to match your 'rock' dataset class name
    winning_rules = {
        "rock": "scissor",
        "paper": "rock",
        "scissor": "paper",
        "scissors": "paper",  # Safety plural fallback
    }

    if winning_rules[player] == computer:
        return "You Win! 🎉"
    else:
        return "Computer Wins! 🤖"


def main():
    # Load custom YOLOv9 weights
    model_path = "runs/detect/train/weights/best.pt"
    model = YOLO(model_path)

    game_state = "WAITING"  # WAITING, COUNTDOWN, SHOW_RESULT
    countdown_start_time = 0
    countdown_duration = 3

    current_live_gesture = "None"
    player_move = None
    computer_move = None
    result_text = ""

    # Memory tracking variables
    last_valid_gesture = "None"
    last_seen_time = 0

    # Clean display mapping for the UI overlays
    display_names = {
        "rock": "ROCK",
        "stone": "ROCK",  # Fallback compatibility
        "paper": "PAPER",
        "scissors": "SCISSORS",
        "scissor": "SCISSORS",
    }

    cap = cv2.VideoCapture(0)

    print("=== Continuous Tracking Rock, Paper, Scissors ===")
    print("Press 'ENTER' to lock in a round.")
    print("Press 'q' in the video window to quit.")
    print("--------------------------------------------------")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Error: Failed to read from webcam.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        display_frame = frame.copy()
        current_time = time.time()

        # Always run inference on every frame loop
        results = model(frame, conf=0.3, verbose=False)

        current_live_gesture = "None"

        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                class_name = r.names[class_id]

                # Standardize 'stone' to 'rock' immediately if the model returns it
                if class_name.lower() == "stone":
                    class_name = "rock"

                current_live_gesture = class_name

                # Update memory cache with timestamp whenever a real hand signal is detected
                last_valid_gesture = class_name
                last_seen_time = current_time

                # Draw the green box live
                x1, y1, x2, y2 = [int(x) for x in box.xyxy[0]]
                conf = float(box.conf[0])
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name.upper()} {conf:.2f}"
                cv2.putText(
                    display_frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        # --- STATE 1: WAITING ---
        if game_state == "WAITING":
            cv2.putText(
                display_frame,
                "Press ENTER to Play",
                (w // 5, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3,
            )

        # --- STATE 2: COUNTDOWN ---
        elif game_state == "COUNTDOWN":
            elapsed = current_time - countdown_start_time
            remaining = int(countdown_duration - elapsed) + 1

            if remaining > 0:
                cv2.putText(
                    display_frame,
                    f"Show Hand in... {remaining}",
                    (w // 6, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 165, 255),
                    3,
                )
            else:
                # Timer runs out
                game_state = "SHOW_RESULT"

                # Unified computer random choice list
                computer_move = random.choice(["rock", "paper", "scissor"])

                # Verification check if the instant frame spots something
                if current_live_gesture != "None":
                    player_move = current_live_gesture
                    reason_string = "Instantaneous detection"
                else:
                    # Fallback lookup to past 1-second activity windows
                    time_difference = current_time - last_seen_time
                    if last_valid_gesture != "None" and time_difference <= 1.0:
                        player_move = last_valid_gesture
                        reason_string = (
                            f"Memory recall cache ({time_difference:.2f}s ago)"
                        )
                    else:
                        player_move = "None"
                        reason_string = "No detection in final window"

                print("\n⏰ TIMER UP! Choices locked.")
                print(f"🤖 Computer chose: {computer_move.upper()}")
                print(
                    f"👤 Player locked in: {player_move.upper()} ({reason_string})"
                )

                if player_move != "None":
                    result_text = determine_winner(
                        player_move.lower(), computer_move.lower()
                    )
                else:
                    result_text = "No hand detected! Try again."

                print(f"📊 Result: {result_text}\n")

        # --- STATE 3: SHOW RESULT ON SCREEN ---
        elif game_state == "SHOW_RESULT":
            p_display = display_names.get(player_move.lower(), "NONE")
            cv2.putText(
                display_frame,
                f"YOU: {p_display}",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            c_display = display_names.get(computer_move.lower(), "UNKNOWN")
            cv2.putText(
                display_frame,
                f"AI:  {c_display}",
                (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )

            text_color = (0, 255, 0) if "Win" in result_text else (0, 0, 255)
            if "Tie" in result_text:
                text_color = (255, 165, 0)

            cv2.putText(
                display_frame,
                result_text,
                (30, h - 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                text_color,
                3,
            )

            cv2.putText(
                display_frame,
                "Press ENTER to play again",
                (30, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2,
            )

        cv2.imshow("Stone Paper Scissors AI Player", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter Key
            if game_state == "WAITING" or game_state == "SHOW_RESULT":
                game_state = "COUNTDOWN"
                countdown_start_time = time.time()
                last_valid_gesture = "None"
                last_seen_time = 0
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()