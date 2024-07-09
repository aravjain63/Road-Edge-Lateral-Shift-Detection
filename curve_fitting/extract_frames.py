import cv2
import os

def FrameCapture(path, output_path):
    # Path to video file
    vidObj = cv2.VideoCapture(path)

    # Get the video's frame rate
    fps = vidObj.get(cv2.CAP_PROP_FPS)
    print(fps)

    # Calculate the frame interval for 0.05 seconds
    frame_interval = int(round(fps * 0.05))

    # Counter for saved frames
    saved_count = 0

    # Counter for total frames
    total_count = 0

    while True:
        # Extract frame
        success, image = vidObj.read()

        if not success:
            break

        if total_count % 5== 0:
            # Crop the image (top-left quarter)
            height = image.shape[0] // 2
            width = image.shape[1] // 2
            cropped_image = image[:height, :width]

            # Save the cropped frame
            cv2.imwrite(f"{output_path}/output{saved_count}.jpg", cropped_image)
            saved_count += 1

        total_count += 1

    # Release the video capture object
    vidObj.release()

    print(f"Total frames processed: {total_count}")
    print(f"Frames saved: {saved_count}")


