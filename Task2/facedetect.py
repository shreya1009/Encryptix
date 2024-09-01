import cv2

# Load the pre-trained Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces_in_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image")
        return
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow('Faces', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_faces_in_video(video_path=None):
    if video_path is None:
        cap = cv2.VideoCapture(0)  # Use the webcam
    else:
        cap = cv2.VideoCapture(video_path)  # Use the provided video file

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow('Faces', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    choice = input("Enter 'video' to use webcam, 'upload' to upload a video file, or 'image' to provide an image path: ").strip().lower()
    if choice == 'image':
        image_path = input("Enter the path to the image (e.g., C:\\Users\\Mohamed Benhasan\\Pictures\\image1.jpg): ").strip()
        detect_faces_in_image(image_path)
    elif choice == 'video':
        detect_faces_in_video()
    elif choice == 'upload':
        video_path = input("Enter the path to the video file (e.g., C:\\Users\\Mohamed Benhasan\\Videos\\video1.mp4): ").strip()
        detect_faces_in_video(video_path)
    else:
        print("Invalid choice. Please enter 'video', 'upload', or 'image'.")

if __name__ == '__main__':
    main()
