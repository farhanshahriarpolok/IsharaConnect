import cv2
import time
import sys

def probe_cameras():
    print("--- IsharaConnect Camera Diagnostic Tool ---")
    print("Probing available camera indices (0-4) across multiple backends...")
    
    backends = [
        ("cv2.CAP_DSHOW", cv2.CAP_DSHOW),
        ("cv2.CAP_MSMF", cv2.CAP_MSMF),
        ("cv2.CAP_ANY", cv2.CAP_ANY)
    ]
    
    working_cameras = []
    
    for idx in range(5):
        for backend_name, backend_id in backends:
            print(f"Probing index {idx} with {backend_name}...")
            cap = cv2.VideoCapture(idx, backend_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    backend = cap.getBackendName()
                    
                    print(f"  [SUCCESS] Camera {idx} ({backend_name} / {backend}) -> {width}x{height} @ {fps} FPS")
                    working_cameras.append({
                        "index": idx,
                        "backend_id": backend_id,
                        "backend_name": backend_name,
                        "desc": f"Index {idx} ({backend_name})"
                    })
                else:
                    print(f"  [FAILED] Camera {idx} opened but failed to read frame.")
            cap.release()
            
    if not working_cameras:
        print("\n[ERROR] No working cameras found on this system.")
        sys.exit(1)
        
    print(f"\nFound {len(working_cameras)} working camera configuration(s).")
    
    # Test the first working camera
    first_cam = working_cameras[0]
    print(f"\nStarting 3-second visual test on {first_cam['desc']}...")
    
    cap = cv2.VideoCapture(first_cam['index'], first_cam['backend_id'])
    if not cap.isOpened():
        print("Failed to re-open camera for visual test.")
        sys.exit(1)
        
    start_time = time.time()
    frames_read = 0
    
    while time.time() - start_time < 3.0:
        ret, frame = cap.read()
        if ret:
            frames_read += 1
            cv2.putText(frame, f"{first_cam['desc']} Test", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Closing in {3.0 - (time.time() - start_time):.1f}s", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("IsharaConnect Camera Diagnostic", frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print(f"Visual test complete. Read {frames_read} frames.")
    print("Diagnostics finished successfully.")

if __name__ == "__main__":
    probe_cameras()
