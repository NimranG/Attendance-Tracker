import os
import dlib
import numpy as np
import cv2
from PIL import Image

images_folder = "images"
output_folder = "output_faces"
os.makedirs(output_folder, exist_ok=True)

detector = dlib.get_frontal_face_detector()

image_files = [f for f in os.listdir(images_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
print(f"Found {len(image_files)} images:")
for f in image_files:
    print(f"- {f}")

for img_file in image_files:
    image_path = os.path.join(images_folder, img_file)
    print(f"\n--- Processing {img_file} ---")
    
    try:
        # Debug: Check if file exists and is readable
        if not os.path.exists(image_path):
            print(f"ERROR: File does not exist: {image_path}")
            continue
        
        file_size = os.path.getsize(image_path)
        print(f"File size: {file_size} bytes")
        
        if file_size == 0:
            print(f"ERROR: File is empty (0 bytes)")
            continue
        
        # Try loading with PIL first (more robust)
        try:
            pil_img = Image.open(image_path)
            print(f"PIL loaded successfully: mode={pil_img.mode}, size={pil_img.size}")
            
            # Convert PIL to OpenCV format
            # Handle different modes (RGBA, L, etc.)
            if pil_img.mode == 'RGBA':
                pil_img = pil_img.convert('RGB')
            elif pil_img.mode == 'L':  # Grayscale
                pil_img = pil_img.convert('RGB')
            elif pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            # Convert PIL to numpy array (RGB)
            img_array = np.array(pil_img)
            print(f"Array shape: {img_array.shape}, dtype: {img_array.dtype}")
            
            # Ensure it's the right format for dlib
            if len(img_array.shape) == 2:  # Grayscale
                rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:  # RGBA
                rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            else:  # Already RGB
                rgb = img_array
            
            # Ensure contiguous 8-bit array - CRITICAL for dlib
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
            
            # Additional fix for macOS dlib: force a copy
            rgb = np.copy(rgb)
            
            # Verify the array is correct
            assert rgb.flags['C_CONTIGUOUS'], "Array is not C-contiguous"
            assert rgb.dtype == np.uint8, f"Wrong dtype: {rgb.dtype}"
            assert len(rgb.shape) == 3, f"Wrong dimensions: {rgb.shape}"
            assert rgb.shape[2] == 3, f"Wrong number of channels: {rgb.shape[2]}"
            
            print(f"Final RGB shape: {rgb.shape}, dtype: {rgb.dtype}, C-contiguous: {rgb.flags['C_CONTIGUOUS']}")
            
        except Exception as pil_error:
            print(f"PIL failed: {pil_error}")
            print("Trying OpenCV fallback...")
            
            # Fallback to OpenCV
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"ERROR: OpenCV also failed to load image")
                continue
            
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
            rgb = np.copy(rgb)  # Force fresh copy for dlib
        
        # Detect faces with dlib
        print("Running face detection...")
        faces = detector(rgb, 1)  # Second parameter is upsampling (1 = detect more faces)
        print(f"✓ Detected {len(faces)} face(s)")
        
        # Draw rectangles on detected faces
        for i, face in enumerate(faces):
            left, top, right, bottom = face.left(), face.top(), face.right(), face.bottom()
            print(f"  Face {i+1}: ({left}, {top}) to ({right}, {bottom})")
            cv2.rectangle(rgb, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Save the output image
        output_path = os.path.join(output_folder, img_file)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(output_path, bgr)
        
        if success:
            print(f"✓ Saved to {output_path}")
        else:
            print(f"ERROR: Failed to save image")
        
    except Exception as e:
        print(f"ERROR processing {img_file}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n=== Processing complete ===")
print(f"Check '{output_folder}' for results")
