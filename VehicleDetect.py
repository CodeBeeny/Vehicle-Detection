import cv2
import numpy as np
import os
import time

def save_violation_image(frame, bbox, violation_type):
    x, y, w, h = bbox
    violation_image = frame[y:y+h, x:x+w]
    filename = f"rule_break/{violation_type}_{int(time.time())}.jpg"
    cv2.imwrite(filename, violation_image)

def is_collision(box1, box2):
    # Improved intersection over union check
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate rectangle boundaries
    left1, right1 = x1, x1 + w1
    top1, bottom1 = y1, y1 + h1
    left2, right2 = x2, x2 + w2
    top2, bottom2 = y2, y2 + h2
    
    # Check if rectangles overlap
    if left1 > right2 or right1 < left2 or top1 > bottom2 or bottom1 < top2:
        return False
    
    # Calculate overlap area
    intersection_area = (min(right1, right2) - max(left1, left2)) * \
                       (min(bottom1, bottom2) - max(top1, top2))
    
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area
    
    iou = intersection_area / float(union_area)
    return iou > 0.3  
# You can adjust this threshold

# Load YOLOv3
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load classes
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Video sources
caps = [
    
    cv2.VideoCapture('side2.mp4'),
    
]

if not os.path.exists('rule_break'):
    os.makedirs('rule_break')

line_positions = [500, 500, 500, 500]
prev_boxes_list = [[] for _ in range(4)]

while True:
    frames = [cap.read()[1] for cap in caps]
    if any(frame is None for frame in frames):
        break

    for i, frame in enumerate(frames):
        height, width, _ = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids, confidences, boxes = [], [], []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    if classes[class_id] in ['car', 'bus', 'truck', 'motorbike']:
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        current_boxes = [boxes[j] for j in indexes.flatten()]
        accident_detected = False

        # Check for collisions between vehicles
        accident_cooldown = 60
        accident_display_frame_count = 0
        if accident_display_frame_count <= 0:
         for j in range(len(current_boxes)):
            for k in range(j + 1, len(current_boxes)):
                if is_collision(current_boxes[j], current_boxes[k]):
                    accident_detected = True
                    accident_display_frame_count = accident_cooldown
                    save_violation_image(frame, current_boxes[j], 'accident')
                    cv2.putText(frame, "Accident Detected!", (50, 150), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        else:
          cv2.putText(frame, "Accident Detected!", (50, 150), 
          cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
          accident_display_frame_count -= 0.01


        # Draw detection boxes
        for j in indexes.flatten():
            x, y, w, h = boxes[j]
            label = str(classes[class_ids[j]])
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Red-light violation
        for j in indexes.flatten():
            x, y, w, h = boxes[j]
            if line_positions[i] and y + h > line_positions[i]:
                cv2.putText(frame, 'Violation', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                save_violation_image(frame, (x, y, w, h), 'red_light_violation')

        cv2.imshow(f'Side {i+1}', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
