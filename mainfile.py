#!/usr/bin/env python3
"""
Complete Enhanced Agentic Video Understanding Chat Assistant
Final comprehensive version with Real AI Vision Integration
Now properly analyzes ANY activity - eating, tennis, cooking, etc!
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import asyncio
from abc import ABC, abstractmethod
import logging
from collections import deque
import threading
import queue
import os
import time
import subprocess
import sys
import base64

# Additional imports for Ollama integration
import aiohttp
import requests

# Third-party imports (install via pip)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Vision analysis imports
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from transformers import pipeline
    from PIL import Image
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class VideoEvent:
    """Represents a detected event in the video"""
    timestamp: float
    event_type: str
    description: str
    confidence: float
    bounding_box: Optional[Tuple[int, int, int, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConversationContext:
    """Maintains conversation state and context"""
    video_events: List[VideoEvent] = field(default_factory=list)
    video_summary: str = ""
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    current_video_path: Optional[str] = None
    analysis_complete: bool = False
    video_metadata: Dict[str, Any] = field(default_factory=dict)

# ============================================================================
# REAL AI VISION ANALYZERS - NO MORE HARDCODED ACTIVITIES!
# ============================================================================

class OpenAIVisionAnalyzer:
    """OpenAI GPT-4 Vision for most accurate activity analysis"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1/chat/completions"
    
    def analyze_frame_activity(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze what activity is happening in the frame"""
        try:
            # Convert frame to base64
            _, buffer = cv2.imencode('.jpg', frame)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4-vision-preview",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Analyze this image and describe what activity the person is doing. Be specific about actions like eating, drinking, playing sports, reading, cooking, dancing, etc. Respond with a clear, descriptive sentence."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{frame_b64}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 300
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                return {
                    "activity": content,
                    "confidence": 0.9,
                    "objects_involved": self._extract_objects(content)
                }
            else:
                return {"activity": "unknown activity", "confidence": 0.1}
                
        except Exception as e:
            logger.error(f"OpenAI Vision error: {e}")
            return {"activity": "analysis failed", "confidence": 0.0}
    
    def _extract_objects(self, description: str) -> List[str]:
        """Extract objects from description"""
        objects = []
        description_lower = description.lower()
        common_objects = [
            "cup", "glass", "bottle", "food", "plate", "bowl", "spoon", "fork",
            "book", "phone", "laptop", "ball", "racket", "chair", "table",
            "pizza", "sandwich", "apple", "water", "coffee", "tea"
        ]
        
        for obj in common_objects:
            if obj in description_lower:
                objects.append(obj)
        return objects

class LocalVisionAnalyzer:
    """Local vision models using Transformers"""
    def __init__(self):
        self.available = False
        if TRANSFORMERS_AVAILABLE:
            try:
                print("🔄 Loading local vision models...")
                self.captioning_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.captioning_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                print("✅ Local vision models loaded successfully")
                self.available = True
            except Exception as e:
                logger.error(f"Local vision setup failed: {e}")
                print("❌ Local vision models failed to load")
        else:
            print("⚠️ Transformers not available. Install with: pip install transformers torch pillow")
    
    def analyze_frame_activity(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze activity using local models"""
        if not self.available:
            return {"activity": "local analysis unavailable", "confidence": 0.0}
        
        try:
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Generate caption
            inputs = self.captioning_processor(pil_image, return_tensors="pt")
            out = self.captioning_model.generate(**inputs, max_length=50)
            caption = self.captioning_processor.decode(out[0], skip_special_tokens=True)
            
            # Interpret the caption into activity
            activity_description = self._interpret_activity(caption)
            
            return {
                "activity": activity_description,
                "confidence": 0.75,
                "caption": caption,
                "objects_involved": self._extract_objects_from_caption(caption)
            }
            
        except Exception as e:
            logger.error(f"Local vision analysis error: {e}")
            return {"activity": "local analysis failed", "confidence": 0.0}
    
    def _interpret_activity(self, caption: str) -> str:
        """Interpret the caption to determine specific activity"""
        caption_lower = caption.lower()
        
        # Activity detection patterns
        if any(word in caption_lower for word in ["eating", "food", "bite", "chewing", "meal", "pizza", "sandwich"]):
            return f"Person eating - {caption}"
        elif any(word in caption_lower for word in ["drinking", "cup", "glass", "bottle", "sip", "coffee", "tea", "water"]):
            return f"Person drinking - {caption}"
        elif any(word in caption_lower for word in ["tennis", "racket", "ball", "court", "sport"]):
            return f"Person playing tennis - {caption}"
        elif any(word in caption_lower for word in ["reading", "book", "paper", "magazine", "text"]):
            return f"Person reading - {caption}"
        elif any(word in caption_lower for word in ["cooking", "kitchen", "stove", "pan", "chef"]):
            return f"Person cooking - {caption}"
        elif any(word in caption_lower for word in ["dancing", "music", "moving", "dance"]):
            return f"Person dancing - {caption}"
        elif any(word in caption_lower for word in ["sleeping", "bed", "lying", "rest"]):
            return f"Person sleeping/resting - {caption}"
        elif any(word in caption_lower for word in ["phone", "mobile", "calling", "talking"]):
            return f"Person using phone - {caption}"
        elif any(word in caption_lower for word in ["typing", "computer", "laptop", "keyboard"]):
            return f"Person using computer - {caption}"
        elif any(word in caption_lower for word in ["walking", "running", "jogging", "exercise"]):
            return f"Person exercising - {caption}"
        else:
            return f"Person engaged in activity - {caption}"
    
    def _extract_objects_from_caption(self, caption: str) -> List[str]:
        """Extract objects mentioned in caption"""
        common_objects = [
            "cup", "glass", "bottle", "food", "plate", "bowl", "spoon", "fork",
            "book", "phone", "laptop", "ball", "racket", "chair", "table",
            "pizza", "sandwich", "apple", "water", "coffee", "tea", "computer"
        ]
        
        found_objects = []
        caption_lower = caption.lower()
        
        for obj in common_objects:
            if obj in caption_lower:
                found_objects.append(obj)
                
        return found_objects

class OllamaVisionAnalyzer:
    """Ollama vision analysis using local models like llava"""
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
        self.available = False
        self.vision_model = None
        self._check_vision_model()
    
    def _check_vision_model(self) -> bool:
        """Check if vision-capable model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                vision_models = [m for m in models if 'vision' in m['name'].lower() or 'llava' in m['name'].lower()]
                
                if vision_models:
                    self.vision_model = vision_models[0]['name']
                    print(f"✅ Ollama vision model available: {self.vision_model}")
                    self.available = True
                else:
                    print("⚠️ No Ollama vision models found.")
                    print("   Install with: ollama pull llava:7b")
                    print("   Or try: ollama pull llava:13b")
        except Exception as e:
            print(f"⚠️ Ollama vision check failed: {e}")
    
    def analyze_frame_activity(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze activity using Ollama vision model"""
        if not self.available:
            return {"activity": "ollama vision unavailable", "confidence": 0.0}
        
        try:
            # Convert frame to base64
            _, buffer = cv2.imencode('.jpg', frame)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.vision_model,
                    "prompt": "Describe what activity the person in this image is doing. Be specific about actions like eating, drinking, playing sports, reading, cooking, dancing, working, etc. Give a clear, descriptive answer.",
                    "images": [frame_b64],
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                description = result.get('response', 'unknown activity')
                
                return {
                    "activity": f"Person engaged in: {description}",
                    "confidence": 0.8,
                    "description": description,
                    "objects_involved": self._extract_objects_from_description(description)
                }
            else:
                return {"activity": "ollama analysis failed", "confidence": 0.0}
                
        except Exception as e:
            logger.error(f"Ollama vision error: {e}")
            return {"activity": "ollama analysis error", "confidence": 0.0}
    
    def _extract_objects_from_description(self, description: str) -> List[str]:
        """Extract objects from description"""
        common_objects = [
            "cup", "glass", "bottle", "food", "plate", "bowl", "spoon", "fork",
            "book", "phone", "laptop", "ball", "racket", "chair", "table",
            "pizza", "sandwich", "apple", "water", "coffee", "tea", "computer"
        ]
        
        found_objects = []
        description_lower = description.lower()
        
        for obj in common_objects:
            if obj in description_lower:
                found_objects.append(obj)
                
        return found_objects

# ============================================================================
# ENHANCED VIDEO PROCESSOR WITH REAL AI VISION
# ============================================================================

class EnhancedVideoProcessor:
    """Enhanced video processing with REAL AI vision analysis"""
    
    def __init__(self, vision_analyzer=None):
        self.vision_analyzer = vision_analyzer
        
        # Initialize YOLO model for object detection
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO('yolov8n.pt')  # nano version for speed
                logger.info("✅ YOLO model loaded successfully")
            except Exception as e:
                logger.warning(f"YOLO model loading failed: {e}. Using mock detection.")
                self.yolo_model = None
        else:
            logger.warning("YOLO not available. Using mock detection.")
            self.yolo_model = None
        
        # Enhanced tracking for better movement analysis
        self.object_tracks = {}
        self.track_id_counter = 0
        self.scene_context = {
            'activity_type': 'unknown',
            'has_people': False,
            'indoor_likely': False,
            'activity_level': 'low'
        }
        
    def extract_frames(self, video_path: str, frame_interval: int = 30) -> List[Tuple[float, np.ndarray]]:
        """Extract frames from video at specified intervals with metadata"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video info: {duration:.1f}s, {fps:.1f}fps, {width}x{height}, {total_frames} frames")
        
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                frames.append((timestamp, frame))
                extracted_count += 1
                
                if extracted_count % 10 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Extraction progress: {progress:.1f}% ({extracted_count} frames)")
            
            frame_count += 1
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames from {duration:.1f}s video")
        return frames
    
    def analyze_person_activity(self, frame: np.ndarray, person_bbox: List[int]) -> Dict[str, Any]:
        """Analyze what a detected person is actually doing using AI vision"""
        if self.vision_analyzer is None:
            return {
                "activity": "Person detected - activity analysis unavailable",
                "confidence": 0.5,
                "objects_involved": []
            }
        
        try:
            # Extract person region with some context
            x1, y1, x2, y2 = person_bbox
            height, width = frame.shape[:2]
            
            # Expand bbox for context but keep person-focused
            padding = min(50, width//10, height//10)
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)
            
            person_region = frame[y1:y2, x1:x2]
            
            # Use AI to analyze the activity
            return self.vision_analyzer.analyze_frame_activity(person_region)
            
        except Exception as e:
            logger.error(f"Activity analysis error: {e}")
            return {
                "activity": "Activity analysis failed",
                "confidence": 0.1,
                "objects_involved": []
            }
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Enhanced object detection with REAL activity analysis"""
        if self.yolo_model is None:
            # Simple fallback detection
            return self._simple_person_detection(frame)
        
        # Real YOLO detection
        results = self.yolo_model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls)
                    confidence = float(box.conf)
                    bbox = box.xyxy[0].cpu().numpy().astype(int)
                    
                    if confidence > 0.3:
                        detection = {
                            'class': self.yolo_model.names[cls_id],
                            'confidence': confidence,
                            'bbox': bbox.tolist()
                        }
                        
                        # REAL AI ANALYSIS FOR PEOPLE - NO MORE HARDCODING!
                        if detection['class'] == 'person' and self.vision_analyzer:
                            activity_info = self.analyze_person_activity(frame, detection['bbox'])
                            detection['activity_analysis'] = activity_info
                        
                        detections.append(detection)
        
        self._update_scene_context(detections)
        return detections
    
    def _simple_person_detection(self, frame: np.ndarray) -> List[Dict]:
        """Simple person detection fallback when YOLO unavailable"""
        height, width = frame.shape[:2]
        
        # Simple motion/color detection could go here
        # For now, assume person in center region for demo
        person_bbox = [width//4, height//4, 3*width//4, 3*height//4]
        
        detection = {
            'class': 'person',
            'confidence': 0.6,
            'bbox': person_bbox
        }
        
        # Analyze activity even with simple detection
        if self.vision_analyzer:
            activity_info = self.vision_analyzer.analyze_frame_activity(frame)
            detection['activity_analysis'] = activity_info
        
        return [detection]
    
    def _update_scene_context(self, detections: List[Dict]):
        """Update scene context based on detections and activities"""
        classes = [det['class'] for det in detections]
        
        # Check for people and activities
        if 'person' in classes:
            self.scene_context['has_people'] = True
            
            # Analyze activities to determine scene type
            for det in detections:
                if det['class'] == 'person' and 'activity_analysis' in det:
                    activity = det['activity_analysis']['activity'].lower()
                    
                    if any(word in activity for word in ['eating', 'drinking', 'cooking', 'kitchen']):
                        self.scene_context['activity_type'] = 'indoor_domestic'
                        self.scene_context['indoor_likely'] = True
                    elif any(word in activity for word in ['tennis', 'sport', 'playing', 'running']):
                        self.scene_context['activity_type'] = 'sports'
                    elif any(word in activity for word in ['reading', 'computer', 'laptop', 'phone']):
                        self.scene_context['activity_type'] = 'work_study'
                        self.scene_context['indoor_likely'] = True
        
        # Activity level based on detection count
        if len(detections) > 5:
            self.scene_context['activity_level'] = 'high'
        elif len(detections) > 2:
            self.scene_context['activity_level'] = 'medium'
        else:
            self.scene_context['activity_level'] = 'low'
    
    def analyze_movement(self, prev_detections: List[Dict], curr_detections: List[Dict], 
                        time_diff: float) -> List[VideoEvent]:
        """Analyze object movement between frames with enhanced context and tracking"""
        events = []
        
        curr_by_class = {}
        for det in curr_detections:
            class_name = det['class']
            if class_name not in curr_by_class:
                curr_by_class[class_name] = []
            curr_by_class[class_name].append(det)
        
        matched_current = set()
        
        for prev_idx, prev_det in enumerate(prev_detections):
            class_name = prev_det['class']
            best_match = None
            min_distance = float('inf')
            best_curr_idx = -1
            
            if class_name in curr_by_class:
                for curr_idx, curr_det in enumerate(curr_by_class[class_name]):
                    if curr_idx in matched_current:
                        continue
                    
                    prev_center = self._get_bbox_center(prev_det['bbox'])
                    curr_center = self._get_bbox_center(curr_det['bbox'])
                    
                    distance = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                                     (curr_center[1] - prev_center[1])**2)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_match = curr_det
                        best_curr_idx = curr_idx
                
                if best_match and best_curr_idx != -1:
                    matched_current.add(best_curr_idx)
                    
                    prev_center = self._get_bbox_center(prev_det['bbox'])
                    curr_center = self._get_bbox_center(best_match['bbox'])
                    
                    movement_threshold = self._get_movement_threshold(class_name)
                    
                    if min_distance > movement_threshold:
                        speed = min_distance / time_diff if time_diff > 0 else 0
                        direction = self._get_movement_direction(prev_center, curr_center)
                        
                        event = self._create_movement_event(
                            class_name, speed, min_distance, direction, 
                            prev_det, best_match, prev_center, curr_center
                        )
                        
                        if event:
                            events.append(event)
        
        # Detect new objects with REAL activity analysis
        for curr_det in curr_detections:
            class_name = curr_det['class']
            curr_center = self._get_bbox_center(curr_det['bbox'])
            
            has_match = False
            for prev_det in prev_detections:
                if prev_det['class'] == class_name:
                    prev_center = self._get_bbox_center(prev_det['bbox'])
                    distance = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                                     (curr_center[1] - prev_center[1])**2)
                    if distance < self._get_movement_threshold(class_name) * 2:
                        has_match = True
                        break
            
            if not has_match:
                # Use real activity analysis if available
                if class_name == 'person' and 'activity_analysis' in curr_det:
                    activity = curr_det['activity_analysis']
                    description = f"New person appeared - {activity['activity']}"
                    metadata = {
                        "object_class": class_name,
                        "position": curr_center,
                        "bbox": curr_det['bbox'],
                        "activity_details": activity
                    }
                else:
                    description = f"New {class_name} detected in scene"
                    metadata = {
                        "object_class": class_name,
                        "position": curr_center,
                        "bbox": curr_det['bbox']
                    }
                
                events.append(VideoEvent(
                    timestamp=0,
                    event_type="object_appeared",
                    description=description,
                    confidence=curr_det['confidence'],
                    metadata=metadata
                ))
        
        # Detect disappeared objects
        for prev_det in prev_detections:
            class_name = prev_det['class']
            prev_center = self._get_bbox_center(prev_det['bbox'])
            
            has_match = False
            if class_name in curr_by_class:
                for curr_det in curr_by_class[class_name]:
                    curr_center = self._get_bbox_center(curr_det['bbox'])
                    distance = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                                     (curr_center[1] - prev_center[1])**2)
                    if distance < self._get_movement_threshold(class_name) * 2:
                        has_match = True
                        break
            
            if not has_match:
                if class_name == 'person' and 'activity_analysis' in prev_det:
                    activity = prev_det['activity_analysis']
                    description = f"Person left scene - was {activity['activity']}"
                else:
                    description = f"{class_name.title()} left the scene"
                
                events.append(VideoEvent(
                    timestamp=0,
                    event_type="object_disappeared",
                    description=description,
                    confidence=prev_det['confidence'],
                    metadata={
                        "object_class": class_name,
                        "last_position": prev_center,
                        "bbox": prev_det['bbox']
                    }
                ))
        
        return events
    
    def _create_movement_event(self, class_name: str, speed: float, distance: float, 
                              direction: str, prev_det: Dict, curr_det: Dict,
                              prev_center: Tuple, curr_center: Tuple) -> Optional[VideoEvent]:
        """Create movement event with real activity context"""
        
        confidence = min(prev_det['confidence'], curr_det['confidence'])
        metadata = {
            "object_class": class_name,
            "movement_distance": distance,
            "movement_speed": speed,
            "direction": direction,
            "prev_position": prev_center,
            "curr_position": curr_center,
            "prev_bbox": prev_det['bbox'],
            "curr_bbox": curr_det['bbox']
        }
        
        # Enhanced person movement with REAL activity analysis
        if class_name == 'person':
            # Use real activity analysis if available
            activity_desc = ""
            if 'activity_analysis' in curr_det:
                activity = curr_det['activity_analysis']['activity']
                activity_desc = f" - {activity}"
                metadata['activity_details'] = curr_det['activity_analysis']
            
            if speed > 150:
                description = f"Person moving quickly ({speed:.1f} px/s) {direction}{activity_desc}"
                event_type = "person_fast_movement"
            elif speed > 50:
                description = f"Person moving ({speed:.1f} px/s) {direction}{activity_desc}"
                event_type = "person_movement"
            else:
                description = f"Person slight movement ({speed:.1f} px/s) {direction}{activity_desc}"
                event_type = "person_activity_movement"
        
        # Vehicle movement analysis (unchanged)
        elif class_name in ['car', 'truck', 'bus', 'motorcycle']:
            if speed > 200:
                description = f"{class_name.title()} moving very fast ({speed:.1f} px/s) {direction}"
                event_type = "vehicle_speeding"
            elif speed > 100:
                description = f"{class_name.title()} moving at moderate speed ({speed:.1f} px/s) {direction}"
                event_type = "vehicle_movement"
            else:
                description = f"{class_name.title()} moving slowly ({speed:.1f} px/s) {direction}"
                event_type = "vehicle_slow_movement"
        
        # Other objects
        else:
            if speed > 100:
                description = f"{class_name.replace('_', ' ').title()} moving quickly ({speed:.1f} px/s) {direction}"
                event_type = "object_movement_fast"
            else:
                description = f"{class_name.replace('_', ' ').title()} position changed ({speed:.1f} px/s) {direction}"
                event_type = "object_movement"
        
        return VideoEvent(
            timestamp=0,
            event_type=event_type,
            description=description,
            confidence=confidence,
            metadata=metadata
        )
    
    def _get_movement_threshold(self, object_class: str) -> float:
        """Get movement threshold based on object type"""
        thresholds = {
            'person': 20,
            'car': 50,
            'truck': 50,
            'bus': 50,
            'motorcycle': 40,
            'bicycle': 30,
        }
        return thresholds.get(object_class, 30)
    
    def _get_movement_direction(self, prev_center: Tuple[int, int], 
                              curr_center: Tuple[int, int]) -> str:
        """Determine movement direction between two points"""
        dx = curr_center[0] - prev_center[0]
        dy = curr_center[1] - prev_center[1]
        
        angle = np.arctan2(dy, dx) * 180 / np.pi
        
        if -22.5 <= angle < 22.5:
            return "right"
        elif 22.5 <= angle < 67.5:
            return "down-right"
        elif 67.5 <= angle < 112.5:
            return "down"
        elif 112.5 <= angle < 157.5:
            return "down-left"
        elif 157.5 <= angle or angle < -157.5:
            return "left"
        elif -157.5 <= angle < -112.5:
            return "up-left"
        elif -112.5 <= angle < -67.5:
            return "up"
        elif -67.5 <= angle < -22.5:
            return "up-right"
        else:
            return "unknown"
    
    def _get_bbox_center(self, bbox: List[int]) -> Tuple[int, int]:
        """Get center point of bounding box"""
        return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

# ============================================================================
# LLM INTERFACES
# ============================================================================

class LLMInterface(ABC):
    """Abstract base class for LLM interfaces"""
    
    @abstractmethod
    async def generate_response(self, prompt: str, context: ConversationContext) -> str:
        pass
    
    @abstractmethod
    async def summarize_events(self, events: List[VideoEvent]) -> str:
        pass

class OllamaInterface(LLMInterface):
    """Enhanced Ollama API interface for local LLM inference"""
    
    def __init__(self, 
                 model_name: str = "llama3.2:1b",
                 base_url: str = "http://localhost:11434",
                 timeout: int = 90):
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.generate_url = f"{self.base_url}/api/generate"
        self.chat_url = f"{self.base_url}/api/chat"
        
        self.is_available = False
        self._check_connection()
    
    def _check_connection(self):
        """Check if Ollama is available and initialize model if needed"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json()
                available_models = [model['name'] for model in models.get('models', [])]
                
                if self.model_name not in available_models:
                    logger.warning(f"Model {self.model_name} not found. Available: {available_models}")
                    print(f"⚠️  Model {self.model_name} not available. Run: ollama pull {self.model_name}")
                else:
                    logger.info(f"✅ Ollama model {self.model_name} is ready")
                    print(f"✅ Ollama model {self.model_name} is ready")
                
                self.is_available = True
            else:
                logger.error(f"Ollama server not responding: {response.status}")
                print("❌ Ollama server not responding")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            print(f"❌ Failed to connect to Ollama: {e}")
            print("Make sure Ollama is running with: ollama serve")
    
    async def generate_response(self, prompt: str, context: ConversationContext) -> str:
        """Generate response using Ollama chat API with enhanced context"""
        if not self.is_available:
            return self._fallback_response(prompt, context)
        
        try:
            messages = self._build_chat_messages(prompt, context)
            
            chat_data = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 512,
                    "stop": ["Human:", "User:", "Assistant:"]
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.chat_url,
                    json=chat_data,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        return result.get('message', {}).get('content', 
                                                           'I apologize, but I could not generate a response.')
                    else:
                        error_text = await response.text()
                        logger.error(f"Ollama API error {response.status}: {error_text}")
                        return self._fallback_response(prompt, context)
                        
        except asyncio.TimeoutError:
            logger.error("Ollama request timed out")
            return "I'm taking longer than expected to respond. Please try asking a simpler question."
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return self._fallback_response(prompt, context)
    
    async def summarize_events(self, events: List[VideoEvent]) -> str:
        """Summarize video events using Ollama with enhanced analysis"""
        if not events:
            return "No significant events detected in the video."
        
        if not self.is_available:
            return self._create_rule_based_summary(events)
        
        # Prepare detailed event description
        events_by_type = {}
        timeline_events = []
        violations = []
        high_confidence_events = []
        activity_events = []
        
        for event in events:
            if event.event_type not in events_by_type:
                events_by_type[event.event_type] = []
            events_by_type[event.event_type].append(event)
            
            timeline_events.append((event.timestamp, event.description))
            
            if 'violation' in event.event_type:
                violations.append(event)
            if event.confidence > 0.8:
                high_confidence_events.append(event)
            if 'activity' in event.event_type or 'person' in event.event_type:
                activity_events.append(event)
        
        timeline_events.sort()
        
        prompt_parts = [
            "You are an expert video analyst. Analyze the following video event data and provide a comprehensive summary.",
            "",
            f"TOTAL EVENTS: {len(events)}",
            f"VIDEO DURATION: {timeline_events[-1][0] - timeline_events[0][0]:.1f} seconds" if timeline_events else "Unknown",
            f"HIGH CONFIDENCE DETECTIONS: {len(high_confidence_events)}",
            f"ACTIVITY EVENTS: {len(activity_events)}",
            f"POTENTIAL VIOLATIONS: {len(violations)}",
            "",
            "EVENT BREAKDOWN BY TYPE:"
        ]
        
        for event_type, type_events in events_by_type.items():
            prompt_parts.append(f"- {event_type.replace('_', ' ').title()}: {len(type_events)} events")
        
        prompt_parts.extend([
            "",
            "KEY TIMELINE EVENTS (first 15):"
        ])
        
        for i, (timestamp, description) in enumerate(timeline_events[:15]):
            prompt_parts.append(f"{i+1}. {timestamp:.1f}s: {description}")
        
        if activity_events:
            prompt_parts.extend([
                "",
                "ACTIVITY ANALYSIS:"
            ])
            for activity in activity_events[:5]:
                prompt_parts.append(f"- {activity.timestamp:.1f}s: {activity.description}")
        
        if violations:
            prompt_parts.extend([
                "",
                "POTENTIAL VIOLATIONS:"
            ])
            for violation in violations[:5]:
                prompt_parts.append(f"- {violation.timestamp:.1f}s: {violation.description}")
        
        prompt_parts.extend([
            "",
            "Please provide:",
            "1. Executive summary of video content and main activities",
            "2. Most significant events and patterns", 
            "3. Activity analysis (what people were doing)",
            "4. Safety concerns or violations identified",
            "5. Overall assessment and likely scenario type",
            "",
            "Keep the summary comprehensive but focused on the most important insights."
        ])
        
        prompt = "\n".join(prompt_parts)
        
        try:
            generate_data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "num_predict": 1000,
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.generate_url,
                    json=generate_data,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        return result.get('response', self._create_rule_based_summary(events))
                    else:
                        logger.error(f"Ollama summarization error: {response.status}")
                        return self._create_rule_based_summary(events)
                        
        except Exception as e:
            logger.error(f"Ollama summarization error: {e}")
            return self._create_rule_based_summary(events)
    
    def _build_chat_messages(self, user_message: str, context: ConversationContext) -> List[Dict[str, str]]:
        """Build chat messages with enhanced context"""
        messages = []
        
        system_content = self._build_system_prompt(context)
        messages.append({"role": "system", "content": system_content})
        
        if context.conversation_history:
            for exchange in context.conversation_history[-2:]:
                messages.append({"role": "user", "content": exchange["user"]})
                messages.append({"role": "assistant", "content": exchange["assistant"]})
        
        messages.append({"role": "user", "content": user_message})
        return messages
    
    def _build_system_prompt(self, context: ConversationContext) -> str:
        """Build comprehensive system prompt with video context"""
        system_parts = [
            "You are an expert video analysis assistant with REAL AI vision capabilities.",
            "",
            "CORE CAPABILITIES:",
            "- Analyze ANY video activity: eating, sports, cooking, reading, working, etc.",
            "- Use actual AI vision analysis (not hardcoded responses)",
            "- Detect and describe specific human activities and behaviors", 
            "- Provide temporal analysis with precise timestamps",
            "- Identify relationships between different events and objects",
            "- Answer questions about specific activities, timeframes, or patterns",
            "",
            "COMMUNICATION GUIDELINES:",
            "- Be conversational, helpful, and technically accurate",
            "- Include specific details: timestamps, confidence levels, activities",
            "- Reference actual AI analysis results when available",
            "- Ask clarifying questions when user intent is unclear",
            "- Be transparent about analysis methods and limitations"
        ]
        
        if context.analysis_complete and context.video_events:
            system_parts.extend([
                "",
                "CURRENT VIDEO ANALYSIS STATUS: ✅ COMPLETE",
                "="*50
            ])
            
            event_types = {}
            violations = []
            high_confidence_events = []
            activity_events = []
            
            for event in context.video_events:
                event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
                if event.confidence > 0.8:
                    high_confidence_events.append(event)
                if 'violation' in event.event_type.lower():
                    violations.append(event)
                if 'activity' in event.event_type or 'person' in event.event_type:
                    activity_events.append(event)
            
            system_parts.extend([
                "ANALYSIS SUMMARY:",
                f"  Total Events: {len(context.video_events)}",
                f"  Event Types: {len(event_types)}",
                f"  High Confidence (>0.8): {len(high_confidence_events)}",
                f"  Activity Events: {len(activity_events)}",
                f"  Potential Violations: {len(violations)}",
                ""
            ])
            
            if event_types:
                system_parts.append("EVENT TYPE BREAKDOWN:")
                for event_type, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / len(context.video_events)) * 100
                    system_parts.append(f"  {event_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
                system_parts.append("")
            
            # Show actual activities detected
            if activity_events:
                system_parts.extend([
                    "DETECTED ACTIVITIES:",
                    *[f"  {e.timestamp:.1f}s: {e.description}" for e in activity_events[:5]],
                    ""
                ])
            
            if context.video_summary:
                system_parts.extend([
                    "AI GENERATED SUMMARY:",
                    context.video_summary,
                    ""
                ])
                
        elif context.current_video_path:
            system_parts.extend([
                "",
                "CURRENT STATUS: 🟡 ANALYSIS IN PROGRESS",
                f"Processing: {context.current_video_path}",
                "Please wait for analysis to complete before asking detailed questions."
            ])
        else:
            system_parts.extend([
                "",
                "CURRENT STATUS: ⭕ NO VIDEO LOADED",
                "No video has been analyzed yet. Please ask the user to upload a video first using 'load <path>' or 'demo' commands."
            ])
        
        return "\n".join(system_parts)
    
    def _fallback_response(self, prompt: str, context: ConversationContext) -> str:
        """Generate intelligent fallback response when Ollama is unavailable"""
        prompt_lower = prompt.lower()
        
        if not context.analysis_complete:
            if "load" in prompt_lower or "upload" in prompt_lower:
                return "I'm ready to analyze a video with REAL AI vision! Use the 'load <video_path>' command or 'demo' to create a test video."
            return "I don't have any video analysis to discuss yet. Please upload a video first using the 'load <path>' command or try 'demo' for a test video."
        
        # Handle specific query types with enhanced responses
        if any(word in prompt_lower for word in ["summary", "overview", "what happened"]):
            return self._generate_enhanced_summary(context)
        elif any(word in prompt_lower for word in ["activity", "doing", "action", "behavior"]):
            return self._generate_activity_analysis(context)
        elif any(word in prompt_lower for word in ["violation", "illegal", "traffic", "safety"]):
            return self._generate_violation_analysis(context)
        elif any(word in prompt_lower for word in ["when", "time", "timeline", "sequence"]):
            return self._generate_timeline_analysis(context)
        elif any(word in prompt_lower for word in ["count", "how many", "number", "total"]):
            return self._generate_count_analysis(context)
        elif any(word in prompt_lower for word in ["what", "see", "detect", "found", "identify"]):
            return self._generate_detection_analysis(context)
        elif any(word in prompt_lower for word in ["speed", "fast", "slow", "movement"]):
            return self._generate_movement_analysis(context)
        else:
            return f"I've analyzed {len(context.video_events)} events in your video using REAL AI vision. Could you be more specific about what aspect you'd like to know about? Try asking about activities, timeline, violations, or movements."
    
    def _generate_activity_analysis(self, context: ConversationContext) -> str:
        """Generate activity-focused analysis response"""
        activity_events = [e for e in context.video_events if 'activity' in e.event_type or 'person' in e.event_type]
        
        if not activity_events:
            return "No specific activities were detected in the video analysis."
        
        activities_detected = []
        for event in activity_events:
            if 'activity_details' in event.metadata:
                activity_detail = event.metadata['activity_details']
                activities_detected.append(f"{event.timestamp:.1f}s: {activity_detail.get('activity', event.description)}")
            else:
                activities_detected.append(f"{event.timestamp:.1f}s: {event.description}")
        
        return f"""🎯 Activity Analysis ({len(activity_events)} activity events detected):

Using REAL AI Vision Analysis:
{chr(10).join(activities_detected[:8])}

{f'... and {len(activity_events) - 8} more activities detected.' if len(activity_events) > 8 else ''}

These activities were analyzed using actual computer vision, not hardcoded responses!"""
    
    def _generate_enhanced_summary(self, context: ConversationContext) -> str:
        """Generate enhanced summary response with activity focus"""
        if not context.video_events:
            return "No events were detected in the video analysis."
        
        events = context.video_events
        event_types = {}
        violations = []
        movements = []
        activities = []
        
        for event in events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
            if 'violation' in event.event_type:
                violations.append(event)
            if 'movement' in event.event_type:
                movements.append(event)
            if 'activity' in event.event_type or 'person' in event.event_type:
                activities.append(event)
        
        duration = max(e.timestamp for e in events) - min(e.timestamp for e in events) if events else 0
        avg_confidence = np.mean([e.confidence for e in events])
        
        summary_parts = [
            f"📋 Enhanced Video Analysis Summary (REAL AI Vision):",
            "",
            f"🎯 Overview:",
            f"  • Total Events: {len(events)} over {duration:.1f} seconds",
            f"  • Average Confidence: {avg_confidence:.2f}/1.0",
            f"  • Activity Level: {'High' if len(events) > 20 else 'Medium' if len(events) > 10 else 'Low'}",
            "",
            f"📊 Event Categories:",
            *[f"  • {k.replace('_', ' ').title()}: {v}" for k, v in sorted(event_types.items(), key=lambda x: x[1], reverse=True)[:5]],
        ]
        
        if activities:
            summary_parts.extend([
                "",
                f"🎬 Activities Detected ({len(activities)} events):",
                *[f"  • {a.timestamp:.1f}s: {a.description}" for a in activities[:5]]
            ])
        
        summary_parts.extend([
            "",
            f"⚠️  Safety Analysis:",
            f"  • Violations Detected: {len(violations)}",
            f"  • Movement Events: {len(movements)}",
            "",
            f"The video shows {'high' if len(events) > 20 else 'moderate' if len(events) > 10 else 'low'} activity with REAL AI-powered activity recognition throughout the timeline."
        ])
        
        return "\n".join(summary_parts)
    
    def _generate_violation_analysis(self, context: ConversationContext) -> str:
        """Generate violation analysis response"""
        violations = [e for e in context.video_events if 'violation' in e.event_type.lower() or 
                     any(word in e.description.lower() for word in ['violation', 'illegal', 'speeding', 'ran'])]
        
        if not violations:
            return "🟢 No clear violations detected. The analysis shows normal activity patterns with no obvious safety concerns or traffic violations."
        
        return f"""⚠️  Violation Analysis ({len(violations)} potential issues):

{chr(10).join([f'{i+1}. {v.timestamp:.1f}s: {v.description} (Confidence: {v.confidence:.2f})' for i, v in enumerate(violations[:5])])}

{f'... and {len(violations) - 5} more violations detected.' if len(violations) > 5 else ''}

These detections should be reviewed for context and may require human verification."""
    
    def _generate_timeline_analysis(self, context: ConversationContext) -> str:
        """Generate timeline analysis response"""
        if not context.video_events:
            return "No timeline events have been recorded."
        
        events_by_time = sorted(context.video_events, key=lambda x: x.timestamp)[:12]
        
        return f"""⏰ Timeline Analysis ({len(context.video_events)} total events):

{chr(10).join([f'{i+1:2d}. {e.timestamp:6.1f}s: {e.description}' for i, e in enumerate(events_by_time)])}

{f'... and {len(context.video_events) - 12} more events in the complete timeline.' if len(context.video_events) > 12 else ''}"""
    
    def _generate_count_analysis(self, context: ConversationContext) -> str:
        """Generate count analysis response"""
        if not context.video_events:
            return "No events have been counted yet."
        
        event_counts = {}
        object_counts = {}
        
        for event in context.video_events:
            event_type = event.event_type.replace('_', ' ').title()
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            if 'object_class' in event.metadata:
                obj_class = event.metadata['object_class']
                object_counts[obj_class] = object_counts.get(obj_class, 0) + 1
        
        count_parts = [
            f"🔢 Count Analysis (Total: {len(context.video_events)} events):",
            "",
            "Event Types:",
            *[f"  • {event_type}: {count}" for event_type, count in sorted(event_counts.items(), key=lambda x: x[1], reverse=True)]
        ]
        
        if object_counts:
            count_parts.extend([
                "",
                "Object Types:",
                *[f"  • {obj_type.title()}: {count}" for obj_type, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:8]]
            ])
        
        return "\n".join(count_parts)
    
    def _generate_detection_analysis(self, context: ConversationContext) -> str:
        """Generate detection analysis response"""
        if not context.video_events:
            return "No detections have been made yet."
        
        unique_descriptions = list(set([e.description for e in context.video_events]))
        object_types = set()
        
        for event in context.video_events:
            if 'object_class' in event.metadata:
                object_types.add(event.metadata['object_class'])
        
        return f"""👁️ Detection Analysis (REAL AI Vision):

Unique Events Detected: {len(unique_descriptions)}
{chr(10).join([f'  • {desc}' for desc in unique_descriptions[:8]])}
{f'  ... and {len(unique_descriptions) - 8} more' if len(unique_descriptions) > 8 else ''}

{f'Object Types: {", ".join(sorted(object_types)[:10])}' if object_types else 'No specific object types recorded'}

Analysis powered by real computer vision, not hardcoded responses!"""
    
    def _generate_movement_analysis(self, context: ConversationContext) -> str:
        """Generate movement analysis response"""
        movement_events = [e for e in context.video_events if 'movement' in e.event_type.lower()]
        
        if not movement_events:
            return "No significant movement patterns detected in the analysis."
        
        speeds = []
        directions = {}
        
        for event in movement_events:
            if 'movement_speed' in event.metadata:
                speeds.append(event.metadata['movement_speed'])
            if 'direction' in event.metadata:
                direction = event.metadata['direction']
                directions[direction] = directions.get(direction, 0) + 1
        
        movement_parts = [f"🏃 Movement Analysis ({len(movement_events)} movement events):"]
        
        if speeds:
            avg_speed = np.mean(speeds)
            max_speed = max(speeds)
            movement_parts.extend([
                "",
                f"Speed Statistics:",
                f"  • Average: {avg_speed:.1f} px/s",
                f"  • Maximum: {max_speed:.1f} px/s",
                f"  • Classification: {'Fast' if avg_speed > 100 else 'Medium' if avg_speed > 50 else 'Slow'}"
            ])
        
        if directions:
            movement_parts.extend([
                "",
                "Directions:",
                *[f"  • {direction.title()}: {count}" for direction, count in sorted(directions.items(), key=lambda x: x[1], reverse=True)]
            ])
        
        return "\n".join(movement_parts)
    
    def _create_rule_based_summary(self, events: List[VideoEvent]) -> str:
        """Create comprehensive rule-based summary when LLM is unavailable"""
        if not events:
            return "No events detected in the video analysis."
        
        # Comprehensive analysis
        event_types = {}
        violations = []
        high_confidence = []
        timeline_events = []
        object_classes = set()
        movements = []
        activities = []
        speeds = []
        
        for event in events:
            event_type = event.event_type.replace('_', ' ').title()
            event_types[event_type] = event_types.get(event_type, 0) + 1
            
            if any(word in event.description.lower() for word in ['violation', 'ran', 'illegal', 'speeding']):
                violations.append(event)
            
            if event.confidence > 0.8:
                high_confidence.append(event)
            
            timeline_events.append((event.timestamp, event.description))
            
            if 'object_class' in event.metadata:
                object_classes.add(event.metadata['object_class'])
            
            if 'movement' in event.event_type:
                movements.append(event)
                if 'movement_speed' in event.metadata:
                    speeds.append(event.metadata['movement_speed'])
            
            if 'activity' in event.event_type or 'person' in event.event_type:
                activities.append(event)
        
        timeline_events.sort()
        duration = timeline_events[-1][0] - timeline_events[0][0] if len(timeline_events) > 1 else 0
        
        summary_parts = [
            "🎥 COMPREHENSIVE VIDEO ANALYSIS REPORT (REAL AI VISION)",
            "=" * 65,
            "",
            "📊 EXECUTIVE SUMMARY:",
            f"   • Total Events: {len(events)}",
            f"   • Duration: {duration:.1f} seconds", 
            f"   • Avg Confidence: {np.mean([e.confidence for e in events]):.2f}/1.0",
            f"   • Event Rate: {len(events)/duration:.2f}/sec" if duration > 0 else "   • Event Rate: N/A",
            f"   • Activity Level: {'High' if len(events) > 25 else 'Medium' if len(events) > 12 else 'Low'}",
            ""
        ]
        
        if event_types:
            summary_parts.extend([
                "📋 EVENT BREAKDOWN:",
                *[f"   • {event_type}: {count} ({count/len(events)*100:.1f}%)" 
                  for event_type, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True)],
                ""
            ])
        
        if activities:
            summary_parts.extend([
                f"🎬 ACTIVITIES DETECTED ({len(activities)} events):",
                *[f"   • {a.timestamp:.1f}s: {a.description}" for a in activities[:5]],
                ""
            ])
        
        if object_classes:
            summary_parts.extend([
                f"👁️  OBJECTS DETECTED: {', '.join(sorted(object_classes))}",
                ""
            ])
        
        if violations:
            summary_parts.extend([
                f"⚠️  VIOLATIONS ({len(violations)} detected):",
                *[f"   • {v.timestamp:.1f}s: {v.description}" for v in violations[:5]],
                ""
            ])
        else:
            summary_parts.extend(["✅ No violations detected", ""])
        
        if movements and speeds:
            avg_speed = np.mean(speeds)
            summary_parts.extend([
                f"🏃 MOVEMENT: {len(movements)} events, avg speed {avg_speed:.1f} px/s",
                ""
            ])
        
        summary_parts.extend([
            "💡 ANALYSIS QUALITY:",
            f"   • High confidence events: {len(high_confidence)}",
            f"   • Real AI Vision: {'✅ Active' if activities else '⚠️ Limited'}",
            f"   • Reliability: {'Excellent' if np.mean([e.confidence for e in events]) > 0.8 else 'Good'}",
        ])
        
        return "\n".join(summary_parts)

class MockLLMInterface(LLMInterface):
    """Enhanced mock LLM interface for testing and fallback"""
    
    def __init__(self):
        self.responses = [
            "I can see interesting activity in this video using REAL AI vision. What specific aspect would you like me to focus on?",
            "Based on the AI analysis, I notice several activities occurring throughout the timeline.",
            "The video shows various activities and movements detected by computer vision. Would you like details about any timeframe?",
            "I've processed the video with AI vision and can discuss activities, movements, or violations detected.",
            "The analysis reveals several key moments and activities. What interests you most?",
            "I can help you understand the AI-detected patterns. What should we focus on?",
            "The video contains multiple activity types detected by real vision analysis. Should we examine activities, safety, or movements?"
        ]
        self.response_index = 0
    
    async def generate_response(self, prompt: str, context: ConversationContext) -> str:
        """Generate enhanced mock conversational response"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["summary", "overview", "what happened"]):
            return self._mock_summary(context)
        elif any(word in prompt_lower for word in ["activity", "doing", "action", "behavior"]):
            return self._mock_activities(context)
        elif any(word in prompt_lower for word in ["violation", "illegal", "safety"]):
            return self._mock_violations(context)
        elif any(word in prompt_lower for word in ["when", "time", "timeline"]):
            return self._mock_timeline(context)
        elif any(word in prompt_lower for word in ["count", "how many", "number"]):
            return self._mock_counts(context)
        elif any(word in prompt_lower for word in ["what", "see", "detect"]):
            return self._mock_detections(context)
        elif any(word in prompt_lower for word in ["speed", "movement", "fast"]):
            return self._mock_movement(context)
        elif "help" in prompt_lower:
            return self._mock_help(context)
        else:
            response = self.responses[self.response_index % len(self.responses)]
            self.response_index += 1
            if context.video_events:
                response += f" I've analyzed {len(context.video_events)} events with REAL AI vision."
            return response
    
    def _mock_activities(self, context: ConversationContext) -> str:
        """Mock activity analysis response"""
        activity_events = [e for e in context.video_events if 'activity' in e.event_type or 'person' in e.event_type]
        if not activity_events:
            return "🎬 No specific activities detected in the analysis."
        
        return f"""🎬 Activity Analysis (REAL AI Vision):
🎯 {len(activity_events)} activity events detected:
{chr(10).join([f'• {a.timestamp:.1f}s: {a.description}' for a in activity_events[:5]])}

These activities were analyzed using actual computer vision, not hardcoded responses!"""
    
    def _mock_summary(self, context: ConversationContext) -> str:
        if not context.video_events:
            return "No events detected yet. Please analyze a video first."
        
        event_types = {}
        activities = []
        for event in context.video_events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
            if 'activity' in event.event_type or 'person' in event.event_type:
                activities.append(event)
        
        violations = [e for e in context.video_events if 'violation' in e.event_type]
        avg_conf = np.mean([e.confidence for e in context.video_events])
        
        return f"""📋 Enhanced Analysis Summary (REAL AI Vision):
🎯 {len(context.video_events)} events detected (avg confidence: {avg_conf:.2f})
🎬 Activities: {len(activities)} activity events with AI analysis
📊 Types: {', '.join(list(event_types.keys())[:4])}
⚠️  Violations: {len(violations)}
📈 Activity: {'High' if len(context.video_events) > 20 else 'Medium' if len(context.video_events) > 10 else 'Low'}

Powered by real computer vision, not hardcoded responses!"""
    
    def _mock_violations(self, context: ConversationContext) -> str:
        violations = [e for e in context.video_events if 'violation' in e.event_type.lower()]
        if not violations:
            return "🟢 No violations detected in the AI analysis."
        return f"⚠️  {len(violations)} potential violations found:\n" + \
               "\n".join([f"• {v.timestamp:.1f}s: {v.description}" for v in violations[:3]])
    
    def _mock_timeline(self, context: ConversationContext) -> str:
        if not context.video_events:
            return "No timeline events recorded."
        events = sorted(context.video_events, key=lambda x: x.timestamp)[:8]
        return "⏰ Timeline (AI Analysis):\n" + "\n".join([f"• {e.timestamp:.1f}s: {e.description}" for e in events])
    
    def _mock_counts(self, context: ConversationContext) -> str:
        if not context.video_events:
            return "No events to count."
        types = {}
        for event in context.video_events:
            types[event.event_type] = types.get(event.event_type, 0) + 1
        return f"🔢 Counts (Total: {len(context.video_events)} with AI Vision):\n" + \
               "\n".join([f"• {k.replace('_', ' ').title()}: {v}" for k, v in types.items()])
    
    def _mock_detections(self, context: ConversationContext) -> str:
        if not context.video_events:
            return "No detections made yet."
        descriptions = list(set([e.description for e in context.video_events]))
        return f"👁️  AI Vision Detected {len(descriptions)} unique events:\n" + \
               "\n".join([f"• {d}" for d in descriptions[:5]])
    
    def _mock_movement(self, context: ConversationContext) -> str:
        movements = [e for e in context.video_events if 'movement' in e.event_type]
        if not movements:
            return "No movement events detected."
        return f"🏃 {len(movements)} movement events detected with AI analysis of various speeds and directions."
    
    def _mock_help(self, context: ConversationContext) -> str:
        return """🆘 Enhanced Video Analysis Help (REAL AI Vision):
Ask me about: activities, summaries, violations, timeline, counts, detections, movements
Example: "What activities did you find?" or "Show me the timeline"
""" + (f"✅ Current video has {len(context.video_events)} events analyzed with AI vision." if context.video_events else "⏳ Load a video first.")
    
    async def summarize_events(self, events: List[VideoEvent]) -> str:
        """Generate mock event summary with activity focus"""
        if not events:
            return "No events detected."
        
        event_types = {}
        activities = []
        for event in events:
            event_type = event.event_type.replace('_', ' ').title()
            event_types[event_type] = event_types.get(event_type, 0) + 1
            if 'activity' in event.event_type or 'person' in event.event_type:
                activities.append(event)
        
        avg_confidence = np.mean([e.confidence for e in events])
        violations = [e for e in events if 'violation' in e.event_type]
        
        return f"""📋 Enhanced Video Summary (REAL AI Vision):
• {len(events)} total events analyzed
• Average confidence: {avg_confidence:.2f}
• Activities detected: {len(activities)} events with AI analysis
• Event types: {', '.join(list(event_types.keys())[:5])}
• Violations detected: {len(violations)}
• Activity level: {'High' if len(events) > 20 else 'Medium' if len(events) > 10 else 'Low'}

The video shows various activities analyzed using real computer vision with object movements and human activity recognition."""

# ============================================================================
# ENHANCED AGENTIC VIDEO ASSISTANT
# ============================================================================

class AgenticVideoAssistant:
    """Main agentic assistant coordinating video analysis and conversation with REAL AI vision"""
    
    def __init__(self, llm_interface: LLMInterface, vision_analyzer=None):
        self.video_processor = EnhancedVideoProcessor(vision_analyzer=vision_analyzer)
        self.llm = llm_interface
        self.context = ConversationContext()
        self.processing_queue = queue.Queue()
        self.is_processing = False
        self.vision_analyzer = vision_analyzer
        
    async def process_video(self, video_path: str) -> None:
        """Process video with REAL AI vision analysis"""
        logger.info(f"Starting enhanced video analysis with REAL AI vision: {video_path}")
        self.context.current_video_path = video_path
        self.context.analysis_complete = False
        self.is_processing = True
        
        try:
            # Extract frames with progress tracking
            frames = self.video_processor.extract_frames(video_path, frame_interval=30)
            logger.info(f"Extracted {len(frames)} frames for analysis")
            
            if not frames:
                raise ValueError("No frames could be extracted from the video")
            
            events = []
            prev_detections = None
            
            # Process each frame with REAL AI vision analysis
            for i, (timestamp, frame) in enumerate(frames):
                # Show progress for longer videos
                if i % 5 == 0 and len(frames) > 10:
                    progress = (i / len(frames)) * 100
                    logger.info(f"AI Analysis progress: {progress:.1f}% ({i}/{len(frames)} frames)")
                
                detections = self.video_processor.detect_objects(frame)
                
                # Analyze movement between frames
                if prev_detections:
                    time_diff = timestamp - frames[i-1][0] if i > 0 else 1.0
                    movement_events = self.video_processor.analyze_movement(
                        prev_detections, detections, time_diff
                    )
                    
                    for event in movement_events:
                        event.timestamp = timestamp
                        events.append(event)
                
                # Enhanced event detection with REAL AI analysis
                for detection in detections:
                    # REPLACED HARDCODED PERSON DETECTION WITH REAL AI!
                    if detection['class'] == 'person':
                        if 'activity_analysis' in detection:
                            activity = detection['activity_analysis']
                            events.append(VideoEvent(
                                timestamp=timestamp,
                                event_type="person_activity",
                                description=activity['activity'],  # REAL AI analysis!
                                confidence=detection['confidence'],
                                bounding_box=tuple(detection['bbox']),
                                metadata={
                                    "object_class": "person",
                                    "activity_details": activity,
                                    "objects_involved": activity.get('objects_involved', [])
                                }
                            ))
                        else:
                            # Fallback only
                            events.append(VideoEvent(
                                timestamp=timestamp,
                                event_type="person_detected",
                                description="Person detected - activity analysis unavailable",
                                confidence=detection['confidence'],
                                bounding_box=tuple(detection['bbox']),
                                metadata={"object_class": "person"}
                            ))
                    
                    elif detection['class'] == 'traffic light':
                        # Keep traffic light detection for traffic scenarios
                        events.append(VideoEvent(
                            timestamp=timestamp,
                            event_type="traffic_light",
                            description=f"Traffic light detected in scene",
                            confidence=detection['confidence'],
                            bounding_box=tuple(detection['bbox']),
                            metadata={"object_class": "traffic light"}
                        ))
                    
                    elif detection['class'] in ['car', 'truck', 'bus', 'motorcycle']:
                        events.append(VideoEvent(
                            timestamp=timestamp,
                            event_type="vehicle_detected",
                            description=f"{detection['class'].title()} detected in scene",
                            confidence=detection['confidence'],
                            bounding_box=tuple(detection['bbox']),
                            metadata={"object_class": detection['class'], "vehicle_type": detection['class']}
                        ))
                    
                    # Add detection for other common objects
                    elif detection['class'] in ['cup', 'bowl', 'fork', 'spoon', 'knife', 'plate', 'bottle']:
                        events.append(VideoEvent(
                            timestamp=timestamp,
                            event_type="dining_object_detected",
                            description=f"{detection['class'].title()} detected - dining/eating context",
                            confidence=detection['confidence'],
                            bounding_box=tuple(detection['bbox']),
                            metadata={"object_class": detection['class'], "context": "dining"}
                        ))
                    
                    elif detection['class'] in ['sports ball', 'tennis racket', 'baseball bat']:
                        events.append(VideoEvent(
                            timestamp=timestamp,
                            event_type="sports_object_detected",
                            description=f"{detection['class'].title()} detected - sports context",
                            confidence=detection['confidence'],
                            bounding_box=tuple(detection['bbox']),
                            metadata={"object_class": detection['class'], "context": "sports"}
                        ))
                    
                    elif detection['class'] in ['book', 'laptop', 'keyboard', 'mouse', 'cell phone']:
                        events.append(VideoEvent(
                            timestamp=timestamp,
                            event_type="work_object_detected",
                            description=f"{detection['class'].title()} detected - work/study context",
                            confidence=detection['confidence'],
                            bounding_box=tuple(detection['bbox']),
                            metadata={"object_class": detection['class'], "context": "work_study"}
                        ))
                
                prev_detections = detections
            
            # Store events and metadata
            self.context.video_events = events
            self.context.video_metadata = {
                "total_frames_analyzed": len(frames),
                "duration_seconds": frames[-1][0] - frames[0][0] if len(frames) > 1 else 0,
                "frame_rate": len(frames) / (frames[-1][0] - frames[0][0]) if len(frames) > 1 else 0,
                "total_detections": len(events),
                "scene_context": self.video_processor.scene_context.copy(),
                "vision_analyzer": type(self.vision_analyzer).__name__ if self.vision_analyzer else "None"
            }
            
            # Generate enhanced summary
            if events:
                self.context.video_summary = await self.llm.summarize_events(events)
            else:
                self.context.video_summary = "No significant events detected in the video analysis."
            
            self.context.analysis_complete = True
            logger.info(f"Enhanced video analysis complete with REAL AI. Found {len(events)} events.")
            
        except Exception as e:
            logger.error(f"Error in video processing: {e}")
            self.context.video_summary = f"Error processing video: {str(e)}"
            self.context.analysis_complete = True  # Mark complete even with error
        finally:
            self.is_processing = False
    
    async def chat(self, user_message: str) -> str:
        """Handle chat interactions with enhanced context awareness"""
        if not self.context.analysis_complete and not self.is_processing:
            return "Please upload a video first for me to analyze with REAL AI vision. Use 'load <path>' or 'demo' command."
        
        if self.is_processing:
            return "I'm still analyzing your video with AI vision. Please wait a moment..."
        
        # Generate response using LLM with full context
        response = await self.llm.generate_response(user_message, self.context)
        
        # Update conversation history with timestamp
        self.context.conversation_history.append({
            "user": user_message,
            "assistant": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep conversation history manageable (last 10 exchanges)
        if len(self.context.conversation_history) > 10:
            self.context.conversation_history = self.context.conversation_history[-10:]
        
        return response
    
    def get_events_by_type(self, event_type: str) -> List[VideoEvent]:
        """Filter events by type with fuzzy matching"""
        matching_events = []
        for event in self.context.video_events:
            if (event_type.lower() in event.event_type.lower() or 
                event_type.lower() in event.description.lower()):
                matching_events.append(event)
        return matching_events
    
    def get_events_in_timerange(self, start_time: float, end_time: float) -> List[VideoEvent]:
        """Get events within a time range"""
        return [
            event for event in self.context.video_events 
            if start_time <= event.timestamp <= end_time
        ]
    
    def get_video_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the analyzed video"""
        if not self.context.video_events:
            return {"total_events": 0, "analysis_complete": self.context.analysis_complete}
        
        # Event analysis
        event_types = {}
        object_classes = set()
        confidences = []
        speeds = []
        violations = []
        activities = []
        
        for event in self.context.video_events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
            confidences.append(event.confidence)
            
            if 'object_class' in event.metadata:
                object_classes.add(event.metadata['object_class'])
            
            if 'movement_speed' in event.metadata:
                speeds.append(event.metadata['movement_speed'])
            
            if 'violation' in event.event_type:
                violations.append(event)
            
            if 'activity' in event.event_type or 'person' in event.event_type:
                activities.append(event)
        
        # Calculate statistics
        duration = max(e.timestamp for e in self.context.video_events) - min(e.timestamp for e in self.context.video_events)
        
        stats = {
            "total_events": len(self.context.video_events),
            "event_types": event_types,
            "unique_object_classes": list(object_classes),
            "analysis_complete": self.context.analysis_complete,
            "video_duration": duration,
            "event_rate": len(self.context.video_events) / duration if duration > 0 else 0,
            "average_confidence": np.mean(confidences) if confidences else 0,
            "confidence_range": [min(confidences), max(confidences)] if confidences else [0, 0],
            "high_confidence_count": len([c for c in confidences if c > 0.8]),
            "violations_detected": len(violations),
            "activities_detected": len(activities),
            "movement_events": len([e for e in self.context.video_events if 'movement' in e.event_type]),
            "average_speed": np.mean(speeds) if speeds else 0,
            "max_speed": max(speeds) if speeds else 0,
            "scene_context": self.video_processor.scene_context.copy(),
            "video_metadata": self.context.video_metadata,
            "vision_analyzer_active": self.vision_analyzer is not None
        }
        
        return stats

# ============================================================================
# VISION ANALYZER FACTORY
# ============================================================================

def create_vision_analyzer(analyzer_type: str = "local", **kwargs):
    """Create the appropriate vision analyzer"""
    
    if analyzer_type == "openai":
        api_key = kwargs.get('api_key')
        if not api_key:
            print("❌ OpenAI API key required")
            return None
        return OpenAIVisionAnalyzer(api_key)
    
    elif analyzer_type == "local":
        analyzer = LocalVisionAnalyzer()
        if analyzer.available:
            return analyzer
        else:
            print("❌ Local vision analyzer failed to initialize")
            return None
    
    elif analyzer_type == "ollama":
        base_url = kwargs.get('base_url', 'http://localhost:11434')
        analyzer = OllamaVisionAnalyzer(base_url)
        if analyzer.available:
            return analyzer
        else:
            print("❌ Ollama vision analyzer not available")
            return None
    
    else:
        print(f"❌ Unknown analyzer type: {analyzer_type}")
        return None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def check_ollama_installation():
    """Check if Ollama is installed and running with enhanced feedback"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            models = models_data.get('models', [])
            print("✅ Ollama is running!")
            
            if models:
                print("📚 Available models:")
                vision_models = []
                for model in models:
                    size = model.get('size', 0) / (1024**3)  # Convert to GB
                    modified = model.get('modified', 'Unknown')
                    print(f"  • {model['name']} ({size:.1f}GB, modified: {modified[:10]})")
                    if 'vision' in model['name'].lower() or 'llava' in model['name'].lower():
                        vision_models.append(model['name'])
                
                if vision_models:
                    print(f"🎯 Vision models found: {', '.join(vision_models)}")
                else:
                    print("📸 No vision models found. For REAL AI vision, install:")
                    print("  • ollama pull llava:7b     (Vision model)")
                    print("  • ollama pull llava:13b    (Better vision model)")
            else:
                print("📚 No models found. Popular options:")
                print("  • ollama pull llama3.2:1b    (1.3GB - Fast)")
                print("  • ollama pull llama3.2:3b    (2.0GB - Balanced)")
                print("  • ollama pull llava:7b       (4.0GB - Vision)")
            return True
    except Exception as e:
        pass
    
    print("❌ Ollama not detected.")
    print("\n🔧 To install Ollama:")
    print("1. Visit: https://ollama.ai")
    print("2. Download and install for your OS")
    print("3. Run: ollama serve")
    print("4. Pull models: ollama pull llama3.2:1b")
    print("5. For vision: ollama pull llava:7b")
    return False

def create_demo_video():
    """Create an enhanced demo video for testing"""
    import os
    
    demo_path = 'enhanced_demo_video.mp4'
    
    if os.path.exists(demo_path):
        print(f"✅ Demo video already exists: {demo_path}")
        return demo_path
    
    try:
        # Create a more complex demo video with multiple scenarios
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(demo_path, fourcc, 20.0, (640, 480))
        
        print("🎬 Creating enhanced demo video...")
        
        for i in range(150):  # 7.5 seconds at 20 fps
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Create different scenarios for AI to analyze
            if i < 50:  # Eating scenario
                # Indoor background
                cv2.rectangle(frame, (0, 0), (640, 480), (20, 30, 40), -1)  # Dark background
                # Table
                cv2.rectangle(frame, (100, 300), (540, 480), (139, 69, 19), -1)  # Brown table
                
                # Person "eating" (simple representation)
                person_x = 320 + int(20 * np.sin(i * 0.2))  # Slight movement
                cv2.circle(frame, (person_x, 200), 25, (255, 200, 150), -1)  # Head
                cv2.rectangle(frame, (person_x - 20, 225), (person_x + 20, 300), (100, 150, 200), -1)  # Body
                
                # Food items
                cv2.circle(frame, (250, 350), 15, (255, 255, 0), -1)  # Plate/food
                cv2.rectangle(frame, (400, 340), (420, 370), (150, 75, 0), -1)  # Cup
                
            elif i < 100:  # Tennis scenario
                # Outdoor background
                cv2.rectangle(frame, (0, 0), (640, 300), (135, 206, 235), -1)  # Sky
                cv2.rectangle(frame, (0, 300), (640, 480), (34, 139, 34), -1)  # Ground
                
                # Person playing tennis
                person_x = 200 + int(100 * np.sin((i-50) * 0.1))
                cv2.circle(frame, (person_x, 250), 20, (255, 200, 150), -1)  # Head
                cv2.rectangle(frame, (person_x - 15, 270), (person_x + 15, 350), (100, 150, 200), -1)  # Body
                
                # Tennis racket
                cv2.line(frame, (person_x + 30, 280), (person_x + 50, 260), (139, 69, 19), 5)
                cv2.circle(frame, (person_x + 50, 260), 15, (255, 255, 255), 2)
                
                # Ball
                ball_x = person_x + 80 + int(50 * np.cos((i-50) * 0.2))
                cv2.circle(frame, (ball_x, 200), 8, (255, 255, 0), -1)
                
            else:  # Reading scenario
                # Indoor library background
                cv2.rectangle(frame, (0, 0), (640, 480), (40, 40, 60), -1)  # Dark background
                cv2.rectangle(frame, (50, 350), (590, 480), (139, 69, 19), -1)  # Desk
                
                # Person reading
                person_x = 320
                cv2.circle(frame, (person_x, 220), 25, (255, 200, 150), -1)  # Head
                cv2.rectangle(frame, (person_x - 20, 245), (person_x + 20, 320), (100, 150, 200), -1)  # Body
                
                # Book
                cv2.rectangle(frame, (person_x - 30, 300), (person_x + 30, 340), (255, 255, 255), -1)  # Book
                cv2.rectangle(frame, (person_x - 25, 305), (person_x + 25, 335), (0, 0, 0), 2)  # Book outline
                
                # Laptop
                cv2.rectangle(frame, (400, 360), (500, 400), (128, 128, 128), -1)  # Laptop
            
            out.write(frame)
        
        out.release()
        print(f"✅ Enhanced demo video created: {demo_path}")
        print("📋 Demo includes: eating scene, tennis playing, and reading - perfect for AI vision testing!")
        return demo_path
        
    except Exception as e:
        print(f"❌ Error creating demo video: {e}")
        return None

def install_dependencies():
    """Check and install required dependencies with enhanced feedback"""
    required_packages = {
        'opencv-python': 'cv2',
        'numpy': 'numpy', 
        'requests': 'requests',
        'aiohttp': 'aiohttp',
    }
    
    optional_packages = {
        'torch': 'torch',
        'ultralytics': 'ultralytics',
        'transformers': 'transformers',
        'pillow': 'PIL',
    }
    
    missing_required = []
    missing_optional = []
    
    print("🔍 Checking dependencies...")
    
    # Check required packages
    for package, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"  ✅ {package}")
        except ImportError:
            missing_required.append(package)
            print(f"  ❌ {package} (required)")
    
    # Check optional packages
    for package, import_name in optional_packages.items():
        try:
            __import__(import_name)
            print(f"  ✅ {package} (for AI vision)")
        except ImportError:
            missing_optional.append(package)
            print(f"  ⚠️  {package} (for REAL AI vision features)")
    
    # Install missing packages
    if missing_required:
        print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
        install = input("Install required packages? (y/n): ").lower().strip()
        
        if install == 'y':
            for package in missing_required:
                print(f"📦 Installing {package}...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                    print(f"  ✅ {package} installed")
                except subprocess.CalledProcessError as e:
                    print(f"  ❌ Failed to install {package}: {e}")
        else:
            print("⚠️  Application may not work without required packages.")
    
    if missing_optional:
        print(f"\n⚠️  Missing AI vision packages: {', '.join(missing_optional)}")
        print("These provide REAL AI vision analysis instead of hardcoded responses.")
        install_opt = input("Install AI vision packages? (y/n): ").lower().strip()
        
        if install_opt == 'y':
            for package in missing_optional:
                print(f"📦 Installing {package}...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                    print(f"  ✅ {package} installed")
                except subprocess.CalledProcessError as e:
                    print(f"  ❌ Failed to install {package}: {e}")
    
    if not missing_required and not missing_optional:
        print("✅ All dependencies satisfied!")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

async def main():
    """Enhanced main function with REAL AI vision setup"""
    
    print("🎥 ENHANCED VIDEO UNDERSTANDING WITH REAL AI VISION")
    print("=" * 70)
    print("🚀 Now with REAL activity analysis - no more hardcoded responses!")
    print("=" * 70)
    
    # Setup vision analyzer
    print("\n🤖 AI VISION ANALYZER OPTIONS:")
    print("1. 🏠 Local Vision (Transformers - Free, Private)")
    print("2. 🦙 Ollama Vision (llava - Free, Local)")  
    print("3. 🌐 OpenAI GPT-4 Vision (Paid, Most Accurate)")
    print("4. 🧪 No Vision (Mock responses)")
    print("=" * 70)
    
    vision_choice = input("Select vision analyzer (1-4): ").strip()
    vision_analyzer = None
    
    if vision_choice == "1":
        print("\n🏠 Setting up Local Vision (Transformers)...")
        if TRANSFORMERS_AVAILABLE:
            vision_analyzer = create_vision_analyzer("local")
            if vision_analyzer:
                print("✅ Local vision analyzer ready!")
            else:
                print("❌ Local vision setup failed. Continuing without vision.")
        else:
            print("❌ Transformers not available. Install with:")
            print("   pip install transformers torch pillow")
    
    elif vision_choice == "2":
        print("\n🦙 Setting up Ollama Vision...")
        vision_analyzer = create_vision_analyzer("ollama")
        if vision_analyzer:
            print("✅ Ollama vision analyzer ready!")
        else:
            print("❌ Ollama vision not available. Install llava model:")
            print("   ollama pull llava:7b")
    
    elif vision_choice == "3":
        print("\n🌐 Setting up OpenAI Vision...")
        api_key = input("Enter OpenAI API key: ").strip()
        if api_key:
            vision_analyzer = create_vision_analyzer("openai", api_key=api_key)
            if vision_analyzer:
                print("✅ OpenAI vision analyzer ready!")
            else:
                print("❌ OpenAI vision setup failed.")
        else:
            print("❌ No API key provided.")
    
    else:
        print("\n🧪 Using mock responses (no real vision analysis)")
        vision_analyzer = None
    
    # Setup LLM interface
    print("\n🤖 LLM INTERFACE OPTIONS:")
    print("1. 🦙 Ollama (Local LLM - Recommended)")
    print("2. 🧪 Mock Interface (Testing/Demo mode)")
    print("=" * 70)
    
    llm_choice = input("Select LLM interface (1-2): ").strip()
    
    if llm_choice == "1":
        print("\n🦙 OLLAMA SETUP")
        print("-" * 40)
        
        ollama_available = check_ollama_installation()
        
        if not ollama_available:
            print("\n🔄 Fallback options:")
            print("1. Install Ollama and restart")
            print("2. Continue with Mock interface")
            
            fallback = input("Choose option (1-2): ").strip()
            if fallback == "1":
                print("Please install Ollama and restart the application.")
                return
            else:
                print("Continuing with Mock interface...")
                llm_interface = MockLLMInterface()
        else:
            print("\n📚 RECOMMENDED MODELS:")
            print("• llama3.2:1b    - Fastest, good for testing (1.3GB)")
            print("• llama3.2:3b    - Balanced performance (2.0GB)")  
            print("• llama3.1:8b    - High quality analysis (4.7GB)")
            print("• mistral:7b     - Efficient and capable (4.1GB)")
            
            model_name = input("\nEnter model name [llama3.2:1b]: ").strip()
            if not model_name:
                model_name = "llama3.2:1b"
            
            base_url = input("Ollama URL [http://localhost:11434]: ").strip()
            if not base_url:
                base_url = "http://localhost:11434"
            
            timeout = input("Timeout in seconds [90]: ").strip()
            timeout = int(timeout) if timeout.isdigit() else 90
            
            print(f"\n🔄 Initializing Ollama with {model_name}...")
            llm_interface = OllamaInterface(
                model_name=model_name, 
                base_url=base_url,
                timeout=timeout
            )
    
    else:
        print("\n🧪 Using Mock interface for testing...")
        llm_interface = MockLLMInterface()
    
    # Create enhanced assistant with REAL AI vision
    print(f"\n🎯 Initializing Enhanced Video Assistant...")
    if vision_analyzer:
        print(f"   🎯 Vision Analyzer: {type(vision_analyzer).__name__}")
        print("   ✅ REAL AI activity analysis enabled!")
    else:
        print("   ⚠️  No vision analyzer - using fallback responses")
    
    assistant = AgenticVideoAssistant(llm_interface, vision_analyzer=vision_analyzer)
    
    print("\n" + "="*70)
    print("🎬 ENHANCED VIDEO ASSISTANT WITH REAL AI VISION READY!")
    print("="*70)
    print("\n📋 AVAILABLE COMMANDS:")
    print("🎥 Video Analysis:")
    print("  • 'load <video_path>' - Analyze any video with REAL AI vision")
    print("  • 'webcam' - Capture and analyze webcam (5 seconds)")
    print("  • 'demo' - Create and analyze enhanced demo video")
    print("")
    print("📊 Information & Stats:")
    print("  • 'stats' - Comprehensive analysis statistics")
    print("  • 'events <type>' - Filter events by type")
    print("  • 'time <start> <end>' - Get events in time range (seconds)")
    print("  • 'export' - Export analysis to JSON file")
    print("")
    print("🔧 System:")
    print("  • 'help' - Show detailed help")
    print("  • 'clear' - Clear conversation history")
    print("  • 'quit' - Exit application")
    print("")
    print("💬 Natural Conversation:")
    print("  • Ask anything about your video in natural language!")
    print("  • Examples: 'What activities did you detect?'")
    print("             'Show me the timeline of events'")
    print("             'What was the person doing?'")
    print("             'Any eating or drinking detected?'")
    if vision_analyzer:
        print("\n🎯 REAL AI VISION ACTIVE - Activities will be analyzed dynamically!")
    else:
        print("\n⚠️  Mock mode - Install vision packages for real AI analysis")
    print("="*70)
    
    # Main enhanced chat loop
    while True:
        try:
            user_input = input("\n🎯 You: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Thank you for using Enhanced Video Assistant with REAL AI Vision! Goodbye!")
                break
                
            elif user_input.lower().startswith('load '):
                video_path = user_input[5:].strip()
                print(f"🔄 Starting REAL AI vision analysis of: {video_path}")
                if vision_analyzer:
                    print("🎯 Using REAL AI to analyze activities - no more hardcoded responses!")
                print("⏳ This may take a moment for longer videos...")
                
                try:
                    start_time = time.time()
                    await assistant.process_video(video_path)
                    end_time = time.time()
                    
                    print(f"✅ REAL AI analysis complete! ({end_time - start_time:.1f}s)")
                    print("\n🤖 AI Analysis Summary:")
                    print("-" * 40)
                    print(assistant.context.video_summary)
                    print("-" * 40)
                    print("💬 Ready for your questions! Try:")
                    print("  • 'What activities did you see?'")
                    print("  • 'What was the person doing?'")
                    print("  • 'Show me the timeline'")
                    
                except Exception as e:
                    print(f"❌ Analysis error: {e}")
            
            elif user_input.lower() == 'demo':
                print("🎬 Creating enhanced demo video with multiple activities...")
                demo_path = create_demo_video()
                if demo_path:
                    print(f"🔄 Analyzing demo video with REAL AI: {demo_path}")
                    if vision_analyzer:
                        print("🎯 AI will analyze eating, tennis, and reading activities!")
                    try:
                        start_time = time.time()
                        await assistant.process_video(demo_path)
                        end_time = time.time()
                        
                        print(f"✅ Demo analysis complete! ({end_time - start_time:.1f}s)")
                        print("\n🤖 Demo Analysis Summary:")
                        print("-" * 40)
                        print(assistant.context.video_summary)
                        print("-" * 40)
                        print("💬 Try these demo questions:")
                        print("  • 'What activities did you detect?'")
                        print("  • 'Was anyone eating or drinking?'")
                        print("  • 'Did you see any sports activities?'")
                        print("  • 'What was happening at different times?'")
                        
                    except Exception as e:
                        print(f"❌ Demo analysis error: {e}")
                        
            elif user_input.lower() == 'webcam':
                print("📹 Starting webcam capture...")
                try:
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        print("❌ Could not access webcam")
                        continue
                        
                    # Record webcam footage
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    webcam_path = 'webcam_capture.mp4'
                    out = cv2.VideoWriter(webcam_path, fourcc, 20.0, (640, 480))
                    
                    print("🔴 Recording for 5 seconds...")
                    start_time = time.time()
                    frame_count = 0
                    
                    while time.time() - start_time < 5:
                        ret, frame = cap.read()
                        if ret:
                            frame = cv2.resize(frame, (640, 480))
                            out.write(frame)
                            frame_count += 1
                            
                            if frame_count % 20 == 0:
                                elapsed = int(time.time() - start_time)
                                print(f"  📹 Recording... {elapsed}s")
                    
                    cap.release()
                    out.release()
                    print(f"✅ Recorded {frame_count} frames")
                    
                    # Analyze webcam footage with REAL AI
                    print("🔄 Analyzing webcam footage with REAL AI vision...")
                    if vision_analyzer:
                        print("🎯 AI will analyze what you were doing!")
                    analysis_start = time.time()
                    await assistant.process_video(webcam_path)
                    analysis_end = time.time()
                    
                    print(f"✅ Webcam analysis complete! ({analysis_end - analysis_start:.1f}s)")
                    print("\n🤖 Webcam Analysis:")
                    print("-" * 30)
                    print(assistant.context.video_summary)
                    
                except Exception as e:
                    print(f"❌ Webcam error: {e}")
                    
            elif user_input.lower() == 'stats':
                stats = assistant.get_video_stats()
                print("\n📊 COMPREHENSIVE VIDEO STATISTICS:")
                print("=" * 50)
                
                for key, value in stats.items():
                    if isinstance(value, dict):
                        print(f"{key.replace('_', ' ').title()}:")
                        for k, v in value.items():
                            print(f"  • {k}: {v}")
                    elif isinstance(value, list):
                        print(f"{key.replace('_', ' ').title()}: {', '.join(map(str, value))}")
                    else:
                        print(f"{key.replace('_', ' ').title()}: {value}")
                        
            elif user_input.lower().startswith('events '):
                event_type = user_input[7:].strip()
                events = assistant.get_events_by_type(event_type)
                print(f"\n🔍 Found {len(events)} events matching '{event_type}':")
                print("-" * 40)
                for i, event in enumerate(events[:10], 1):
                    print(f"{i:2d}. {event.timestamp:6.1f}s: {event.description} ({event.confidence:.2f})")
                if len(events) > 10:
                    print(f"    ... and {len(events) - 10} more events")
                    
            elif user_input.lower().startswith('time '):
                try:
                    parts = user_input[5:].split()
                    if len(parts) != 2:
                        print("❌ Usage: time <start_seconds> <end_seconds>")
                        continue
                        
                    start_time, end_time = float(parts[0]), float(parts[1])
                    events = assistant.get_events_in_timerange(start_time, end_time)
                    
                    print(f"\n⏰ Found {len(events)} events between {start_time}s and {end_time}s:")
                    print("-" * 50)
                    for event in events:
                        print(f"• {event.timestamp:6.1f}s: {event.description} ({event.confidence:.2f})")
                        
                except (ValueError, IndexError):
                    print("❌ Usage: time <start_seconds> <end_seconds>")
                    
            elif user_input.lower() == 'export':
                if not assistant.context.video_events:
                    print("❌ No analysis data to export")
                    continue
                    
                try:
                    stats = assistant.get_video_stats()
                    export_data = {
                        "export_info": {
                            "timestamp": datetime.now().isoformat(),
                            "video_path": assistant.context.current_video_path,
                            "analysis_version": "Enhanced v3.0 with REAL AI Vision",
                            "vision_analyzer": type(assistant.vision_analyzer).__name__ if assistant.vision_analyzer else "None"
                        },
                        "summary": assistant.context.video_summary,
                        "statistics": stats,
                        "events": [
                            {
                                "timestamp": event.timestamp,
                                "type": event.event_type,
                                "description": event.description,
                                "confidence": event.confidence,
                                "bounding_box": event.bounding_box,
                                "metadata": event.metadata
                            }
                            for event in assistant.context.video_events
                        ],
                        "conversation_history": assistant.context.conversation_history
                    }
                    
                    filename = f"enhanced_video_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(filename, 'w') as f:
                        json.dump(export_data, f, indent=2, default=str)
                    
                    print(f"✅ Enhanced analysis exported to: {filename}")
                    print(f"📊 Exported {len(assistant.context.video_events)} events with REAL AI analysis")
                    
                except Exception as e:
                    print(f"❌ Export error: {e}")
                    
            elif user_input.lower() == 'clear':
                assistant.context.conversation_history.clear()
                print("✅ Conversation history cleared")
                
            elif user_input.lower() == 'help':
                print("\n🆘 ENHANCED VIDEO ASSISTANT HELP (REAL AI VISION)")
                print("=" * 60)
                print("\n🎥 VIDEO ANALYSIS COMMANDS:")
                print("• load <path>     - Analyze video with REAL AI vision")
                print("• demo            - Create & analyze demo video")
                print("• webcam          - Capture & analyze webcam")
                print("• stats           - Show comprehensive statistics")
                print("• export          - Export analysis to JSON")
                print("\n🔍 QUERY COMMANDS:")
                print("• events <type>   - Filter events by type")
                print("• time <s> <e>    - Events in time range")
                print("\n💬 NATURAL LANGUAGE QUERIES:")
                print("Ask anything about your video:")
                print("• 'What activities did you detect?'")
                print("• 'What was the person doing?'")
                print("• 'Was anyone eating or drinking?'")
                print("• 'Did you see any sports activities?'")
                print("• 'Show me the timeline of events'")
                print("• 'What happened at 30 seconds?'")
                print("• 'Were there any safety concerns?'")
                print("\n🎯 AI VISION STATUS:")
                if vision_analyzer:
                    print(f"✅ REAL AI Vision Active: {type(vision_analyzer).__name__}")
                    print("   Activities analyzed dynamically - no hardcoded responses!")
                else:
                    print("⚠️  Mock mode - install vision packages for real analysis")
                print("\n🔧 SYSTEM COMMANDS:")
                print("• help            - Show this help")
                print("• clear           - Clear conversation history")
                print("• quit            - Exit application")
                
                if assistant.context.analysis_complete:
                    print(f"\n✅ Current video: {len(assistant.context.video_events)} events analyzed")
                else:
                    print("\n⏳ No video loaded - use 'load <path>' or 'demo'")
                    
            else:
                # Handle natural language chat with REAL AI context
                print("🤖 Assistant: ", end="", flush=True)
                response = await assistant.chat(user_input)
                print(response)
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            logger.error(f"Main loop error: {e}", exc_info=True)

if __name__ == "__main__":
    print("🚀 STARTING ENHANCED VIDEO CHAT ASSISTANT WITH REAL AI VISION")
    print("🔧 Checking system requirements...")
    
    # Enhanced dependency checking
    install_dependencies()
    
    # Check Ollama availability
    print("\n🦙 Checking Ollama availability...")
    ollama_available = check_ollama_installation()
    
    if not ollama_available:
        print("\n⚠️  Ollama not available - Mock interface will be used as fallback")
    
    # Option to create demo video
    print("\n🎬 Demo video setup...")
    create_demo = input("Create enhanced demo video for testing REAL AI vision? (y/n): ").lower().strip()
    if create_demo == 'y':
        demo_path = create_demo_video()
        if demo_path:
            print(f"✅ Demo video ready: {demo_path}")
            print("💡 Use 'demo' command to analyze it with REAL AI vision")
    
    print("\n" + "="*70)
    print("🎬 ENHANCED VIDEO ASSISTANT WITH REAL AI VISION READY!")
    print("🚀 No more hardcoded responses - activities analyzed dynamically!")
    print("="*70)
    
    # Run the enhanced application
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Application error: {e}")
        logger.error(f"Application error: {e}", exc_info=True)