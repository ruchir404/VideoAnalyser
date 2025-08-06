# VideoAnalyser
# Enhanced Video Understanding Chat Assistant with Real AI Vision

A comprehensive agentic video analysis system that uses real AI vision models to understand and analyze video content, enabling natural language conversations about detected activities, objects, and events.

## 🎯 Project Overview

This project implements an intelligent video analysis assistant that combines computer vision, object detection, and large language models to provide conversational insights about video content. Unlike traditional hardcoded video analysis tools, this system uses real AI vision models to dynamically analyze activities like eating, sports, cooking, reading, and more.

### Key Features

- 🎥 Real AI Vision Analysis: Uses Transformers, Ollama Vision (llava), or OpenAI GPT-4 Vision for dynamic activity recognition
- 🤖 Natural Language Chat: Conversational interface powered by Ollama or mock LLM
- ⚡ Real-time Processing: Frame-by-frame analysis with object detection and tracking
- 📊 Comprehensive Analytics: Event detection, movement analysis, and timeline tracking
- 🔍 Flexible Querying: Ask questions about specific activities, timeframes, or patterns
- 📈 Export Capabilities: JSON export of complete analysis results
- 🎬 Demo Mode: Built-in demo video generation for testing

## 🏗️ Architecture Diagram


```
┌─────────────────────────────────────────────────────────────────┐
│                      User Interface Layer                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Commands      │  │  Natural Lang   │  │   Statistics    │ │
│  │   (load/demo)   │  │     Queries     │  │   & Export      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                  │
┌───────────────────────────────────────────────────────────────── ┐
│                 AgenticVideoAssistant (Main Controller)          │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              ConversationContext                            │ │
│  │  • Video Events  • Conversation History  • Metadata         │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                    │                              │
┌─────────────────────────────────┐    ┌─────────────────────────────────┐
│        Video Processing         │    │         LLM Interface           │
│  ┌─────────────────────────────┐│    │ ┌─────────────────────────────┐ │
│  │   EnhancedVideoProcessor    ││    │  │     OllamaInterface         │ │
│  │  • Frame Extraction         ││    │  │  • Response Generation      │ │
│  │  • Object Detection (YOLO)  ││    │  │  • Event Summarization      │ │
│  │  • Movement Analysis        ││    │  │  • Context Management       │ │
│  │  • Scene Context Tracking   ││    │  └─────────────────────────────┘ │
│  └─────────────────────────────┘│    │             OR                  │
└─────────────────────────────────┘    │  ┌─────────────────────────────┐ │
                    │                   │  │    MockLLMInterface         │ │
                    │                   │  │  • Fallback Responses       │ │
                    │                   │  │  • Rule-based Analysis      │ │
┌──────────────────────────────────┐    │  └─────────────────────────────┘ │
│       AI Vision Analysis         │    └─────────────────────────────────┘
│  ┌─────────────────────────────┐ │
│  │    LocalVisionAnalyzer      │ │
│  │  • BLIP Image Captioning    │ │
│  │  • Activity Interpretation  │ │
│  └─────────────────────────────┘ │
│             OR                   |
│  ┌─────────────────────────────┐ │
│  │   OllamaVisionAnalyzer      │ │
│  │  • llava Vision Model       │ │
│  │  • Local Processing         │ │
│  └─────────────────────────────┘ │
│             OR                   │
│  ┌─────────────────────────────┐ │
│  │   OpenAIVisionAnalyzer      │ │
│  │  • GPT-4 Vision API         │ │
│  │  • High Accuracy Analysis   │ │
│  └─────────────────────────────┘ │
└──────────────────────────────────┘
                    │
┌──────────────────────────────────┐
│          Data Layer              │
│  ┌─────────────────────────────┐ │
│  │       VideoEvent            │ │
│  │  • Timestamp                │ │
│  │  • Event Type               │ │
│  │  • Description              │ │
│  │  • Confidence               │ │
│  │  • Bounding Box             │ │
│  │  • Metadata                 │ │
│  └─────────────────────────────┘ │
└──────────────────────────────────┘
```

## 💻 Tech Stack Justification

### Backend Technologies

#### Computer Vision & AI
- OpenCV (`cv2`): Industry-standard library for video processing, frame extraction, and basic computer vision operations
- YOLO (Ultralytics): State-of-the-art real-time object detection for identifying people, vehicles, and objects
- Transformers (Hugging Face): Provides BLIP model for image captioning and local AI vision analysis
- PyTorch: Deep learning framework supporting the vision models

#### AI Vision Models
1. Local Vision (Transformers + BLIP)
   - Pros: Free, private, works offline
   - Cons: Lower accuracy than cloud models
   - Use case: Privacy-focused deployments

2. Ollama Vision (llava)
   - Pros: Free, local processing, good accuracy
   - Cons: Requires model download, GPU recommended
   - Use case: Balance of privacy and performance

3. OpenAI GPT-4 Vision
   - Pros: Highest accuracy, detailed descriptions
   - Cons: Paid API, requires internet
   - Use case: Production systems requiring best results

#### Large Language Models
- Ollama: Local LLM inference with models like Llama 3.2, provides conversational AI without cloud dependencies
- aiohttp: Async HTTP client for efficient API communication
- requests: HTTP library for synchronous API calls

#### Data Processing
- NumPy: Efficient numerical computations for video frame processing
- AsyncIO: Asynchronous programming for responsive user interaction
- JSON: Structured data export and configuration management

### Architecture Benefits

1. Modular Design: Separate vision analyzers and LLM interfaces allow easy swapping of AI models
2. Real AI Integration: No hardcoded responses - actual AI models analyze video content
3. Scalable Processing: Frame-by-frame analysis with configurable intervals
4. Context Awareness: Maintains conversation state and video analysis context
5. Fallback Systems: Graceful degradation when AI services are unavailable

## 🚀 Setup and Installation Instructions

### Prerequisites

- Python 3.8+
- 4GB+ RAM (8GB+ recommended for vision models)
- Webcam (optional, for live capture)
- GPU (optional, improves vision model performance)

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd video-understanding-assistant

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install core dependencies
pip install opencv-python numpy requests aiohttp
```

### 2. Install AI Vision Dependencies (Optional but Recommended)

For REAL AI vision analysis, install vision packages:

```bash
# For local vision analysis
pip install transformers torch pillow

# For enhanced object detection
pip install ultralytics
```

### 3. Setup Ollama (Recommended for LLM)

#### Install Ollama:
1. Visit [https://ollama.ai](https://ollama.ai)
2. Download for your operating system
3. Install and start Ollama

#### Install Models:
```bash
# Fast model (1.3GB) - Good for testing
ollama pull llama3.2:1b

# Balanced model (2.0GB) - Recommended
ollama pull llama3.2:3b

# Vision model (4GB) - For Ollama vision analysis
ollama pull llava:7b

# Start Ollama server
ollama serve
```

### 4. Setup OpenAI Vision (Optional)

If using OpenAI GPT-4 Vision:
1. Get API key from [OpenAI](https://platform.openai.com/api-keys)
2. Have API key ready when prompted by the application

### 5. Verify Installation

Run the dependency checker:
```bash
python mainfile.py
```

The application will check dependencies and guide you through any missing packages.

## 📖 Usage Instructions

### Starting the Application

```bash
python mainfile.py
```

### 1. Choose AI Vision Analyzer

When starting, you'll be prompted to select a vision analyzer:

```
🤖 AI VISION ANALYZER OPTIONS:
1. 🏠 Local Vision (Transformers - Free, Private)
2. 🦙 Ollama Vision (llava - Free, Local)
3. 🌐 OpenAI GPT-4 Vision (Paid, Most Accurate)
4. 🧪 No Vision (Mock responses)
```

Recommendations:
- Beginners: Start with option 4 (No Vision) for quick testing
- Privacy-focused: Use option 1 (Local Vision)
- Best balance: Use option 2 (Ollama Vision) if you have llava installed
- Best accuracy: Use option 3 (OpenAI Vision) for production use

### 2. Choose LLM Interface

Select your language model interface:

```
🤖 LLM INTERFACE OPTIONS:
1. 🦙 Ollama (Local LLM - Recommended)
2. 🧪 Mock Interface (Testing/Demo mode)
```

### 3. Video Analysis Commands

#### Load and Analyze Video
```bash
load /path/to/your/video.mp4
```

Example:
```
🎯 You: load /Users/john/Documents/cooking_video.mp4
🔄 Starting REAL AI vision analysis of: /Users/john/Documents/cooking_video.mp4
🎯 Using REAL AI to analyze activities - no more hardcoded responses!
⏳ This may take a moment for longer videos...
✅ REAL AI analysis complete! (45.2s)
```

#### Create Demo Video
```bash
demo
```

What it does:
- Creates a test video with eating, tennis, and reading scenes
- Perfect for testing the AI vision capabilities
- Analyzes the demo with real AI vision

#### Webcam Capture
```bash
webcam
```

What it does:
- Captures 5 seconds of webcam footage
- Analyzes what you're doing in real-time
- Great for testing personal activities

### 4. Natural Language Queries

After analyzing a video, ask questions in natural language:

#### Activity Analysis
```
🎯 You: What activities did you detect in the video?
🤖 Assistant: I detected several activities using real AI vision:
• 5.2s: Person eating - individual appears to be consuming food from a plate
• 15.7s: Person drinking - taking a sip from what appears to be a cup
• 28.4s: Person cooking - working at a kitchen counter with cooking utensils
• 42.1s: Person reading - looking at what appears to be a book or document
```

#### Timeline Queries
```
🎯 You: What happened between 20 and 30 seconds?
🤖 Assistant: Between 20-30 seconds, I detected:
• 22.3s: Person standing near kitchen counter
• 24.8s: Person handling cooking utensils
• 27.2s: Person stirring something in a pan
```

#### Safety and Violations
```
🎯 You: Were there any safety concerns?
🤖 Assistant: No safety violations detected. The video shows normal domestic activities with appropriate behavior patterns.
```

#### Object Detection
```
🎯 You: What objects did you see in the video?
🤖 Assistant: I identified these objects:
• Kitchen utensils (spoon, pan, knife)
• Food items (plate with food, cup)
• Furniture (table, chair)
• Reading materials (book, papers)
```

### 5. Information Commands

#### Statistics
```bash
stats
```

Shows comprehensive analysis:
- Total events detected
- Event types breakdown
- Confidence levels
- Activity analysis
- Movement patterns
- Object detection summary

#### Event Filtering
```bash
events person_activity
events movement
events dining_object
```

#### Time Range Analysis
```bash
time 10 25
```

Finds all events between 10 and 25 seconds

#### Export Analysis
```bash
export
```

Creates JSON file with:
- Complete event timeline
- AI analysis results
- Conversation history
- Statistics and metadata

### 6. Advanced Usage Examples

#### Cooking Video Analysis
```
🎯 You: load cooking_tutorial.mp4
✅ Analysis complete!

🎯 You: What cooking techniques did you observe?
🤖 Assistant: I observed several cooking techniques:
• Chopping vegetables with a knife
• Sautéing ingredients in a pan
• Stirring and mixing motions
• Plating and presentation activities
```

#### Sports Video Analysis
```
🎯 You: load tennis_match.mp4
✅ Analysis complete!

🎯 You: Describe the tennis playing I saw
🤖 Assistant: Tennis analysis shows:
• Player serving with racket overhead motion
• Ball tracking across the court
• Running and positioning movements
• Backhand and forehand stroke techniques
```

#### Security Footage Analysis
```
🎯 You: load security_footage.mp4
✅ Analysis complete!

🎯 You: Any unusual activities or violations?
🤖 Assistant: Security analysis reveals:
• Normal pedestrian traffic patterns
• No violations detected
• Consistent movement speeds
• No unusual loitering or suspicious behavior
```

## 🎮 Interactive Commands Reference

| Command | Description | Example |
|---------|-------------|---------|
| `load <path>` | Analyze video file | `load video.mp4` |
| `demo` | Create and analyze demo video | `demo` |
| `webcam` | Capture and analyze webcam | `webcam` |
| `stats` | Show comprehensive statistics | `stats` |
| `events <type>` | Filter events by type | `events person_activity` |
| `time <start> <end>` | Events in time range | `time 10 30` |
| `export` | Export analysis to JSON | `export` |
| `help` | Show detailed help | `help` |
| `clear` | Clear conversation history | `clear` |
| `quit` | Exit application | `quit` |

## 🤖 AI Vision Capabilities

### What the AI Can Detect and Analyze

#### Human Activities
- Eating & Drinking: Recognizes food consumption, beverage sipping
- Sports: Tennis, running, exercise movements
- Work Activities: Computer use, reading, writing
- Cooking: Food preparation, kitchen activities
- Entertainment: Dancing, music, leisure activities

#### Objects and Context
- Kitchen Items: Plates, cups, utensils, appliances
- Sports Equipment: Balls, rackets, sports gear
- Technology: Laptops, phones, keyboards
- Furniture: Tables, chairs, work surfaces
- Vehicles: Cars, trucks, motorcycles

#### Movement Analysis
- Speed Detection: Fast, medium, slow movement classification
- Direction Tracking: 8-directional movement analysis
- Activity Level: High, medium, low activity classification
- Pattern Recognition: Repetitive vs. random movements

## 📊 Output Examples

### Analysis Summary Example
```
📋 COMPREHENSIVE VIDEO ANALYSIS REPORT (REAL AI VISION)
=================================================================

📊 EXECUTIVE SUMMARY:
   • Total Events: 47
   • Duration: 120.3 seconds
   • Avg Confidence: 0.84/1.0
   • Event Rate: 0.39/sec
   • Activity Level: High

📋 EVENT BREAKDOWN:
   • Person Activity: 18 (38.3%)
   • Object Movement: 12 (25.5%)
   • Dining Object Detected: 8 (17.0%)
   • Vehicle Movement: 5 (10.6%)
   • Sports Object Detected: 4 (8.5%)

🎬 ACTIVITIES DETECTED (18 events):
   • 5.2s: Person eating - individual consuming food from plate
   • 15.7s: Person drinking - taking sip from cup
   • 28.4s: Person cooking - working at kitchen counter
   • 42.1s: Person reading - looking at book

👁️ OBJECTS DETECTED: person, cup, plate, spoon, book, chair

✅ No violations detected

💡 ANALYSIS QUALITY:
   • High confidence events: 35
   • Real AI Vision: ✅ Active
   • Reliability: Excellent
```

### Conversation Example
```
🎯 You: What was the most interesting activity you detected?

🤖 Assistant: The most interesting activity was the cooking sequence from 25-45 seconds. The AI vision detected:

1. Ingredient Preparation (25.3s): Person chopping vegetables with precise knife movements
2. Cooking Process (32.1s): Sautéing ingredients in a pan with proper stirring technique
3. Timing Coordination (38.7s): Multiple cooking actions happening simultaneously
4. Plating Activity (44.2s): Careful presentation and plating of the finished dish

What made this particularly notable was the AI's ability to distinguish between different cooking techniques and identify the progression from raw ingredients to finished meal. The confidence levels were high (0.85-0.92) throughout this sequence.

Would you like me to analyze any specific cooking techniques or time periods in more detail?
```

## 🔧 Troubleshooting

### Common Issues

#### 1. "Ollama not detected"
Solution:
```bash
# Install Ollama from https://ollama.ai
# Then run:
ollama serve

# Pull a model:
ollama pull llama3.2:1b
```

#### 2. "Local vision models failed to load"
Solution:
```bash
pip install transformers torch pillow
# Restart the application
```

#### 3. "YOLO model loading failed"
Solution:
```bash
pip install ultralytics
# The app will download yolov8n.pt automatically
```

#### 4. Video file not found
Solution:
- Check file path is correct
- Use forward slashes or raw strings: `r"C:\path\to\video.mp4"`
- Ensure video format is supported (mp4, avi, mov)

#### 5. Low accuracy results
Solutions:
- Use OpenAI Vision for best accuracy
- Ensure good video quality and lighting
- Use appropriate frame intervals (default: 30)

### Performance Optimization

#### For Better Speed:
- Use smaller models (llama3.2:1b)
- Increase frame intervals
- Use local vision instead of cloud APIs

#### For Better Accuracy:
- Use OpenAI GPT-4 Vision
- Use higher quality video input
- Decrease frame intervals for more samples

## 🚀 Advanced Features

### Custom Vision Analyzers
The system supports pluggable vision analyzers. You can add your own:

```python
class CustomVisionAnalyzer:
    def analyze_frame_activity(self, frame):
        # Your custom analysis logic
        return {
            "activity": "Custom detected activity",
            "confidence": 0.9,
            "objects_involved": ["custom_object"]
        }
```

### Batch Processing
Process multiple videos programmatically:

```python
videos = ["video1.mp4", "video2.mp4", "video3.mp4"]
for video in videos:
    await assistant.process_video(video)
    # Export results for each video
```

### Integration APIs
The assistant can be integrated into larger systems:

```python
# Get analysis results programmatically
stats = assistant.get_video_stats()
events = assistant.get_events_by_type("person_activity")
timeline = assistant.get_events_in_timerange(10, 30)
```

## 📝 Contributing

This project demonstrates real AI vision integration for video understanding. Areas for contribution:

1. Additional Vision Models: Integrate more vision APIs or models
2. Enhanced Activity Recognition: Improve activity classification
3. Performance Optimization: Speed up processing for longer videos
4. UI Development: Create web or GUI interface
5. Cloud Integration: Add cloud storage and processing options

## 📄 License

This project is provided as a demonstration of AI-powered video analysis capabilities. Please ensure compliance with model licenses and API terms when using in production.

---

🎯 Ready to analyze your videos with REAL AI vision? Run `python mainfile.py` and start exploring!

