# Real-Time Earrings Object Detection and Cost Evaluation System

[![Demo Video](https://img.shields.io/badge/Demo-Video-red)](https://youtu.be/l5GYNe5-2P8)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-8.0.146-green.svg)](https://github.com/ultralytics/ultralytics)
[![Flask](https://img.shields.io/badge/Flask-2.0+-lightgrey.svg)](https://flask.palletsprojects.com/)

## Project Overview

This project combines computer vision with business intelligence to create an intelligent earring component detection and cost analysis system. It can identify 50 different types of earring components in real-time and provide instant cost calculations, inventory information, and supplier details.

##  Features

- **Real-time Object Detection**: Uses YOLOv8 to detect earring components from images
- **Cost Evaluation**: Automatically calculates total cost based on detected components
- **Inventory Management**: Integrates with SQL Server database for real-time inventory tracking
- **Supplier Information**: Provides detailed supplier contact and pricing information
- **Web Interface**: User-friendly Flask web application with image upload capability
- **Multi-class Detection**: Recognizes 50 different earring component categories
- **Database Integration**: SQL Server backend for component and supplier data

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │   YOLOv8 Model  │    │  SQL Database   │
│   (Flask App)   │◄──►│   (best.pt)     │◄──►│  (earring DB)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Image Upload   │    │ Object Detection│    │ Cost Calculation│
│  & Display      │    │ & Classification│    │ & Supplier Info │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Project Structure

```
├── Demo/                          # Web application demo
│   └── ngrok http 5000/          # Flask app with ngrok tunnel
│       ├── app.py                # Main Flask application
│       ├── best.pt               # Trained YOLOv8 model
│       └── templates/            # HTML templates
├── Yolo/                         # YOLO training and data
│   ├── data/                     # Dataset configuration
│   │   ├── earring.yaml          # YOLO dataset config
│   │   └── classes.txt           # Class definitions
│   ├── Traning_records/          # Training results and models
│   └── yolo_v8/                  # Training notebooks
├── Kaggle/                       # Kaggle competition notebooks
├── Tool/                         # Utility tools and scripts
├── Video/                        # Demo videos and presentations
└── Img/                          # Sample images
```

## Quick Start
### Prerequisites

- Python 3.8 or higher
- SQL Server 2019 or higher
- CUDA-capable GPU (recommended for training)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Jewelry_Object_Detection_Cost_Eval
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Database Setup**
   - Install SQL Server 2022
   - Create database named `earring`
   - Import CSV files from `Demo/` directory:
     - `parts.csv` → `parts` table
     - `supplier_details.csv` → `supplier_details` table

4. **Update Database Connection**
   - Edit `Demo/ngrok http 5000/app.py`
   - Update `connection_database` variable with your server details

   
### Running the Application

1. **Start the Flask app**
   ```bash
   cd Demo/ngrok\ http\ 5000/
   python app.py
   ```

2. **Access the web interface**
   - Open browser and navigate to `http://localhost:5000`
   - Upload an image containing earring components
   - View detection results, cost analysis, and supplier information


## Supported Component Classes
The system can detect 50 different earring components including:
- **Pins & Hooks**: Black dot pins, silver hooks, golden hooks
- **Rings**: Green rings, white rings, yellow rings, black rings
- **Decorative Elements**: Flowers, pearls, gemstones, beads
- **Specialty Items**: Bear shapes, dinosaur shapes, fox faces
- **Materials**: Metal squares, polygons, plates


### Training Configuration
- **Model**: YOLOv8 (medium/large variants)
- **Dataset**: Custom earring component dataset
- **Classes**: 50 different component types
- **Image Size**: 480x480 pixels
- **Augmentation**: Flip, mixup, copy-paste


### YOLO Configuration (`earring.yaml`)
```yaml
train: /path/to/train/images/
val: /path/to/validation/images/
names:
  0: 01_black_dot_pin
  1: 02_sliver_hook
  # ... 48 more classes
```


### Flask Configuration
- Upload folder: `./static/`
- Allowed extensions: jpg, jpeg, png, gif
- Database connection string in `app.py`


## Performance Metrics
The trained model provides:
- **Confusion Matrix**: Classification accuracy visualization
- **Precision-Recall Curves**: Model performance analysis
- **F1 Score**: Balanced precision and recall
- **Training Logs**: Detailed training progress


## Development Tools & Technologies
- **Computer Vision**: YOLOv8 (Ultralytics), OpenCV for image processing and object detection
- **Machine Learning**: PyTorch backend, custom training pipelines, data augmentation techniques
- **Web Development**: Flask framework, HTML/CSS templates, responsive web interface
- **Database**: SQL Server integration with pyodbc, CSV data management
- **Data Annotation**: LabelImg for training dataset preparation and bounding box annotation
- **Development Environment**: Jupyter Notebooks for experimentation, training scripts, and model evaluation
- **Utilities**: Image preprocessing tools, training data split scripts, database connectivity testing


## Acknowledgments
- **Ultralytics**: YOLOv8 implementation
- **Flask**: Web framework
- **SQL Server**: Database management
- **OpenCV**: Computer vision library


## Support
For questions and support:
- Check the demo video: [YouTube Demo](https://youtu.be/l5GYNe5-2P8)
- Review training notebooks in `Yolo/yolo_v8/`
- Examine the Flask application in `Demo/ngrok http 5000/`


