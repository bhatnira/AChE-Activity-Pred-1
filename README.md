# AChE Activity Prediction Suite

A comprehensive molecular prediction platform for Acetylcholinesterase (AChE) inhibition analysis using advanced machine learning and graph neural networks.

## 🚀 Features

### 🧬 **Multiple Prediction Models**
- **Graph Neural Networks**: DeepChem GraphConvModel for molecular graph analysis
- **Circular Fingerprints**: TPOT-optimized models with LIME explanations
- **RDKit Descriptors**: Traditional molecular descriptor-based predictions

### 🔬 **Analysis Capabilities**
- **Single Molecule Prediction**: Individual SMILES analysis with confidence scores
- **Batch Processing**: Excel/CSV and SDF file support for high-throughput analysis
- **LIME Explanations**: Interpretable AI with feature importance visualization
- **Similarity Maps**: Atomic contribution analysis for graph models
- **IC50 Predictions**: Quantitative inhibition concentration estimates

### 📊 **User-Friendly Interface**
- **Streamlit Web Application**: Interactive, responsive web interface
- **Progress Tracking**: Real-time processing status for batch operations
- **Export Functionality**: CSV downloads with complete prediction data
- **Molecular Visualization**: Built-in structure rendering and analysis

## 🛠️ Installation & Setup

### Prerequisites
- **Docker** (recommended) or Python 3.10+
- **Git** for repository cloning
- **4GB+ RAM** for optimal performance

### Option 1: Docker Setup (Recommended)

1. **Clone the Repository**
   ```bash
   git clone https://github.com/bhatnira/AChE-Activity-Pred-1.git
   cd AChE-Activity-Pred-1
   ```

2. **Build and Run with Docker Compose**
   ```bash
   docker-compose up -d --build
   ```

3. **Access the Application**
   - Open your web browser and navigate to: `http://localhost:10000`
   - The application will be available on ports 8501-8504 as well

4. **Stop the Application**
   ```bash
   docker-compose down
   ```

### Option 2: Local Python Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/bhatnira/AChE-Activity-Pred-1.git
   cd AChE-Activity-Pred-1
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv ache_env
   source ache_env/bin/activate  # On Windows: ache_env\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   streamlit run main_app.py --server.port 10000
   ```

5. **Access the Application**
   - Open your web browser and navigate to: `http://localhost:10000`

## 🖥️ Application Interface

### **Main Dashboard**
- **Home**: Overview and navigation hub
- **Graph Neural Networks**: Advanced molecular graph analysis
- **Circular Fingerprints**: Traditional ML with interpretability
- **RDKit Descriptors**: Descriptor-based predictions

### **Prediction Modes**

#### **Single Molecule Analysis**
1. Enter a SMILES string (e.g., `CCO` for ethanol)
2. Select prediction confidence level
3. View results with IC50 estimates and activity classification
4. Download LIME explanations (where available)

#### **Batch Processing**
1. **Excel/CSV Files**: Upload files with SMILES column
2. **SDF Files**: Upload structure data files directly
3. **Progress Monitoring**: Real-time processing status
4. **Results Export**: Download complete analysis as CSV
5. **Individual LIME**: Access explanations for specific molecules

## 📁 Project Structure

```
AChE-Activity-Pred-1/
├── main_app.py                 # Main application launcher
├── app_graph_combined.py       # Graph neural network interface
├── app_circular.py             # Circular fingerprint interface
├── app_rdkit.py               # RDKit descriptor interface
├── app_launcher.py            # Application router
├── requirements.txt           # Python dependencies
├── docker-compose.yml         # Docker orchestration
├── Dockerfile                 # Container configuration
├── .streamlit/                # Streamlit configuration
├── models/                    # Pre-trained model files
│   ├── *.pkl                 # TPOT pipeline models
│   ├── GraphConv_model_files/ # Graph neural network models
│   └── checkpoint-2000/       # ChemBERTa model checkpoint
└── data/                      # Training and reference data
```

## 🔬 Model Information

### **Graph Neural Networks**
- **Architecture**: DeepChem GraphConvModel
- **Features**: Molecular graph representation learning
- **Output**: Classification + regression with atomic contributions
- **Interpretability**: Similarity maps and attention weights

### **Circular Fingerprints**
- **Method**: Enhanced RDKit Morgan fingerprints with fallback
- **ML Pipeline**: TPOT-optimized ensemble models
- **Features**: 3-tier robust fingerprint generation
- **Interpretability**: LIME feature importance analysis

### **RDKit Descriptors**
- **Features**: Traditional molecular descriptors
- **ML Pipeline**: TPOT-optimized classification/regression
- **Speed**: Fast predictions for large datasets
- **Reliability**: Well-established chemical descriptors

## 📊 Input Formats

### **SMILES Strings**
- Standard chemical notation (e.g., `c1ccccc1` for benzene)
- Support for complex organic molecules
- Automatic validation and preprocessing

### **SDF Files**
- Standard structure data format
- Multiple molecules per file
- Preserves 3D structural information

### **Excel/CSV Files**
- Must contain a column with SMILES strings
- Flexible column naming (auto-detection)
- Support for additional metadata columns

## 🎯 Use Cases

### **Drug Discovery**
- Screen compound libraries for AChE inhibition
- Lead optimization with structure-activity insights
- Virtual screening of chemical databases

### **Academic Research**
- Alzheimer's disease research
- Neurotransmitter system studies
- Computational chemistry education

### **Pharmaceutical Industry**
- Early-stage drug development
- Safety assessment and toxicology
- Regulatory submission support

## 🔧 Configuration

### **Environment Variables**
- `STREAMLIT_SERVER_PORT`: Application port (default: 10000)
- `STREAMLIT_SERVER_ADDRESS`: Server address (default: 0.0.0.0)

### **Model Configuration**
- Models are automatically loaded on application startup
- GPU acceleration available for graph neural networks
- Memory optimization for large batch processing

## 🚨 Troubleshooting

### **Common Issues**

#### **Port Already in Use**
```bash
# Check what's using the port
lsof -i :10000
# Kill the process or use a different port
docker-compose down && docker-compose up -d
```

#### **Memory Issues**
- Ensure at least 4GB RAM available
- For large batch files, process in smaller chunks
- Consider using Docker for better resource management

#### **Model Loading Errors**
- Verify all model files are present in the repository
- Check Docker container logs: `docker logs container_name`
- Restart the application: `docker-compose restart`

#### **Dependency Issues**
- Update pip: `pip install --upgrade pip`
- Clear cache: `pip cache purge`
- Use virtual environment for local setup

## 📝 Example Usage

### **Single Prediction**
```python
# Example SMILES for testing
smiles_examples = [
    "CCO",                    # Ethanol
    "c1ccccc1",              # Benzene  
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"    # Caffeine
]
```

### **Batch File Format**
```csv
SMILES,Compound_Name,MW
CCO,Ethanol,46.07
c1ccccc1,Benzene,78.11
CC(C)CC1=CC=C(C=C1)C(C)C(=O)O,Ibuprofen,206.28
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DeepChem**: Graph neural network implementations
- **RDKit**: Molecular informatics toolkit
- **TPOT**: Automated machine learning pipeline optimization
- **Streamlit**: Web application framework
- **LIME**: Local interpretable model-agnostic explanations

## 📞 Support

For questions, issues, or contributions:
- **GitHub Issues**: [Create an issue](https://github.com/bhatnira/AChE-Activity-Pred-1/issues)
- **Repository**: [AChE-Activity-Pred-1](https://github.com/bhatnira/AChE-Activity-Pred-1)

---

**⚡ Quick Start**: `git clone https://github.com/bhatnira/AChE-Activity-Pred-1.git && cd AChE-Activity-Pred-1 && docker-compose up -d --build`

**🌐 Access**: http://localhost:10000
