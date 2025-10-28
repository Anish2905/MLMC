# 🎨 AI Model Compression Studio - Web GUI

**Beautiful, user-friendly interface for compressing machine learning models**

---

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
pip install flask flask-cors torch torchvision psutil
```

### Step 2: Start the Server

```bash
python compression_server.py
```

You'll see:
```
====================================================================
  🚀 AI Model Compression Studio - Server Starting
====================================================================

  Server URL: http://localhost:5000
  Upload folder: ./uploads
  Output folder: ./compressed_output

  Press Ctrl+C to stop the server
====================================================================
```

### Step 3: Open in Browser

Navigate to: **http://localhost:5000**

That's it! 🎉

---

## 📖 How to Use

### 1. **Upload Your Model**

- **Drag & drop** your model file into the upload zone, OR
- **Click** the upload zone to browse files
- Supported formats: `.pth`, `.pt`, `.ckpt`, `.h5`, `.pb`
- Maximum file size: 500 MB

### 2. **Configure Settings**

**Model Type:**
- Select your model architecture (CNN, ResNet, Transformer, etc.)
- Enter number of classes (for CNN models)

**Compression Techniques** (select multiple):
- ✅ **Quantization-Aware Training (QAT)** - Recommended for all models
- ✅ **Mixed-Precision** - Keeps important layers in higher precision
- ✅ **Knowledge Distillation** - Creates smaller student model
- ⚪ **4-bit Quantization** - For LLMs (extreme compression)
- ⚪ **QLoRA** - For LLM fine-tuning

**Compression Target:**
- **Balanced** (Recommended) - Good compression with minimal accuracy loss
- **Maximum Compression** - Smallest size possible
- **Maximum Accuracy** - Prioritizes model performance
- **Maximum Speed** - Fastest inference

### 3. **Start Compression**

Click **"🎯 Start Compression"** and watch the magic happen!

You'll see:
- Real-time progress (5 stages)
- Live console logs
- Visual step indicators

### 4. **View Results**

After compression completes, you'll see:
- **Accuracy** - Model performance
- **Model Size** - Compressed file size
- **Speedup** - Inference speed improvement
- **Compression Ratio** - How much smaller

Plus beautiful **before/after comparison charts**!

### 5. **Download**

Three download options:
- 📥 **Compressed Model** - Ready to deploy
- 📄 **Report** - Detailed JSON report
- 📊 **Visualizations** - Coming soon!

---

## 🎯 Features

### Beautiful UI
- Modern, gradient design
- Smooth animations
- Responsive layout
- Real-time feedback

### Smart Compression
- Multiple techniques combined
- Automatic optimization
- Accuracy preservation
- Size reduction up to 10x

### Easy to Use
- Drag & drop upload
- One-click compression
- No coding required
- Instant downloads

---

## 💻 Architecture

```
┌─────────────────────────────────────────────┐
│          Browser (Frontend)                  │
│   ┌───────────────────────────────────┐    │
│   │  compression_web_gui.html         │    │
│   │  - HTML/CSS/JavaScript            │    │
│   │  - Beautiful UI                   │    │
│   │  - Real-time updates              │    │
│   └───────────────┬───────────────────┘    │
└───────────────────┼─────────────────────────┘
                    │ HTTP REST API
                    │
┌───────────────────▼─────────────────────────┐
│        Server (Backend)                      │
│   ┌───────────────────────────────────┐    │
│   │  compression_server.py            │    │
│   │  - Flask API                      │    │
│   │  - File handling                  │    │
│   │  - Compression pipeline           │    │
│   └───────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

---

## 📡 API Endpoints

The GUI communicates with these endpoints:

### `POST /api/upload`
Upload model file
- **Input:** FormData with 'model' file
- **Output:** `{success: true, job_id: "...", filename: "...", size_mb: 123}`

### `POST /api/compress`
Start compression job
- **Input:** `{job_id: "...", settings: {...}}`
- **Output:** `{success: true, job_id: "..."}`

### `GET /api/status/<job_id>`
Check compression status
- **Output:** `{status: "processing", progress: 45, logs: [...], results: {...}}`

### `GET /api/download/<job_id>/<file_type>`
Download results
- **file_type:** "model" or "report"
- **Output:** File download

### `DELETE /api/cleanup/<job_id>`
Clean up temporary files
- **Output:** `{success: true}`

---

## 🛠️ Configuration

### Server Configuration

Edit `compression_server.py`:

```python
# File upload settings
UPLOAD_FOLDER = './uploads'
COMPRESSED_FOLDER = './compressed_output'
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB

# Allowed file types
ALLOWED_EXTENSIONS = {'pth', 'pt', 'ckpt', 'h5', 'pb'}

# Server settings
app.run(debug=True, host='0.0.0.0', port=5000)
```

### Frontend Configuration

Edit `compression_web_gui.html`:

```javascript
// API base URL
const API_BASE_URL = 'http://localhost:5000/api';

// Status polling interval (milliseconds)
const POLLING_INTERVAL = 1000;
```

---

## 🔧 Troubleshooting

### Problem: "Connection refused"
**Solution:** Make sure the server is running (`python compression_server.py`)

### Problem: "Upload failed"
**Solutions:**
- Check file size (max 500MB)
- Verify file format (.pth, .pt, etc.)
- Check server logs for errors

### Problem: "Compression failed"
**Solutions:**
- Ensure model file is valid PyTorch checkpoint
- Check server logs for detailed error
- Try with different model type setting

### Problem: CORS errors
**Solution:** Already handled with `flask-cors`, but if issues persist:
```python
# In compression_server.py, add:
from flask_cors import CORS
CORS(app, resources={r"/api/*": {"origins": "*"}})
```

### Problem: Slow compression
**Solutions:**
- Large models take longer
- Check CPU/GPU usage
- Consider using "Maximum Speed" target
- Reduce number of compression techniques

---

## 🚀 Deployment

### Production Deployment

#### Using Gunicorn (Linux/Mac)

```bash
pip install gunicorn

gunicorn -w 4 -b 0.0.0.0:5000 compression_server:app
```

#### Using Waitress (Windows)

```bash
pip install waitress

waitress-serve --port=5000 compression_server:app
```

#### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "compression_server:app"]
```

Build and run:

```bash
docker build -t model-compression .
docker run -p 5000:5000 model-compression
```

---

## 📊 Example Workflow

### Compressing a ResNet50 Model

1. **Upload:** Drag `resnet50_trained.pth` (90MB)
2. **Configure:**
   - Model Type: ResNet
   - Number of Classes: 1000
   - Enable: QAT ✓, Mixed-Precision ✓, Distillation ✓
   - Target: Balanced
3. **Start:** Click "Start Compression"
4. **Wait:** ~30 seconds (depends on model size)
5. **Results:**
   - Original: 90MB → Compressed: 12MB
   - Compression: 7.5×
   - Accuracy: 94.5% → 92.1% (-2.4%)
   - Speedup: 2.8×
6. **Download:** Get compressed model!

---

## 🎨 Customization

### Change Colors

Edit CSS variables in `compression_web_gui.html`:

```css
:root {
    --primary: #6366f1;      /* Main theme color */
    --secondary: #10b981;    /* Success color */
    --danger: #ef4444;       /* Error color */
    --warning: #f59e0b;      /* Warning color */
}
```

### Add New Model Types

In `compression_server.py`:

```python
def load_model(self, model_path, model_type, num_classes):
    if model_type == 'my_custom_model':
        # Add your custom model loading logic
        model = MyCustomModel(num_classes=num_classes)
        # ...
    # ...
```

Then add to HTML dropdown:

```html
<select id="modelType">
    <!-- ... existing options ... -->
    <option value="my_custom_model">My Custom Model</option>
</select>
```

---

## 📝 Requirements

### Python Packages

```txt
flask>=2.0.0
flask-cors>=3.0.0
torch>=2.0.0
torchvision>=0.15.0
psutil>=5.9.0
```

### System Requirements

- **RAM:** 8GB minimum, 16GB recommended
- **Storage:** 2GB free space
- **CPU:** Multi-core recommended
- **GPU:** Optional but speeds up compression

---

## 🆘 Getting Help

### Check Server Logs

The server prints detailed logs:

```
[INFO] File uploaded: model.pth (45.23 MB)
[INFO] Starting compression job: 1234567890_model.pth
[INFO] Loading model...
[INFO] Applying QAT...
[INFO] Compression complete!
```

### Enable Debug Mode

In `compression_server.py`:

```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

This provides detailed error messages in the browser!

---

## 🌟 Advanced Features

### Batch Processing

Want to compress multiple models? Coming soon!

### Custom Compression Profiles

Save your favorite settings for quick reuse!

### Visualization Generation

Automatic plots and charts of compression results!

### Cloud Storage Integration

Upload directly to S3, Google Cloud, or Azure!

---

## 📞 Support

**Issues?** Common solutions:
1. Check server is running
2. Verify file format
3. Review server logs
4. Try with smaller model first
5. Check Python version (3.8+)

**Still stuck?** Check the main README.md for more help!

---

## ✨ Tips & Tricks

1. **Start with defaults** - The recommended settings work great!
2. **Use balanced target** - Best compression/accuracy tradeoff
3. **Try one technique first** - Then combine for more compression
4. **Monitor logs** - Watch for warnings or errors
5. **Test compressed model** - Always validate accuracy on your data

---

## 🎉 Enjoy!

You now have a powerful, beautiful tool for compressing AI models!

**No coding required** - Just drag, drop, and download! 🚀

---

Made with ❤️ for the ML community
