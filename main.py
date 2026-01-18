# import pipeline and run here
import logging
import io
import sys
from flask import Flask, jsonify, render_template_string
from poc.src.classification.pipeline import run

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Store output from pipeline execution
pipeline_output = []


@app.route('/', methods=['GET'])
def home():
    """Home page with Hello World and output display"""
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML Classification Pipeline</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            h1 { color: #333; }
            button { padding: 10px 20px; font-size: 16px; cursor: pointer; background-color: #4CAF50; color: white; border: none; border-radius: 4px; }
            button:hover { background-color: #45a049; }
            .output-box { border: 1px solid #ccc; padding: 15px; margin-top: 20px; background-color: #f9f9f9; border-radius: 4px; min-height: 200px; max-height: 500px; overflow-y: auto; white-space: pre-wrap; word-wrap: break-word; }
            .status { margin-top: 10px; padding: 10px; border-radius: 4px; display: none; }
            .status.success { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .status.error { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Hello World! 👋</h1>
            <p>Welcome to the ML Classification Pipeline</p>
            
            <button onclick="runPipeline()">Run Pipeline</button>
            
            <div id="status" class="status"></div>
            
            <h2>Pipeline Output:</h2>
            <div class="output-box" id="output">Click "Run Pipeline" to see output...</div>
        </div>
        
        <script>
            async function runPipeline() {
                const outputBox = document.getElementById('output');
                const statusDiv = document.getElementById('status');
                
                outputBox.textContent = 'Running pipeline...';
                statusDiv.style.display = 'none';
                
                try {
                    const response = await fetch('/run-pipeline', { method: 'POST' });
                    const data = await response.json();
                    
                    if (response.ok) {
                        outputBox.textContent = data.output || 'Pipeline completed successfully!';
                        statusDiv.textContent = '✓ ' + data.message;
                        statusDiv.className = 'status success';
                    } else {
                        outputBox.textContent = 'Error: ' + data.message;
                        statusDiv.textContent = '✗ ' + data.message;
                        statusDiv.className = 'status error';
                    }
                } catch (error) {
                    outputBox.textContent = 'Error: ' + error.message;
                    statusDiv.textContent = '✗ ' + error.message;
                    statusDiv.className = 'status error';
                }
                statusDiv.style.display = 'block';
            }
        </script>
    </body>
    </html>
    '''
    return render_template_string(html_template)


@app.route('/run-pipeline', methods=['POST'])
def run_pipeline():
    """Run the classification pipeline and capture output"""
    try:
        logger.info("Starting pipeline execution...")
        
        # Capture stdout
        output_capture = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = output_capture
        
        run()
        
        # Restore stdout
        sys.stdout = old_stdout
        captured_output = output_capture.getvalue()
        
        logger.info("Pipeline execution completed successfully!")
        return jsonify({
            "status": "success",
            "message": "Pipeline executed successfully",
            "output": captured_output
        }), 200
    except Exception as e:
        sys.stdout = old_stdout
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e),
            "output": ""
        }), 500


if __name__ == "__main__":
    logger.info("Starting Flask API server...")
    app.run(host='0.0.0.0', port=5000, debug=False)
