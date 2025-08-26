#!/usr/bin/env python3
"""
Simple HTTP server for the MetaPromptBench
Serves static files and provides API endpoints for loading evaluation data
"""

import os
import json
import glob
from datetime import datetime
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import mimetypes

class EvaluationViewerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.dirname(os.path.abspath(__file__)), **kwargs)
    
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()
    
    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        # API endpoints
        if parsed_path.path == '/api/scan-results':
            self.handle_scan_results()
        elif parsed_path.path == '/api/load-file':
            self.handle_load_file(parse_qs(parsed_path.query))
        else:
            # Serve static files
            super().do_GET()
    
    def handle_scan_results(self):
        """Scan the generated directory for evaluation results"""
        try:
            results = self.scan_evaluation_results()
            self.send_json_response(results)
        except Exception as e:
            self.send_error_response(str(e))
    
    def handle_load_file(self, query_params):
        """Load a specific evaluation file"""
        try:
            file_path = query_params.get('path', [None])[0]
            if not file_path:
                raise ValueError("No file path provided")
            
            # Security check - ensure path is within the project directory
            abs_path = os.path.abspath(file_path)
            project_root = os.path.abspath(os.path.dirname(__file__))
            if not abs_path.startswith(project_root):
                raise ValueError("Invalid file path")
            
            with open(abs_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.send_json_response(data)
        except Exception as e:
            self.send_error_response(str(e))
    
    def scan_evaluation_results(self):
        """Scan for evaluation result files in the generated directory"""
        base_dir = os.path.join(os.path.dirname(__file__), 'generated')
        results = {
            'directories': [],
            'files': []
        }
        
        if not os.path.exists(base_dir):
            return results
        
        # Scan version directories
        for version_dir in sorted(os.listdir(base_dir), reverse=True):
            version_path = os.path.join(base_dir, version_dir)
            if not os.path.isdir(version_path) or version_dir == 'latest':
                continue
            
            eval_results_path = os.path.join(version_path, 'evaluation_results')
            if not os.path.exists(eval_results_path):
                continue
            
            version_info = {
                'version': version_dir,
                'path': version_path,
                'timestamp': self.extract_timestamp_from_version(version_dir),
                'benchmarks': []
            }
            
            # Scan benchmark directories
            for benchmark_dir in os.listdir(eval_results_path):
                benchmark_path = os.path.join(eval_results_path, benchmark_dir)
                if not os.path.isdir(benchmark_path):
                    continue
                
                benchmark_info = {
                    'name': benchmark_dir,
                    'path': benchmark_path,
                    'files': []
                }
                
                # Scan for JSON files
                json_files = glob.glob(os.path.join(benchmark_path, '*.json'))
                for json_file in json_files:
                    file_info = self.analyze_json_file(json_file)
                    if file_info:
                        benchmark_info['files'].append(file_info)
                
                if benchmark_info['files']:
                    version_info['benchmarks'].append(benchmark_info)
            
            if version_info['benchmarks']:
                results['directories'].append(version_info)
        
        return results
    
    def extract_timestamp_from_version(self, version_str):
        """Extract timestamp from version string like v2025_08_22_1747"""
        try:
            # Remove 'v' prefix and split
            parts = version_str[1:].split('_')
            if len(parts) >= 4:
                year, month, day, time = parts[:4]
                hour = time[:2]
                minute = time[2:] if len(time) > 2 else '00'
                
                dt = datetime(int(year), int(month), int(day), int(hour), int(minute))
                return dt.isoformat()
        except:
            pass
        return None
    
    def analyze_json_file(self, file_path):
        """Analyze a JSON file to extract metadata"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_info = {
                'name': os.path.basename(file_path),
                'path': file_path,
                'size': os.path.getsize(file_path),
                'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                'type': 'unknown'
            }
            
            # Determine file type and extract metadata
            if isinstance(data, dict):
                if 'metadata' in data and 'individual_results' in data:
                    file_info['type'] = 'individual_results'
                    file_info['metadata'] = data['metadata']
                elif 'signature' in data and 'instructions' in data.get('signature', {}):
                    file_info['type'] = 'prompt_config'
                    file_info['instructions'] = data['signature']['instructions']
                    file_info['field_count'] = len(data['signature'].get('fields', []))
            elif isinstance(data, list):
                file_info['type'] = 'list_data'
                file_info['count'] = len(data)
            
            return file_info
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None
    
    def send_json_response(self, data):
        """Send a JSON response"""
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(data, indent=2).encode('utf-8'))
        except BrokenPipeError:
            # Client disconnected, ignore the error
            pass
    
    def send_error_response(self, error_message):
        """Send an error response"""
        try:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            error_data = {'error': error_message}
            self.wfile.write(json.dumps(error_data).encode('utf-8'))
        except BrokenPipeError:
            # Client disconnected, ignore the error
            pass

def main():
    """Start the server"""
    port = 8000
    server_address = ('', port)
    
    print(f"Starting MetaPromptBench server...")
    print(f"Server running at http://localhost:{port}/")
    print(f"Open http://localhost:{port}/evaluation_viewer.html in your browser")
    print("Press Ctrl+C to stop the server")
    
    try:
        httpd = HTTPServer(server_address, EvaluationViewerHandler)
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")

if __name__ == '__main__':
    main()
