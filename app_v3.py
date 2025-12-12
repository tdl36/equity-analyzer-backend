"""
TDL Equity Analyzer - Backend API Proxy
Bypasses CORS restrictions by proxying requests to Anthropic's API
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

@app.route('/api/chat', methods=['POST'])
def chat():
    """Proxy chat requests to Anthropic API"""
    try:
        data = request.json
        api_key = data.get('api_key')
        messages = data.get('messages', [])
        system = data.get('system', '')
        
        if not api_key:
            return jsonify({'error': 'API key is required'}), 400
        
        # Prepare request to Anthropic
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01'
        }
        
        payload = {
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': 4096,
            'messages': messages
        }
        
        if system:
            payload['system'] = system
        
        # Make request to Anthropic
        response = requests.post(ANTHROPIC_API_URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            error_data = response.json()
            return jsonify({'error': error_data.get('error', {}).get('message', 'API request failed')}), response.status_code
        
        result = response.json()
        assistant_content = result.get('content', [{}])[0].get('text', '')
        
        return jsonify({
            'response': assistant_content,
            'usage': result.get('usage', {})
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/parse', methods=['POST'])
def parse():
    """Use Claude to intelligently parse stock analysis into sections"""
    try:
        data = request.json
        api_key = data.get('api_key')
        content = data.get('content', '')
        
        if not api_key:
            return jsonify({'error': 'API key is required'}), 400
        
        # Prepare request to Anthropic
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01'
        }
        
        payload = {
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': 4096,
            'messages': [
                {'role': 'user', 'content': content}
            ],
            'system': 'You are a precise JSON extractor. Extract content into the exact JSON format requested. Return ONLY valid JSON with no markdown formatting, no code blocks, no explanation - just the raw JSON object.'
        }
        
        # Make request to Anthropic
        response = requests.post(ANTHROPIC_API_URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            error_data = response.json()
            return jsonify({'error': error_data.get('error', {}).get('message', 'API request failed')}), response.status_code
        
        result = response.json()
        assistant_content = result.get('content', [{}])[0].get('text', '')
        
        return jsonify({
            'response': assistant_content,
            'usage': result.get('usage', {})
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})


@app.route('/api/email', methods=['POST'])
def send_email():
    """Send email via SMTP"""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    try:
        data = request.json
        smtp_server = data.get('smtp_server')
        smtp_port = data.get('smtp_port', 587)
        email = data.get('email')
        password = data.get('password')
        recipient = data.get('recipient')
        subject = data.get('subject')
        body = data.get('body')
        
        # Validate required fields
        if not all([smtp_server, email, password, recipient, subject, body]):
            return jsonify({'error': 'Missing required email fields'}), 400
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = email
        msg['To'] = recipient
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(email, password)
            server.send_message(msg)
        
        return jsonify({'success': True, 'message': 'Email sent successfully'})
        
    except smtplib.SMTPAuthenticationError:
        return jsonify({'error': 'SMTP authentication failed. Check your email and password/app password.'}), 401
    except smtplib.SMTPException as e:
        return jsonify({'error': f'SMTP error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    print("=" * 50)
    print("TDL Equity Analyzer - Backend Server")
    print("=" * 50)
    print(f"Starting server on http://0.0.0.0:{port}")
    print("=" * 50)
    app.run(host='0.0.0.0', port=port, debug=False)
