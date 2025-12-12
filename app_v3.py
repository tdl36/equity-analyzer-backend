"""
TDL Equity Analyzer - Backend API Proxy
Bypasses CORS restrictions by proxying requests to Anthropic's API
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import json

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


@app.route('/api/analyze-multi', methods=['POST'])
def analyze_multi():
    """Analyze multiple PDF documents and generate investment thesis"""
    try:
        data = request.json
        api_key = data.get('apiKey')
        documents = data.get('documents', [])
        existing_analysis = data.get('existingAnalysis')
        
        if not api_key:
            return jsonify({'error': 'API key is required'}), 400
        
        if not documents:
            return jsonify({'error': 'No documents provided'}), 400
        
        # Filter enabled documents
        enabled_docs = [d for d in documents if d.get('enabled', True)]
        
        if not enabled_docs:
            return jsonify({'error': 'No enabled documents'}), 400
        
        # Build the content array for Claude
        content = []
        
        # Add each document as a PDF
        for doc in enabled_docs:
            doc_content = doc.get('content', '')
            doc_name = doc.get('name', 'document.pdf')
            
            # If content is base64 PDF
            if doc_content:
                # Remove data URL prefix if present
                if ',' in doc_content:
                    doc_content = doc_content.split(',')[1]
                
                content.append({
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": doc_content
                    }
                })
        
        # Add the analysis prompt
        analysis_prompt = """Analyze these broker research documents and create a comprehensive investment analysis.

Return a JSON object with this exact structure:
{
    "ticker": "STOCK_TICKER",
    "company": "Company Name",
    "thesis": {
        "summary": "2-3 sentence investment thesis summary",
        "pillars": [
            {"title": "Pillar 1 Title", "description": "Detailed explanation"},
            {"title": "Pillar 2 Title", "description": "Detailed explanation"},
            {"title": "Pillar 3 Title", "description": "Detailed explanation"}
        ]
    },
    "signposts": [
        {"signpost": "Key metric or event to watch", "target": "Target value or outcome", "timeframe": "When to expect"},
        {"signpost": "Another signpost", "target": "Target", "timeframe": "Timeframe"}
    ],
    "threats": [
        {"threat": "Risk factor 1", "mitigation": "How to monitor or mitigate"},
        {"threat": "Risk factor 2", "mitigation": "How to monitor or mitigate"}
    ]
}

Focus on:
1. Why own this stock? (Investment Thesis)
2. What are we looking for? (Signposts - specific KPIs, events, milestones)
3. Where can we be wrong? (Threats - bear case scenarios)

Return ONLY valid JSON, no markdown, no explanation."""

        if existing_analysis:
            analysis_prompt = f"""Update this existing analysis with new information from the documents.

Existing Analysis:
{json.dumps(existing_analysis, indent=2)}

Review the new documents and:
1. Update or confirm the investment thesis
2. Add any new signposts or update existing ones
3. Add any new threats or update existing ones
4. Note what has changed

Return the updated analysis as JSON with the same structure, plus a "changes" array describing what's new or different.

Return ONLY valid JSON, no markdown, no explanation."""

        content.append({
            "type": "text",
            "text": analysis_prompt
        })
        
        # Prepare request to Anthropic
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01'
        }
        
        payload = {
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': 8192,
            'messages': [
                {'role': 'user', 'content': content}
            ],
            'system': 'You are an expert equity research analyst. Analyze documents thoroughly and provide institutional-quality investment analysis. Always respond with valid JSON only.'
        }
        
        # Make request to Anthropic with longer timeout
        response = requests.post(ANTHROPIC_API_URL, headers=headers, json=payload, timeout=180)
        
        if response.status_code != 200:
            error_data = response.json()
            error_msg = error_data.get('error', {}).get('message', 'API request failed')
            return jsonify({'error': error_msg}), response.status_code
        
        result = response.json()
        assistant_content = result.get('content', [{}])[0].get('text', '')
        
        # Parse the JSON response
        try:
            # Clean up the response - remove markdown code blocks if present
            cleaned = assistant_content.strip()
            if cleaned.startswith('```'):
                cleaned = cleaned.split('\n', 1)[1]  # Remove first line
            if cleaned.endswith('```'):
                cleaned = cleaned.rsplit('\n', 1)[0]  # Remove last line
            if cleaned.startswith('json'):
                cleaned = cleaned[4:].strip()
            
            analysis = json.loads(cleaned)
            
            # Extract changes if present
            changes = analysis.pop('changes', [])
            
            return jsonify({
                'analysis': analysis,
                'changes': changes,
                'usage': result.get('usage', {})
            })
            
        except json.JSONDecodeError as e:
            # If JSON parsing fails, return raw content with error
            return jsonify({
                'error': f'Failed to parse analysis: {str(e)}',
                'raw_response': assistant_content
            }), 500
        
    except requests.Timeout:
        return jsonify({'error': 'Request timed out. Try with fewer or smaller documents.'}), 504
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


@app.route('/api/email-overview', methods=['POST'])
def send_overview_email():
    """Send Overview email via SMTP with HTML support"""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    try:
        data = request.json
        ticker = data.get('ticker', '')
        company_name = data.get('companyName', '')
        html_body = data.get('htmlBody', '')
        recipient = data.get('email')
        subject = data.get('customSubject', f'{ticker} - Stock Overview')
        smtp_config = data.get('smtpConfig', {})
        
        # Extract SMTP settings
        use_gmail = smtp_config.get('use_gmail', True)
        gmail_user = smtp_config.get('gmail_user', '')
        gmail_password = smtp_config.get('gmail_app_password', '')
        from_email = smtp_config.get('from_email', gmail_user)
        
        if not recipient:
            return jsonify({'error': 'Recipient email is required'}), 400
        
        if use_gmail and (not gmail_user or not gmail_password):
            return jsonify({'error': 'Gmail credentials required'}), 400
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['From'] = from_email
        msg['To'] = recipient
        msg['Subject'] = subject
        
        # Create plain text version
        plain_text = html_body.replace('<h1>', '\n').replace('</h1>', '\n' + '='*50 + '\n')
        plain_text = plain_text.replace('<h2>', '\n\n').replace('</h2>', '\n' + '-'*30 + '\n')
        plain_text = plain_text.replace('<p>', '').replace('</p>', '\n')
        plain_text = plain_text.replace('<br>', '\n').replace('<em>', '').replace('</em>', '')
        
        msg.attach(MIMEText(plain_text, 'plain'))
        msg.attach(MIMEText(html_body, 'html'))
        
        # Send via Gmail
        if use_gmail:
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(gmail_user, gmail_password)
                server.send_message(msg)
        
        return jsonify({'success': True, 'message': 'Overview email sent successfully'})
        
    except smtplib.SMTPAuthenticationError:
        return jsonify({'error': 'Gmail authentication failed. Check your email and app password.'}), 401
    except smtplib.SMTPException as e:
        return jsonify({'error': f'SMTP error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/email-analysis', methods=['POST'])
def send_analysis_email():
    """Send Analysis email via SMTP (fallback for overview)"""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    try:
        data = request.json
        analysis = data.get('analysis', {})
        recipient = data.get('email')
        subject = data.get('customSubject', f"{analysis.get('ticker', 'Stock')} - Analysis")
        smtp_config = data.get('smtpConfig', {})
        
        # Extract SMTP settings
        use_gmail = smtp_config.get('use_gmail', True)
        gmail_user = smtp_config.get('gmail_user', '')
        gmail_password = smtp_config.get('gmail_app_password', '')
        from_email = smtp_config.get('from_email', gmail_user)
        
        if not recipient:
            return jsonify({'error': 'Recipient email is required'}), 400
        
        if use_gmail and (not gmail_user or not gmail_password):
            return jsonify({'error': 'Gmail credentials required'}), 400
        
        # Build email body from analysis
        ticker = analysis.get('ticker', 'N/A')
        company = analysis.get('company', 'N/A')
        thesis = analysis.get('thesis', {})
        signposts = analysis.get('signposts', [])
        threats = analysis.get('threats', [])
        conclusion = analysis.get('conclusion', '')
        
        email_body = f"""EQUITY RESEARCH SUMMARY: {ticker}
{company}

{'='*60}
1. INVESTMENT THESIS (Why do we own the stock?)
{'='*60}
{thesis.get('summary', 'N/A')}

"""
        if thesis.get('pillars'):
            for pillar in thesis['pillars']:
                email_body += f"• {pillar.get('pillar', '')}: {pillar.get('detail', '')}\n"
        
        email_body += f"""
{'='*60}
2. SIGNPOSTS (What are we looking for?)
{'='*60}
"""
        for sp in signposts:
            email_body += f"• {sp.get('signpost', '')}: {sp.get('target', '')}\n"
        
        email_body += f"""
{'='*60}
3. THESIS THREATS (Where can we be wrong?)
{'='*60}
"""
        for threat in threats:
            email_body += f"• {threat.get('threat', '')}\n"
        
        if conclusion:
            email_body += f"""
{'='*60}
4. CONCLUSION
{'='*60}
{conclusion}
"""
        
        email_body += f"""
---
Generated by TDL Equity Analyzer
{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = recipient
        msg['Subject'] = subject
        msg.attach(MIMEText(email_body, 'plain'))
        
        # Send via Gmail
        if use_gmail:
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(gmail_user, gmail_password)
                server.send_message(msg)
        
        return jsonify({'success': True, 'message': 'Analysis email sent successfully'})
        
    except smtplib.SMTPAuthenticationError:
        return jsonify({'error': 'Gmail authentication failed. Check your email and app password.'}), 401
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
