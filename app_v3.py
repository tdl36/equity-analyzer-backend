#!/usr/bin/env python3
"""
Equity Research Analyzer V3 - Portfolio Memory
Supports multiple documents with persistent storage across sessions
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import urllib.request
import urllib.error
from pathlib import Path
import os
import webbrowser
from datetime import datetime
import time
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

PORT = 8080

# Database configuration
# Set DB_PATH to a cloud-synced folder for multi-device access
# Examples:
#   Dropbox: 'C:/Users/YourName/Dropbox/EquityAnalyzer/analyses.db'
#   Google Drive: 'C:/Users/YourName/Google Drive/EquityAnalyzer/analyses.db'
#   OneDrive: 'C:/Users/YourName/OneDrive/EquityAnalyzer/analyses.db'
#   iCloud (Mac): '/Users/YourName/Library/Mobile Documents/com~apple~CloudDocs/EquityAnalyzer/analyses.db'
#   Local only: 'analyses.db' (current directory)

DB_PATH = os.environ.get('EQUITY_ANALYZER_DB', 'analyses.db')

# Create database directory if in a custom path
if '/' in DB_PATH or '\\' in DB_PATH:
    db_dir = os.path.dirname(DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
        print(f"📁 Created database directory: {db_dir}")

class AnalyzerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.db_path = DB_PATH
        self.init_database()
        super().__init__(*args, **kwargs)
    
    def init_database(self):
        """Initialize SQLite database for storing analyses"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create analyses table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                company_name TEXT,
                analysis_json TEXT NOT NULL,
                documents_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def do_GET(self):
        if self.path == '/':
            self.path = '/analyzer_v3.html'
        elif self.path == '/api/analyses':
            self.list_analyses()
            return
        elif self.path.startswith('/api/analysis/'):
            ticker = self.path.split('/')[-1]
            self.get_analysis(ticker)
            return
        return SimpleHTTPRequestHandler.do_GET(self)
    
    def do_POST(self):
        if self.path == '/api/analyze-multi':
            self.handle_multi_analyze()
        elif self.path == '/api/save-analysis':
            self.save_analysis()
        elif self.path == '/api/delete-analysis':
            self.delete_analysis()
        elif self.path == '/api/email-analysis':
            self.email_analysis()
        else:
            self.send_error(404)
    
    def list_analyses(self):
        """List all saved analyses"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT ticker, company_name, created_at, updated_at
                FROM analyses
                ORDER BY updated_at DESC
            ''')
            
            results = cursor.fetchall()
            conn.close()
            
            analyses = [
                {
                    'ticker': row[0],
                    'company': row[1],
                    'created': row[2],
                    'updated': row[3]
                }
                for row in results
            ]
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(analyses).encode('utf-8'))
            
        except Exception as e:
            self.send_error_response(500, str(e))
    
    def get_analysis(self, ticker):
        """Get a specific analysis by ticker"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT analysis_json, documents_json
                FROM analyses
                WHERE ticker = ?
            ''', (ticker.upper(),))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                response = {
                    'analysis': json.loads(result[0]),
                    'documents': json.loads(result[1]) if result[1] else []
                }
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
            else:
                self.send_error_response(404, f"No analysis found for {ticker}")
                
        except Exception as e:
            self.send_error_response(500, str(e))
    
    def save_analysis(self):
        """Save or update an analysis"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            ticker = request_data.get('ticker', '').upper()
            company_name = request_data.get('companyName', '')
            analysis = request_data.get('analysis', {})
            documents = request_data.get('documents', [])
            
            if not ticker:
                self.send_error_response(400, "Ticker required")
                return
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if exists
            cursor.execute('SELECT id FROM analyses WHERE ticker = ?', (ticker,))
            exists = cursor.fetchone()
            
            if exists:
                # Update
                cursor.execute('''
                    UPDATE analyses
                    SET company_name = ?, analysis_json = ?, documents_json = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE ticker = ?
                ''', (company_name, json.dumps(analysis), json.dumps(documents), ticker))
                action = 'updated'
            else:
                # Insert
                cursor.execute('''
                    INSERT INTO analyses (ticker, company_name, analysis_json, documents_json)
                    VALUES (?, ?, ?, ?)
                ''', (ticker, company_name, json.dumps(analysis), json.dumps(documents)))
                action = 'saved'
            
            conn.commit()
            conn.close()
            
            print(f"✅ Analysis {action} for {ticker}")
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'success': True, 'action': action, 'ticker': ticker}).encode('utf-8'))
            
        except Exception as e:
            self.send_error_response(500, str(e))
    
    def delete_analysis(self):
        """Delete an analysis"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            ticker = request_data.get('ticker', '').upper()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM analyses WHERE ticker = ?', (ticker,))
            conn.commit()
            conn.close()
            
            print(f"🗑️  Analysis deleted for {ticker}")
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'success': True, 'ticker': ticker}).encode('utf-8'))
            
        except Exception as e:
            self.send_error_response(500, str(e))
    
    def format_email_report(self, analysis):
        """Format analysis into professional email report with clean formatting"""
        ticker = analysis.get('ticker', 'N/A')
        company = analysis.get('company', 'Company')
        
        # Build email body
        report = []
        report.append(f"EQUITY RESEARCH SUMMARY: {ticker}")
        report.append(f"{company}")
        report.append("")
        report.append("")
        
        # SECTION 1: INVESTMENT THESIS
        report.append("1. INVESTMENT THESIS (Why do we own the stock?)")
        report.append("")
        
        thesis_pillars = analysis.get('thesis', {}).get('pillars', [])
        thesis_summary = analysis.get('thesis', {}).get('summary', '')
        
        # Create 4-5 bullet points from thesis
        bullets = []
        if thesis_summary:
            bullets.append(f"• {thesis_summary}")
        
        for pillar in thesis_pillars[:4]:  # Max 4 pillars to keep 4-5 bullets total
            title = pillar.get('title', '')
            desc = pillar.get('description', '')
            bullets.append(f"• {title}: {desc}")
        
        # Add bullets with spacing between them
        for i, bullet in enumerate(bullets[:5]):
            report.append(bullet)
            if i < len(bullets[:5]) - 1:  # Add blank line between bullets, but not after last one
                report.append("")
        
        report.append("")
        report.append("")
        
        # SECTION 2: SIGNPOSTS
        report.append("2. SIGNPOSTS (What are we looking for?)")
        report.append("")
        
        signposts = analysis.get('signposts', [])
        signpost_bullets = []
        for sp in signposts[:5]:  # Max 5 signposts
            metric = sp.get('metric', '')
            target = sp.get('target', '')
            timeframe = sp.get('timeframe', '')
            signpost_bullets.append(f"• {metric}: {target} ({timeframe})")
        
        # Pad if less than 4
        if len(signposts) < 4:
            signpost_bullets.append("• Monitor: Additional forward-looking KPIs to be identified as thesis develops")
        
        # Add signposts with spacing between them
        for i, bullet in enumerate(signpost_bullets):
            report.append(bullet)
            if i < len(signpost_bullets) - 1:  # Add blank line between bullets
                report.append("")
        
        report.append("")
        report.append("")
        
        # SECTION 3: THESIS THREATS
        report.append("3. THESIS THREATS (Where can we be wrong?)")
        report.append("")
        
        threats = analysis.get('threats', [])
        threat_bullets = []
        for threat in threats[:5]:  # Max 5 threats
            threat_desc = threat.get('threat', '')
            likelihood = threat.get('likelihood', '')
            impact = threat.get('impact', '')
            trigger = threat.get('triggerPoints', '')
            
            threat_text = f"• {threat_desc} (Likelihood: {likelihood}, Impact: {impact})"
            if trigger:
                threat_text += f"\n  → {trigger}"
            threat_bullets.append(threat_text)
        
        # Pad if less than 4
        if len(threats) < 4:
            threat_bullets.append("• Structural Risk: Specific company risks to be monitored as analysis deepens")
        
        # Add threats with spacing between them
        for i, bullet in enumerate(threat_bullets):
            report.append(bullet)
            if i < len(threat_bullets) - 1:  # Add blank line between bullets
                report.append("")
        
        return "\n".join(report)
    
    def email_analysis(self):
        """Send analysis via email"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            analysis = request_data.get('analysis')
            recipient_email = request_data.get('email')
            custom_subject = request_data.get('customSubject')
            smtp_config = request_data.get('smtpConfig', {})
            
            if not analysis or not recipient_email:
                self.send_error_response(400, "Missing analysis or email")
                return
            
            # Format email body
            email_body = self.format_email_report(analysis)
            
            ticker = analysis.get('ticker', 'N/A')
            company = analysis.get('company', 'Company')
            
            # Create email
            msg = MIMEMultipart()
            msg['From'] = smtp_config.get('from_email', recipient_email)
            msg['To'] = recipient_email
            # Use custom subject if provided, otherwise use default
            msg['Subject'] = custom_subject if custom_subject else f"Investment Summary: {ticker} - {company}"
            
            msg.attach(MIMEText(email_body, 'plain'))
            
            # Send via SMTP
            try:
                if smtp_config.get('use_gmail', False):
                    # Gmail SMTP
                    smtp_server = 'smtp.gmail.com'
                    smtp_port = 587
                    smtp_user = smtp_config.get('gmail_user')
                    smtp_password = smtp_config.get('gmail_app_password')
                    
                    if not smtp_user or not smtp_password:
                        raise Exception("Gmail credentials not provided")
                    
                    server = smtplib.SMTP(smtp_server, smtp_port)
                    server.starttls()
                    server.login(smtp_user, smtp_password)
                    server.send_message(msg)
                    server.quit()
                else:
                    # Local SMTP (for testing)
                    server = smtplib.SMTP('localhost', 1025)
                    server.send_message(msg)
                    server.quit()
                
                print(f"📧 Email sent to {recipient_email} for {ticker}")
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'success': True,
                    'message': f'Email sent to {recipient_email}'
                }).encode('utf-8'))
                
            except Exception as email_error:
                raise Exception(f"Email sending failed: {str(email_error)}")
            
        except Exception as e:
            print(f"❌ Email error: {str(e)}")
            self.send_error_response(500, str(e))
    
    def handle_multi_analyze(self):
        """Multi-document sequential refinement analysis"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            api_key = request_data.get('apiKey')
            documents = request_data.get('documents', [])
            existing_analysis = request_data.get('existingAnalysis')  # Check for existing analysis
            
            # Filter out excluded documents from existing analysis
            if existing_analysis and existing_analysis.get('documentHistory'):
                active_docs = [d for d in existing_analysis['documentHistory'] if d.get('includeInContext', True)]
                if len(active_docs) < len(existing_analysis['documentHistory']):
                    excluded_count = len(existing_analysis['documentHistory']) - len(active_docs)
                    print(f"   ℹ️  Excluding {excluded_count} stale document(s) from context")
                    existing_analysis['documentHistory'] = active_docs
            
            if not api_key:
                self.send_error_response(400, "Missing API key")
                return
            
            if not documents:
                self.send_error_response(400, "No documents provided")
                return
            
            # Filter to only enabled documents
            enabled_docs = [d for d in documents if d.get('enabled', True)]
            
            if not enabled_docs:
                self.send_error_response(400, "No enabled documents")
                return
            
            print(f"\n{'='*60}")
            if existing_analysis:
                print(f"🔄 Updating existing analysis for {existing_analysis.get('company', 'Unknown')}")
                print(f"   Adding {len(enabled_docs)} new document(s) to existing thesis")
            else:
                print(f"🔄 Processing {len(enabled_docs)} documents sequentially...")
            print(f"{'='*60}\n")
            
            # Small initial delay to ensure API is ready
            if len(enabled_docs) > 1 or existing_analysis:
                print("⏳ Initializing... (2s)")
                time.sleep(2)
            
            # Process documents sequentially
            cumulative_analysis = existing_analysis  # Start with existing if provided
            all_changes = []
            
            for idx, doc in enumerate(enabled_docs, 1):
                doc_metadata = doc.get('metadata', {})
                doc_weight = doc.get('weight', 1.0)
                
                print(f"📄 Document {idx}/{len(enabled_docs)}: {doc_metadata.get('filename', 'Unknown')}")
                print(f"   Weight: {doc_weight}x | Type: {doc_metadata.get('type', 'Unknown')}")
                
                # Add delay between requests to avoid rate limiting (except for first doc)
                if idx > 1 or existing_analysis:
                    delay = 5  # 5 second delay between requests
                    print(f"   ⏳ Waiting {delay}s to avoid rate limits...")
                    time.sleep(delay)
                
                if cumulative_analysis is None:
                    # First document and no existing analysis - create initial analysis
                    print(f"   → Creating initial thesis...")
                    cumulative_analysis = self.analyze_initial_document(
                        api_key, 
                        doc['pdfData'],
                        doc_metadata,
                        doc_weight
                    )
                else:
                    # Have existing analysis OR subsequent documents - refine
                    print(f"   → Refining existing thesis...")
                    result = self.refine_with_new_document(
                        api_key,
                        cumulative_analysis,
                        doc['pdfData'],
                        doc_metadata,
                        doc_weight
                    )
                    cumulative_analysis = result['analysis']
                    all_changes.extend(result.get('changes', []))
                
                print(f"   ✓ Processed\n")
            
            # Add metadata
            response = {
                'analysis': cumulative_analysis,
                'changes': all_changes,
                'documentsProcessed': len(enabled_docs),
                'processedAt': datetime.now().isoformat()
            }
            
            if existing_analysis:
                print(f"✅ Analysis updated with {len(enabled_docs)} new document(s)!\n")
            else:
                print(f"✅ Multi-document analysis complete!\n")
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            print(f"❌ Error: {str(e)}\n")
            self.send_error_response(500, str(e))
    
    # [Include all the analyze_initial_document, refine_with_new_document, and API call methods from V2]
    # [Keeping this concise - the methods are identical to V2]
    
    def analyze_initial_document(self, api_key, pdf_data, metadata, weight):
        """Analyze first document to create base thesis"""
        prompt = f"""You are an institutional equity research analyst. This is the FIRST document in a series.

Document Type: {metadata.get('type', 'Unknown')}
Source: {metadata.get('analyst', 'Unknown')}
Date: {metadata.get('date', 'Unknown')}
Weight: {weight}x

Analyze and create a FOUNDATIONAL investment thesis.

Format as JSON with this structure (be sure to include the ticker field):
{{
  "company": "Company Name",
  "ticker": "TICKER",
  "lastUpdated": "{datetime.now().strftime('%Y-%m-%d')}",
  "documentHistory": [{{
    "filename": "{metadata.get('filename', 'Unknown')}",
    "type": "{metadata.get('type', 'broker_report')}",
    "analyst": "{metadata.get('analyst', 'Unknown')}",
    "date": "{metadata.get('date', 'Unknown')}",
    "weight": {weight}
  }}],
  "thesis": {{
    "summary": "2-3 sentence summary",
    "pillars": [{{"title": "Title", "description": "Description", "confidence": "High/Medium/Low", "sources": ["{metadata.get('filename')}"], "weight": {weight}}}],
    "valuation": "Valuation perspective",
    "returnProfile": "Expected returns"
  }},
  "signposts": [{{
    "category": "Category",
    "metric": "Metric",
    "target": "Target",
    "timeframe": "Timeframe",
    "significance": "Why it matters",
    "sources": ["{metadata.get('filename')}"],
    "confidence": "High/Medium/Low"
  }}],
  "threats": [{{
    "category": "Category",
    "threat": "Threat",
    "likelihood": "High/Medium/Low",
    "impact": "High/Medium/Low",
    "triggerPoints": "Triggers",
    "mitigation": "Mitigation",
    "sources": ["{metadata.get('filename')}"],
    "weight": {weight}
  }}]
}}

Provide 3-5 pillars, 5-8 signposts, 4-6 threats."""
        return self.call_anthropic_single(api_key, pdf_data, prompt)
    
    def refine_with_new_document(self, api_key, existing_analysis, new_pdf_data, metadata, weight):
        """Refine existing analysis with new document"""
        prompt = f"""You previously analyzed {existing_analysis.get('company')}.

EXISTING THESIS:
{json.dumps(existing_analysis, indent=2)}

NEW DOCUMENT:
Type: {metadata.get('type')}
Source: {metadata.get('analyst')}
Date: {metadata.get('date')}
Weight: {weight}x

Update the thesis incorporating this document. Weight {weight}x means this is {weight}x as important.

Return JSON with:
{{
  "analysis": {{...updated full analysis...}},
  "changes": ["Change 1", "Change 2", ...]
}}"""
        return self.call_anthropic_refinement(api_key, new_pdf_data, prompt)
    
    def call_anthropic_with_retry(self, api_key, pdf_data, prompt, max_retries=3, max_tokens=4000):
        """Call Anthropic API with retry logic"""
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(
                    'https://api.anthropic.com/v1/messages',
                    data=json.dumps({
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": max_tokens,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_data}},
                                {"type": "text", "text": prompt}
                            ]
                        }]
                    }).encode('utf-8'),
                    headers={
                        'Content-Type': 'application/json',
                        'x-api-key': api_key,
                        'anthropic-version': '2023-06-01'
                    }
                )
                
                with urllib.request.urlopen(req, timeout=120) as response:
                    result = json.loads(response.read().decode('utf-8'))
                    text = result['content'][0]['text']
                    
                    if '```json' in text:
                        text = text.split('```json')[1].split('```')[0].strip()
                    elif '```' in text:
                        text = text.split('```')[1].split('```')[0].strip()
                    
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        if attempt < max_retries - 1:
                            print(f"   🔄 Retrying due to malformed response...")
                            time.sleep(2)
                            continue
                        raise
                    
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < max_retries - 1:
                    wait = (2 ** attempt) * 3
                    print(f"   ⚠️  Rate limit. Retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                raise
        
        raise Exception("Failed after all retries")
    
    def call_anthropic_single(self, api_key, pdf_data, prompt):
        return self.call_anthropic_with_retry(api_key, pdf_data, prompt, max_tokens=4000)
    
    def call_anthropic_refinement(self, api_key, pdf_data, prompt):
        return self.call_anthropic_with_retry(api_key, pdf_data, prompt, max_tokens=5000)
    
    def send_error_response(self, code, message):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'error': {'message': message}}).encode('utf-8'))
    
    def do_OPTIONS(self):
        self.send_response(200)
        # CORS headers are automatically added by our custom end_headers() method
        self.end_headers()
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, DELETE')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

def main():
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Resolve absolute path for display
    db_display_path = os.path.abspath(DB_PATH)
    is_synced = any(cloud in db_display_path.lower() for cloud in ['dropbox', 'google drive', 'onedrive', 'icloud', 'documents/com~apple~'])
    
    print("=" * 70)
    print("🚀 Equity Research Analyzer V3 - Portfolio Memory")
    print("=" * 70)
    print(f"\n✅ Server running on http://localhost:{PORT}")
    print(f"📂 Database: {db_display_path}")
    if is_synced:
        print("   ☁️  CLOUD SYNCED - Available across devices!")
    print("\n✨ NEW IN V3:")
    print("   • Save analyses for multiple stocks")
    print("   • Load previous analyses instantly")
    print("   • Portfolio view of all stocks")
    print("   • Update existing analyses")
    print("   • Delete old analyses")
    if is_synced:
        print("   • Cloud sync enabled - access from any device!")
    print("\n🌐 Opening browser...")
    print("\n⚠️  To stop: Press Ctrl+C")
    print("=" * 70)
    
    webbrowser.open(f'http://localhost:{PORT}')
    
    httpd = HTTPServer(('', PORT), AnalyzerHandler)
    
    try:
        print(f"\n✨ Ready! Access at: http://localhost:{PORT}\n")
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!")
        httpd.shutdown()

if __name__ == "__main__":
    main()
