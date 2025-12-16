"""
TDL Equity Analyzer - Backend API with PostgreSQL
Cross-device sync for portfolio analyses and overviews
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import json
import base64
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
import anthropic

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

# ============================================
# DATABASE CONNECTION
# ============================================

def get_db_connection():
    """Get PostgreSQL database connection"""
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise Exception('DATABASE_URL environment variable not set')
    
    # Render uses postgres:// but psycopg2 needs postgresql://
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    
    conn = psycopg2.connect(database_url, cursor_factory=RealDictCursor)
    return conn

def init_db():
    """Initialize database tables"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Portfolio Analyses table
        cur.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_analyses (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(20) UNIQUE NOT NULL,
                company VARCHAR(255),
                analysis JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Stock Overviews table
        cur.execute('''
            CREATE TABLE IF NOT EXISTS stock_overviews (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(20) UNIQUE NOT NULL,
                company_name VARCHAR(255),
                company_overview TEXT,
                business_model TEXT,
                business_mix TEXT,
                opportunities TEXT,
                risks TEXT,
                conclusion TEXT,
                raw_content TEXT,
                history JSONB DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Add business_mix column if it doesn't exist (migration)
        cur.execute('''
            DO $$ 
            BEGIN 
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                              WHERE table_name='stock_overviews' AND column_name='business_mix') THEN
                    ALTER TABLE stock_overviews ADD COLUMN business_mix TEXT;
                END IF;
            END $$;
        ''')
        
        # Chat Histories table
        cur.execute('''
            CREATE TABLE IF NOT EXISTS chat_histories (
                id VARCHAR(100) PRIMARY KEY,
                title VARCHAR(255),
                messages JSONB DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Meeting Summaries table
        cur.execute('''
            CREATE TABLE IF NOT EXISTS meeting_summaries (
                id VARCHAR(100) PRIMARY KEY,
                title VARCHAR(255),
                raw_notes TEXT,
                summary TEXT,
                questions TEXT,
                topic VARCHAR(100) DEFAULT 'General',
                topic_type VARCHAR(20) DEFAULT 'other',
                source_type VARCHAR(20) DEFAULT 'paste',
                source_files JSONB DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Add columns if they don't exist (migration)
        cur.execute('''
            DO $$ 
            BEGIN 
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                              WHERE table_name='meeting_summaries' AND column_name='topic') THEN
                    ALTER TABLE meeting_summaries ADD COLUMN topic VARCHAR(100) DEFAULT 'General';
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                              WHERE table_name='meeting_summaries' AND column_name='topic_type') THEN
                    ALTER TABLE meeting_summaries ADD COLUMN topic_type VARCHAR(20) DEFAULT 'other';
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                              WHERE table_name='meeting_summaries' AND column_name='source_type') THEN
                    ALTER TABLE meeting_summaries ADD COLUMN source_type VARCHAR(20) DEFAULT 'paste';
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                              WHERE table_name='meeting_summaries' AND column_name='source_files') THEN
                    ALTER TABLE meeting_summaries ADD COLUMN source_files JSONB DEFAULT '[]';
                END IF;
            END $$;
        ''')
        
        # Document Files table - stores actual document content for re-analysis
        cur.execute('''
            CREATE TABLE IF NOT EXISTS document_files (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(20) NOT NULL,
                filename VARCHAR(255) NOT NULL,
                file_data TEXT NOT NULL,
                file_type VARCHAR(50),
                mime_type VARCHAR(100),
                metadata JSONB DEFAULT '{}',
                file_size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, filename)
            )
        ''')
        
        # Research Categories (tickers + topics)
        cur.execute('''
            CREATE TABLE IF NOT EXISTS research_categories (
                id VARCHAR(100) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                type VARCHAR(20) DEFAULT 'ticker',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Research Documents (files/text under categories)
        cur.execute('''
            CREATE TABLE IF NOT EXISTS research_documents (
                id VARCHAR(100) PRIMARY KEY,
                category_id VARCHAR(100) REFERENCES research_categories(id) ON DELETE CASCADE,
                name VARCHAR(255) NOT NULL,
                content TEXT,
                file_names JSONB DEFAULT '[]',
                smart_name VARCHAR(500),
                original_filename VARCHAR(500),
                published_date VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Migration: Add new columns if they don't exist
        cur.execute('''
            DO $$ 
            BEGIN 
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                              WHERE table_name='research_documents' AND column_name='smart_name') THEN
                    ALTER TABLE research_documents ADD COLUMN smart_name VARCHAR(500);
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                              WHERE table_name='research_documents' AND column_name='original_filename') THEN
                    ALTER TABLE research_documents ADD COLUMN original_filename VARCHAR(500);
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                              WHERE table_name='research_documents' AND column_name='published_date') THEN
                    ALTER TABLE research_documents ADD COLUMN published_date VARCHAR(100);
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                              WHERE table_name='research_documents' AND column_name='has_stored_files') THEN
                    ALTER TABLE research_documents ADD COLUMN has_stored_files BOOLEAN DEFAULT FALSE;
                END IF;
            END $$;
        ''')
        
        # Research Document Files (stored PDFs/files for re-analysis)
        cur.execute('''
            CREATE TABLE IF NOT EXISTS research_document_files (
                id SERIAL PRIMARY KEY,
                document_id VARCHAR(100) REFERENCES research_documents(id) ON DELETE CASCADE,
                filename VARCHAR(500) NOT NULL,
                file_type VARCHAR(100),
                file_data TEXT,
                file_size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Research Analyses (framework results under documents)
        cur.execute('''
            CREATE TABLE IF NOT EXISTS research_analyses (
                id VARCHAR(100) PRIMARY KEY,
                document_id VARCHAR(100) REFERENCES research_documents(id) ON DELETE CASCADE,
                prompt_id VARCHAR(100),
                prompt_name VARCHAR(255),
                prompt_icon VARCHAR(10),
                result TEXT,
                usage JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create index for faster lookups
        cur.execute('''
            CREATE INDEX IF NOT EXISTS idx_document_files_ticker 
            ON document_files(ticker)
        ''')
        
        conn.commit()
        cur.close()
        conn.close()
        print("Database tables initialized")
    except Exception as e:
        print(f"Database init error (may be normal on first run): {e}")

# Initialize database on startup
try:
    init_db()
except:
    pass  # Will init when DATABASE_URL is available


# ============================================
# PORTFOLIO ANALYSES ENDPOINTS
# ============================================

@app.route('/api/analyses', methods=['GET'])
def get_analyses():
    """Get all saved portfolio analyses"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
            SELECT ticker, company, analysis, updated_at 
            FROM portfolio_analyses 
            ORDER BY ticker ASC
        ''')
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        result = []
        for row in rows:
            result.append({
                'ticker': row['ticker'],
                'company': row['company'],
                'analysis': row['analysis'],
                'updated': row['updated_at'].isoformat() if row['updated_at'] else None
            })
        
        return jsonify(result)
    except Exception as e:
        print(f"Error getting analyses: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analysis/<ticker>', methods=['GET'])
def get_analysis(ticker):
    """Get a specific portfolio analysis by ticker"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
            SELECT ticker, company, analysis, updated_at 
            FROM portfolio_analyses 
            WHERE ticker = %s
        ''', (ticker.upper(),))
        row = cur.fetchone()
        cur.close()
        conn.close()
        
        if not row:
            return jsonify({'error': 'Analysis not found'}), 404
        
        return jsonify({
            'ticker': row['ticker'],
            'company': row['company'],
            'analysis': row['analysis'],
            'updated': row['updated_at'].isoformat() if row['updated_at'] else None
        })
    except Exception as e:
        print(f"Error getting analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/save-analysis', methods=['POST'])
def save_analysis():
    """Save or update a portfolio analysis"""
    try:
        data = request.json
        ticker = data.get('ticker', '').upper()
        company = data.get('companyName', data.get('company', ''))
        analysis = data.get('analysis', {})
        
        if not ticker:
            return jsonify({'error': 'Ticker is required'}), 400
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Upsert - insert or update
        cur.execute('''
            INSERT INTO portfolio_analyses (ticker, company, analysis, updated_at)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (ticker) 
            DO UPDATE SET 
                company = EXCLUDED.company,
                analysis = EXCLUDED.analysis,
                updated_at = EXCLUDED.updated_at
            RETURNING ticker
        ''', (ticker, company, json.dumps(analysis), datetime.utcnow()))
        
        result = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({'success': True, 'ticker': result['ticker']})
    except Exception as e:
        print(f"Error saving analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete-analysis', methods=['POST'])
def delete_analysis():
    """Delete a portfolio analysis"""
    try:
        data = request.json
        ticker = data.get('ticker', '').upper()
        
        if not ticker:
            return jsonify({'error': 'Ticker is required'}), 400
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('DELETE FROM portfolio_analyses WHERE ticker = %s', (ticker,))
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error deleting analysis: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================
# STOCK OVERVIEWS ENDPOINTS
# ============================================

@app.route('/api/overviews', methods=['GET'])
def get_overviews():
    """Get all saved stock overviews"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
            SELECT ticker, company_name, company_overview, business_model, business_mix,
                   opportunities, risks, conclusion, raw_content, history, updated_at 
            FROM stock_overviews 
            ORDER BY ticker ASC
        ''')
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        result = []
        for row in rows:
            result.append({
                'ticker': row['ticker'],
                'companyName': row['company_name'],
                'companyOverview': row['company_overview'],
                'businessModel': row['business_model'],
                'businessMix': row.get('business_mix', ''),
                'opportunities': row['opportunities'],
                'risks': row['risks'],
                'conclusion': row['conclusion'],
                'rawContent': row['raw_content'],
                'history': row['history'] or [],
                'updatedAt': row['updated_at'].isoformat() if row['updated_at'] else None
            })
        
        return jsonify(result)
    except Exception as e:
        print(f"Error getting overviews: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/save-overview', methods=['POST'])
def save_overview():
    """Save or update a stock overview"""
    try:
        data = request.json
        ticker = data.get('ticker', '').upper()
        
        if not ticker:
            return jsonify({'error': 'Ticker is required'}), 400
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Upsert
        cur.execute('''
            INSERT INTO stock_overviews (
                ticker, company_name, company_overview, business_model, business_mix,
                opportunities, risks, conclusion, raw_content, history, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker) 
            DO UPDATE SET 
                company_name = EXCLUDED.company_name,
                company_overview = EXCLUDED.company_overview,
                business_model = EXCLUDED.business_model,
                business_mix = EXCLUDED.business_mix,
                opportunities = EXCLUDED.opportunities,
                risks = EXCLUDED.risks,
                conclusion = EXCLUDED.conclusion,
                raw_content = EXCLUDED.raw_content,
                history = EXCLUDED.history,
                updated_at = EXCLUDED.updated_at
            RETURNING ticker
        ''', (
            ticker,
            data.get('companyName', ''),
            data.get('companyOverview', ''),
            data.get('businessModel', ''),
            data.get('businessMix', ''),
            data.get('opportunities', ''),
            data.get('risks', ''),
            data.get('conclusion', ''),
            data.get('rawContent', ''),
            json.dumps(data.get('history', [])),
            datetime.utcnow()
        ))
        
        result = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({'success': True, 'ticker': result['ticker']})
    except Exception as e:
        print(f"Error saving overview: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete-overview', methods=['POST'])
def delete_overview():
    """Delete a stock overview"""
    try:
        data = request.json
        ticker = data.get('ticker', '').upper()
        
        if not ticker:
            return jsonify({'error': 'Ticker is required'}), 400
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('DELETE FROM stock_overviews WHERE ticker = %s', (ticker,))
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error deleting overview: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================
# CHAT HISTORY ENDPOINTS
# ============================================

@app.route('/api/chats', methods=['GET'])
def get_chats():
    """Get all chat histories"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
            SELECT id, title, messages, updated_at 
            FROM chat_histories 
            ORDER BY updated_at DESC
        ''')
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        result = []
        for row in rows:
            result.append({
                'id': row['id'],
                'title': row['title'],
                'messages': row['messages'] or [],
                'updatedAt': row['updated_at'].isoformat() if row['updated_at'] else None
            })
        
        return jsonify(result)
    except Exception as e:
        print(f"Error getting chats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/save-chat', methods=['POST'])
def save_chat():
    """Save or update a chat history"""
    try:
        data = request.json
        chat_id = data.get('id', '')
        
        if not chat_id:
            return jsonify({'error': 'Chat ID is required'}), 400
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute('''
            INSERT INTO chat_histories (id, title, messages, updated_at)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (id) 
            DO UPDATE SET 
                title = EXCLUDED.title,
                messages = EXCLUDED.messages,
                updated_at = EXCLUDED.updated_at
            RETURNING id
        ''', (
            chat_id,
            data.get('title', 'New Chat'),
            json.dumps(data.get('messages', [])),
            datetime.utcnow()
        ))
        
        result = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({'success': True, 'id': result['id']})
    except Exception as e:
        print(f"Error saving chat: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete-chat', methods=['POST'])
def delete_chat():
    """Delete a chat history"""
    try:
        data = request.json
        chat_id = data.get('id', '')
        
        if not chat_id:
            return jsonify({'error': 'Chat ID is required'}), 400
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('DELETE FROM chat_histories WHERE id = %s', (chat_id,))
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error deleting chat: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================
# MEETING SUMMARY ENDPOINTS
# ============================================

@app.route('/api/summaries', methods=['GET'])
def get_summaries():
    """Get all meeting summaries"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
            SELECT id, title, raw_notes, summary, questions, topic, topic_type, source_type, source_files, created_at 
            FROM meeting_summaries 
            ORDER BY created_at DESC
        ''')
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        result = []
        for row in rows:
            result.append({
                'id': row['id'],
                'title': row['title'],
                'rawNotes': row['raw_notes'],
                'summary': row['summary'],
                'questions': row['questions'],
                'topic': row.get('topic') or 'General',
                'topicType': row.get('topic_type') or 'other',
                'sourceType': row.get('source_type') or 'paste',
                'sourceFiles': row.get('source_files') or [],
                'createdAt': row['created_at'].isoformat() if row['created_at'] else None
            })
        
        return jsonify(result)
    except Exception as e:
        print(f"Error getting summaries: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/save-summary', methods=['POST'])
def save_summary():
    """Save or update a meeting summary"""
    try:
        data = request.json
        summary_id = data.get('id', '')
        
        if not summary_id:
            return jsonify({'error': 'Summary ID is required'}), 400
        
        # Convert sourceFiles list to JSON
        source_files = data.get('sourceFiles', [])
        if isinstance(source_files, list):
            source_files = json.dumps(source_files)
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute('''
            INSERT INTO meeting_summaries (id, title, raw_notes, summary, questions, topic, topic_type, source_type, source_files, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) 
            DO UPDATE SET 
                title = EXCLUDED.title,
                raw_notes = EXCLUDED.raw_notes,
                summary = EXCLUDED.summary,
                questions = EXCLUDED.questions,
                topic = EXCLUDED.topic,
                topic_type = EXCLUDED.topic_type,
                source_type = EXCLUDED.source_type,
                source_files = EXCLUDED.source_files
            RETURNING id
        ''', (
            summary_id,
            data.get('title', 'Meeting Summary'),
            data.get('rawNotes', ''),
            data.get('summary', ''),
            data.get('questions', ''),
            data.get('topic', 'General'),
            data.get('topicType', 'other'),
            data.get('sourceType', 'paste'),
            source_files,
            data.get('createdAt', datetime.utcnow().isoformat())
        ))
        
        result = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({'success': True, 'id': result['id']})
    except Exception as e:
        print(f"Error saving summary: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete-summary', methods=['POST'])
def delete_summary():
    """Delete a meeting summary"""
    try:
        data = request.json
        summary_id = data.get('id', '')
        
        if not summary_id:
            return jsonify({'error': 'Summary ID is required'}), 400
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('DELETE FROM meeting_summaries WHERE id = %s', (summary_id,))
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error deleting summary: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/email-summary-section', methods=['POST'])
def email_summary_section():
    """Email a specific section of a summary (takeaways or questions)"""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    import re
    
    try:
        data = request.json
        email = data.get('email', '')
        subject = data.get('subject', 'Summary Notes')
        section = data.get('section', '')  # 'takeaways' or 'questions'
        content = data.get('content', '')
        title = data.get('title', 'Meeting Summary')
        topic = data.get('topic', 'General')
        smtp_config = data.get('smtpConfig', {})
        
        if not email or not content:
            return jsonify({'error': 'Email and content are required'}), 400
        
        # Get SMTP configuration
        use_gmail = smtp_config.get('use_gmail', True)
        gmail_user = smtp_config.get('gmail_user', '')
        gmail_password = smtp_config.get('gmail_app_password', '')
        from_email = smtp_config.get('from_email', gmail_user)
        
        if use_gmail and (not gmail_user or not gmail_password):
            return jsonify({'error': 'Gmail credentials required. Please set them in Settings.'}), 400
        
        # Convert HTML to plain text
        plain_text = re.sub(r'<[^>]+>', '', content)
        plain_text = plain_text.replace('&nbsp;', ' ').replace('&amp;', '&')
        
        # Format the section label
        section_label = "Key Takeaways" if section == 'takeaways' else "Follow-up Questions"
        header_color = "#0d9488" if section == 'takeaways' else "#d97706"
        
        # Build HTML email
        html_content = f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1 {{
                    color: {header_color};
                    border-bottom: 2px solid {header_color};
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #374151;
                    margin-top: 24px;
                }}
                h3 {{
                    color: #4b5563;
                }}
                ul, ol {{
                    padding-left: 24px;
                }}
                li {{
                    margin-bottom: 8px;
                }}
                strong {{
                    color: #111;
                }}
                .header {{
                    background: linear-gradient(135deg, {header_color} 0%, {'#0891b2' if section == 'takeaways' else '#ea580c'} 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 24px;
                }}
                .topic-badge {{
                    display: inline-block;
                    background: rgba(255,255,255,0.2);
                    padding: 4px 12px;
                    border-radius: 12px;
                    font-size: 12px;
                    margin-top: 8px;
                }}
                .content {{
                    background: #f9fafb;
                    padding: 24px;
                    border-radius: 8px;
                    border: 1px solid #e5e7eb;
                }}
                .footer {{
                    margin-top: 24px;
                    padding-top: 16px;
                    border-top: 1px solid #e5e7eb;
                    font-size: 12px;
                    color: #6b7280;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1 style="color: white; border: none; margin: 0;">{section_label}</h1>
                <p style="margin: 8px 0 0 0; opacity: 0.9;">{title}</p>
                <span class="topic-badge">{topic}</span>
            </div>
            <div class="content">
                {content}
            </div>
            <div class="footer">
                Generated by TDL Equity Analyzer
            </div>
        </body>
        </html>
        """
        
        # Create email message
        msg = MIMEMultipart('alternative')
        msg['From'] = from_email
        msg['To'] = email
        msg['Subject'] = subject
        
        # Attach both plain text and HTML versions
        msg.attach(MIMEText(plain_text, 'plain'))
        msg.attach(MIMEText(html_content, 'html'))
        
        # Send via Gmail SMTP
        if use_gmail:
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(gmail_user, gmail_password)
                server.send_message(msg)
        
        return jsonify({'success': True, 'message': 'Email sent successfully'})
        
    except smtplib.SMTPAuthenticationError:
        return jsonify({'error': 'Gmail authentication failed. Check your email and app password.'}), 401
    except smtplib.SMTPException as e:
        return jsonify({'error': f'SMTP error: {str(e)}'}), 500
    except Exception as e:
        print(f"Error sending summary email: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/extract-summary-text', methods=['POST'])
def extract_summary_text():
    """Extract text from uploaded files (PDF, DOCX, images, TXT) for summary generation"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        all_text = []
        first_filename = files[0].filename
        
        for file in files:
            filename = file.filename.lower()
            file_content = file.read()
            extracted_text = ''
            
            try:
                # Handle PDF files
                if filename.endswith('.pdf'):
                    try:
                        import io
                        from PyPDF2 import PdfReader
                        pdf_reader = PdfReader(io.BytesIO(file_content))
                        for page in pdf_reader.pages:
                            text = page.extract_text()
                            if text:
                                extracted_text += text + '\n\n'
                    except ImportError:
                        # Fallback: try pdfplumber
                        try:
                            import pdfplumber
                            import io
                            with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                                for page in pdf.pages:
                                    text = page.extract_text()
                                    if text:
                                        extracted_text += text + '\n\n'
                        except ImportError:
                            return jsonify({'error': 'PDF processing libraries not available'}), 500
                
                # Handle Word documents
                elif filename.endswith('.docx') or filename.endswith('.doc'):
                    try:
                        import io
                        from docx import Document
                        doc = Document(io.BytesIO(file_content))
                        for para in doc.paragraphs:
                            if para.text.strip():
                                extracted_text += para.text + '\n'
                        # Also extract from tables
                        for table in doc.tables:
                            for row in table.rows:
                                row_text = ' | '.join(cell.text.strip() for cell in row.cells if cell.text.strip())
                                if row_text:
                                    extracted_text += row_text + '\n'
                    except ImportError:
                        return jsonify({'error': 'Word document processing library not available'}), 500
                
                # Handle plain text files
                elif filename.endswith('.txt'):
                    try:
                        extracted_text = file_content.decode('utf-8')
                    except UnicodeDecodeError:
                        extracted_text = file_content.decode('latin-1')
                
                # Handle images (try OCR if available)
                elif filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    try:
                        import pytesseract
                        from PIL import Image
                        import io
                        image = Image.open(io.BytesIO(file_content))
                        extracted_text = pytesseract.image_to_string(image)
                    except ImportError:
                        # OCR not available, try to describe what we have
                        extracted_text = f"[Image file: {file.filename} - OCR not available. Please copy text manually if needed.]"
                    except Exception as ocr_error:
                        extracted_text = f"[Image file: {file.filename} - Could not extract text: {str(ocr_error)}]"
                
                else:
                    # Try to read as text
                    try:
                        extracted_text = file_content.decode('utf-8')
                    except:
                        extracted_text = f"[Unsupported file type: {file.filename}]"
                
                if extracted_text.strip():
                    # Add filename header if multiple files
                    if len(files) > 1:
                        all_text.append(f"=== {file.filename} ===\n{extracted_text}")
                    else:
                        all_text.append(extracted_text)
                        
            except Exception as file_error:
                print(f"Error processing file {file.filename}: {file_error}")
                all_text.append(f"[Error processing {file.filename}: {str(file_error)}]")
        
        combined_text = '\n\n'.join(all_text)
        
        if not combined_text.strip():
            return jsonify({'error': 'Could not extract any text from the uploaded files'}), 400
        
        return jsonify({
            'success': True,
            'text': combined_text,
            'filename': first_filename,
            'fileCount': len(files)
        })
        
    except Exception as e:
        print(f"Error extracting text: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================
# DOCUMENT STORAGE ENDPOINTS
# ============================================

@app.route('/api/documents/<ticker>', methods=['GET'])
def get_documents(ticker):
    """Get all stored documents for a ticker"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute('''
            SELECT filename, file_type, mime_type, metadata, file_size, created_at
            FROM document_files 
            WHERE ticker = %s
            ORDER BY created_at DESC
        ''', (ticker.upper(),))
        docs = cur.fetchall()
        cur.close()
        conn.close()
        
        return jsonify({
            'documents': [{
                'filename': d['filename'],
                'fileType': d['file_type'],
                'mimeType': d['mime_type'],
                'metadata': d['metadata'] or {},
                'fileSize': d['file_size'],
                'createdAt': d['created_at'].isoformat() if d['created_at'] else None,
                'stored': True
            } for d in docs]
        })
    except Exception as e:
        print(f"Error getting documents: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/documents/<ticker>/content', methods=['GET'])
def get_documents_with_content(ticker):
    """Get all stored documents with file content for re-analysis"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute('''
            SELECT filename, file_data, file_type, mime_type, metadata
            FROM document_files 
            WHERE ticker = %s
        ''', (ticker.upper(),))
        docs = cur.fetchall()
        cur.close()
        conn.close()
        
        return jsonify({
            'documents': [{
                'filename': d['filename'],
                'fileData': d['file_data'],
                'fileType': d['file_type'],
                'mimeType': d['mime_type'],
                'metadata': d['metadata'] or {},
                'stored': True
            } for d in docs]
        })
    except Exception as e:
        print(f"Error getting document content: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/documents/save', methods=['POST'])
def save_documents():
    """Save document files to database for a ticker"""
    try:
        data = request.json
        ticker = data.get('ticker', '').upper()
        documents = data.get('documents', [])
        
        if not ticker:
            return jsonify({'error': 'Ticker is required'}), 400
        
        if not documents:
            return jsonify({'error': 'No documents provided'}), 400
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        saved_count = 0
        for doc in documents:
            filename = doc.get('filename')
            file_data = doc.get('fileData')
            file_type = doc.get('fileType', 'pdf')
            mime_type = doc.get('mimeType', 'application/pdf')
            metadata = doc.get('metadata', {})
            
            if not filename or not file_data:
                continue
            
            # Calculate approximate file size (base64 is ~1.33x original)
            file_size = len(file_data) * 3 // 4
            
            cur.execute('''
                INSERT INTO document_files (ticker, filename, file_data, file_type, mime_type, metadata, file_size)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, filename) 
                DO UPDATE SET 
                    file_data = EXCLUDED.file_data,
                    file_type = EXCLUDED.file_type,
                    mime_type = EXCLUDED.mime_type,
                    metadata = EXCLUDED.metadata,
                    file_size = EXCLUDED.file_size
            ''', (ticker, filename, file_data, file_type, mime_type, json.dumps(metadata), file_size))
            saved_count += 1
        
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({'success': True, 'savedCount': saved_count})
    except Exception as e:
        print(f"Error saving documents: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/documents/delete', methods=['POST'])
def delete_document():
    """Delete a specific document for a ticker"""
    try:
        data = request.json
        ticker = data.get('ticker', '').upper()
        filename = data.get('filename')
        
        if not ticker or not filename:
            return jsonify({'error': 'Ticker and filename are required'}), 400
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
            DELETE FROM document_files 
            WHERE ticker = %s AND filename = %s
        ''', (ticker, filename))
        deleted = cur.rowcount > 0
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({'success': True, 'deleted': deleted})
    except Exception as e:
        print(f"Error deleting document: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/documents/delete-all', methods=['POST'])
def delete_all_documents():
    """Delete all documents for a ticker"""
    try:
        data = request.json
        ticker = data.get('ticker', '').upper()
        
        if not ticker:
            return jsonify({'error': 'Ticker is required'}), 400
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('DELETE FROM document_files WHERE ticker = %s', (ticker,))
        deleted_count = cur.rowcount
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({'success': True, 'deletedCount': deleted_count})
    except Exception as e:
        print(f"Error deleting all documents: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/documents/storage-stats', methods=['GET'])
def get_storage_stats():
    """Get storage statistics"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute('''
            SELECT 
                ticker,
                COUNT(*) as doc_count,
                SUM(file_size) as total_size
            FROM document_files 
            GROUP BY ticker
            ORDER BY total_size DESC
        ''')
        stats = cur.fetchall()
        
        cur.execute('SELECT SUM(file_size) as total FROM document_files')
        total = cur.fetchone()
        
        cur.close()
        conn.close()
        
        return jsonify({
            'byTicker': [{
                'ticker': s['ticker'],
                'docCount': s['doc_count'],
                'totalSize': s['total_size'] or 0
            } for s in stats],
            'totalSize': total['total'] or 0 if total else 0
        })
    except Exception as e:
        print(f"Error getting storage stats: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================
# ANTHROPIC API PROXY ENDPOINTS
# ============================================

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
        historical_weights = data.get('historicalWeights', [])
        weighting_config = data.get('weightingConfig', {})
        
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
        
        # Check if using simple weighting mode
        simple_mode = weighting_config.get('mode') == 'simple'
        existing_weight = weighting_config.get('existingAnalysisWeight', 70) if simple_mode else None
        new_docs_weight = weighting_config.get('newDocsWeight', 30) if simple_mode else None
        
        # Build weighting information string
        weight_info = ""
        
        if simple_mode and existing_analysis:
            # Simple mode: clear instruction about preservation vs updates
            weight_info = f"""=== ANALYSIS UPDATE MODE: SIMPLE WEIGHTING ===

PRESERVATION RATIO: {existing_weight}% existing / {new_docs_weight}% new

This means:
- PRESERVE {existing_weight}% of the existing thesis, pillars, signposts, and threats
- Allow only {new_docs_weight}% worth of modifications from the new document(s)
- The new document is SUPPLEMENTARY, not a replacement

New document(s) being added:
"""
            for doc in enabled_docs:
                doc_name = doc.get('filename', 'document.pdf')
                weight_info += f"- {doc_name}\n"
            
            weight_info += f"""
Remember: With {existing_weight}% preservation, you should keep most existing content intact.
Only add minor refinements, new data points, or small additions from the new document.
Do NOT rewrite or fundamentally change the existing analysis.

"""
        elif simple_mode and not existing_analysis:
            # Simple mode but new analysis - just list documents
            weight_info = "DOCUMENT WEIGHTING:\n\n"
            weight_info += "NEW DOCUMENTS (being analyzed now):\n"
            for doc in enabled_docs:
                doc_name = doc.get('filename', 'document.pdf')
                weight_info += f"- {doc_name}\n"
            weight_info += "\n"
        elif not simple_mode:
            # Advanced mode: per-document weights
            # Calculate total weight including both new and historical docs
            new_doc_weight = sum(doc.get('weight', 1) for doc in enabled_docs)
            hist_doc_weight = sum(hw.get('weight', 1) for hw in historical_weights)
            total_weight = new_doc_weight + hist_doc_weight
            
            # Historical documents (from existing analysis)
            if historical_weights:
                weight_info += "PREVIOUSLY ANALYZED DOCUMENTS (their insights are in the existing analysis):\n"
                for hw in historical_weights:
                    hw_name = hw.get('filename', 'document')
                    hw_weight = hw.get('weight', 1)
                    hw_pct = round((hw_weight / total_weight) * 100) if total_weight > 0 else 0
                    weight_info += f"- {hw_name}: {hw_pct}% weight\n"
                weight_info += "\n"
            
            # New documents being analyzed now
            weight_info += "NEW DOCUMENTS (being analyzed now):\n"
            for doc in enabled_docs:
                doc_name = doc.get('filename', 'document.pdf')
                doc_weight = doc.get('weight', 1)
                doc_pct = round((doc_weight / total_weight) * 100) if total_weight > 0 else 0
                weight_info += f"- {doc_name}: {doc_pct}% weight\n"
            
            weight_info += "\nWhen synthesizing the analysis:\n"
            weight_info += "- Give MORE emphasis to higher-weighted documents\n"
            weight_info += "- If updating existing analysis, respect the weights of previously analyzed documents\n"
            weight_info += "- Higher-weighted historical docs = keep more of their conclusions in the existing analysis\n"
        
        content.append({
            "type": "text",
            "text": weight_info
        })
        
        # Calculate total weight for document headers (use simple mode weight or calculated weight)
        if simple_mode and existing_analysis:
            # In simple mode, all new docs share the new_docs_weight equally
            per_doc_weight = new_docs_weight / len(enabled_docs) if enabled_docs else 0
        else:
            new_doc_weight = sum(doc.get('weight', 1) for doc in enabled_docs)
            hist_doc_weight = sum(hw.get('weight', 1) for hw in historical_weights)
            total_weight = new_doc_weight + hist_doc_weight
        
        # Add each document
        for doc in enabled_docs:
            doc_content = doc.get('fileData', '')
            doc_name = doc.get('filename', 'document.pdf')
            doc_type = doc.get('fileType', 'pdf')
            mime_type = doc.get('mimeType', 'application/pdf')
            
            if simple_mode and existing_analysis:
                doc_pct = round(per_doc_weight)
                # In simple mode, label as supplementary to reinforce it's not primary
                doc_header = f"\n=== NEW DOCUMENT (Supplementary - {doc_pct}% update weight): {doc_name} ==="
            else:
                doc_weight = doc.get('weight', 1)
                doc_pct = round((doc_weight / total_weight) * 100) if total_weight > 0 else 0
                doc_header = f"\n=== DOCUMENT: {doc_name} (Weight: {doc_pct}%) ==="
            
            if not doc_content:
                continue
            
            # Add document header with weight
            content.append({
                "type": "text",
                "text": doc_header
            })
                
            if doc_type == 'pdf':
                content.append({
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": doc_content
                    }
                })
            elif doc_type == 'image':
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type or "image/png",
                        "data": doc_content
                    }
                })
            else:
                try:
                    decoded_text = base64.b64decode(doc_content).decode('utf-8')
                    content.append({
                        "type": "text",
                        "text": decoded_text
                    })
                except:
                    continue
        
        if not content:
            return jsonify({'error': 'No valid documents to analyze'}), 400
        
        # Add the analysis prompt
        analysis_prompt = """Analyze these broker research documents and create a comprehensive investment analysis.

Return a JSON object with this exact structure:
{
    "ticker": "STOCK_TICKER",
    "company": "Company Name",
    "thesis": {
        "summary": "2-3 sentence investment thesis summary",
        "pillars": [
            {"title": "Pillar 1 Title", "description": "Detailed explanation", "confidence": "High/Medium/Low", "sources": [{"filename": "Document name", "excerpt": "Brief supporting quote"}]},
            {"title": "Pillar 2 Title", "description": "Detailed explanation", "confidence": "High/Medium/Low", "sources": [{"filename": "Document name", "excerpt": "Brief supporting quote"}]}
        ]
    },
    "signposts": [
        {"metric": "Key metric or KPI name", "target": "Target value or outcome", "timeframe": "When to expect", "category": "Financial/Operational/Strategic/Market", "confidence": "High/Medium/Low", "sources": [{"filename": "Document name", "excerpt": "Brief supporting quote"}]},
        {"metric": "Another metric name", "target": "Target", "timeframe": "Timeframe", "category": "Financial/Operational/Strategic/Market", "confidence": "High/Medium/Low", "sources": [{"filename": "Document name", "excerpt": "Brief supporting quote"}]}
    ],
    "threats": [
        {"threat": "Risk factor description", "likelihood": "High/Medium/Low", "impact": "High/Medium/Low", "triggerPoints": "What to watch for - early warning signs", "sources": [{"filename": "Document name", "excerpt": "Brief supporting quote"}]},
        {"threat": "Another risk", "likelihood": "Medium", "impact": "High", "triggerPoints": "Monitoring triggers", "sources": [{"filename": "Document name", "excerpt": "Brief supporting quote"}]}
    ],
    "documentMetadata": [
        {"filename": "exact_filename.pdf", "docType": "broker_report", "source": "Citi", "publishDate": "YYYY-MM-DD", "authors": ["Analyst Name"], "title": "Report Title"},
        {"filename": "transcript.pdf", "docType": "earnings_call", "source": "Company Name", "publishDate": "YYYY-MM-DD", "quarter": "Q3 2025", "title": "Q3 2025 Earnings Call"},
        {"filename": "email_screenshot.png", "docType": "email", "source": "Sender Name/Org", "publishDate": "YYYY-MM-DD", "title": "Email Subject"}
    ]
}

DOCUMENT METADATA EXTRACTION (CRITICAL):
For EACH document provided, identify the document type and extract appropriate metadata:

**For Broker Reports:**
- "docType": "broker_report"
- "source": Investment bank/broker name (e.g., "Citi", "Morgan Stanley", "Goldman Sachs", "Wolfe Research")
- "publishDate": Report date in YYYY-MM-DD format
- "authors": Array of analyst names
- "title": Report title/headline

**For Earnings Call Transcripts:**
- "docType": "earnings_call"  
- "source": Company name (e.g., "Union Pacific", "Apple Inc.")
- "publishDate": Call date in YYYY-MM-DD format
- "quarter": Fiscal quarter (e.g., "Q3 2025", "FY 2025")
- "title": e.g., "Q3 2025 Earnings Call Transcript"

**For SEC Filings (10-K, 10-Q, 8-K):**
- "docType": "sec_filing"
- "source": Company name
- "publishDate": Filing date in YYYY-MM-DD format
- "filingType": "10-K", "10-Q", "8-K", etc.
- "title": Filing description

**For Emails/Email Screenshots:**
- "docType": "email"
- "source": Sender name or organization
- "publishDate": Email date in YYYY-MM-DD format
- "title": Email subject line if visible

**For Company Presentations:**
- "docType": "presentation"
- "source": Company name or presenting organization
- "publishDate": Presentation date in YYYY-MM-DD format
- "title": Presentation title

**For News Articles:**
- "docType": "news"
- "source": Publication name (e.g., "Wall Street Journal", "Reuters")
- "publishDate": Article date in YYYY-MM-DD format
- "authors": Array of journalist names if visible
- "title": Article headline

**For Screenshots/Images (if content type unclear):**
- "docType": "screenshot"
- "source": Infer source if visible in image
- "publishDate": Infer date if visible, otherwise null
- "title": Brief description of content

Always include "filename" with the exact filename provided.

IMPORTANT STYLE RULES:
- Do NOT reference any sellside broker names (e.g., "Goldman Sachs believes...", "According to Morgan Stanley...")
- Do NOT reference specific analyst names
- Do NOT include specific broker price targets
- Write as independent analysis that synthesizes the information without attribution to sources in the prose
- The output should read like original independent research, not a summary of broker views

DOCUMENT WEIGHTING:
- Each document has an assigned weight percentage shown at the start
- Give MORE emphasis to higher-weighted documents when forming conclusions
- Higher-weighted documents should have more influence on the thesis, signposts, and threats
- If documents conflict, prefer the view from the higher-weighted document

Focus on:
1. Why own this stock? (Investment Thesis) - include confidence level and source citations
2. What are we looking for? (Signposts - specific KPIs, events, milestones with metric names)
3. Where can we be wrong? (Threats - bear case scenarios with likelihood, impact, and trigger points)

For each pillar, signpost, and threat, include:
- "sources": Array of source documents that support this point, with filename and a brief excerpt
- Use the actual document filenames provided in the analysis

Return ONLY valid JSON, no markdown, no explanation."""

        if existing_analysis:
            # Build weighting instruction specific to the mode
            if simple_mode:
                weighting_instruction = f"""
CRITICAL WEIGHTING INSTRUCTION (SIMPLE MODE):
You MUST preserve {existing_weight}% of the existing analysis. The new documents can only contribute {new_docs_weight}% worth of changes.

What this means:
- KEEP {existing_weight}% of the existing thesis, pillars, signposts, and threats UNCHANGED
- Only make MINOR refinements or additions based on the new document(s)
- Do NOT fundamentally rewrite or replace the existing analysis
- Do NOT treat the new document as a "primary source" - it is a SUPPLEMENTARY source
- The new document should ADD to or SLIGHTLY REFINE the existing analysis, not replace it

Example of correct behavior with {existing_weight}% existing / {new_docs_weight}% new:
- If existing thesis has 3 pillars, keep all 3, maybe slightly update wording or add a 4th minor pillar
- If existing has 5 signposts, keep them mostly intact, maybe add 1-2 new ones or update targets slightly
- Do NOT remove or majorly rewrite existing content unless it's factually contradicted

In the "changes" array, describe what minor updates were made, NOT that you've rewritten the analysis.
"""
            else:
                weighting_instruction = """
DOCUMENT WEIGHTING:
- Each document has an assigned weight percentage shown at the start
- Give MORE emphasis to higher-weighted documents when forming conclusions
- Higher-weighted documents should have more influence on the thesis, signposts, and threats
- If documents conflict, prefer the view from the higher-weighted document
"""
            
            analysis_prompt = f"""Update this existing analysis with new information from the documents.

Existing Analysis:
{json.dumps(existing_analysis, indent=2)}

{weighting_instruction}

Review the new documents and:
1. Update or confirm the investment thesis (respecting the weighting above)
2. Add any new signposts or update existing ones
3. Add any new threats or update existing ones
4. Note what has changed in the "changes" array
5. Update sources for each point based on all documents analyzed
6. Extract metadata for ALL documents (both new and from existing analysis)

DOCUMENT METADATA EXTRACTION (CRITICAL):
For EACH document (new AND previously analyzed), identify the document type and extract appropriate metadata:

**For Broker Reports:** docType="broker_report", source=Broker name, publishDate, authors=Analyst names, title
**For Earnings Calls:** docType="earnings_call", source=Company name, publishDate, quarter, title
**For SEC Filings:** docType="sec_filing", source=Company, publishDate, filingType, title
**For Emails:** docType="email", source=Sender, publishDate, title=Subject
**For Presentations:** docType="presentation", source=Company/Org, publishDate, title
**For News:** docType="news", source=Publication, publishDate, authors, title
**For Screenshots:** docType="screenshot", source=Inferred source, publishDate=if visible, title=description

For previously analyzed documents in the existing analysis, use the filenames from documentHistory and extract what metadata you can infer from the existing analysis context.

IMPORTANT STYLE RULES:
- Do NOT reference any sellside broker names (e.g., "Goldman Sachs believes...", "According to Morgan Stanley...")
- Do NOT reference specific analyst names
- Do NOT include specific broker price targets
- Write as independent analysis that synthesizes the information without attribution to sources in the prose
- The output should read like original independent research, not a summary of broker views

For each pillar, signpost, and threat, include:
- "sources": Array of source documents that support this point, with filename and a brief excerpt
- "confidence": High/Medium/Low for pillars and signposts
- Use the actual document filenames provided

Return the updated analysis as JSON with the same structure (including "documentMetadata" array), plus a "changes" array describing what minor updates were made.

Return ONLY valid JSON, no markdown, no explanation."""

        content.append({
            "type": "text",
            "text": analysis_prompt
        })
        
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
        
        response = requests.post(ANTHROPIC_API_URL, headers=headers, json=payload, timeout=180)
        
        if response.status_code != 200:
            error_data = response.json()
            error_msg = error_data.get('error', {}).get('message', 'API request failed')
            return jsonify({'error': error_msg}), response.status_code
        
        result = response.json()
        assistant_content = result.get('content', [{}])[0].get('text', '')
        
        # Parse the JSON response
        try:
            cleaned = assistant_content.strip()
            if cleaned.startswith('```'):
                cleaned = cleaned.split('\n', 1)[1]
            if cleaned.endswith('```'):
                cleaned = cleaned.rsplit('\n', 1)[0]
            if cleaned.startswith('json'):
                cleaned = cleaned[4:].strip()
            
            analysis = json.loads(cleaned)
            changes = analysis.pop('changes', [])
            document_metadata = analysis.pop('documentMetadata', [])
            
            return jsonify({
                'analysis': analysis,
                'changes': changes,
                'documentMetadata': document_metadata,
                'usage': result.get('usage', {})
            })
            
        except json.JSONDecodeError as e:
            return jsonify({
                'error': f'Failed to parse analysis: {str(e)}',
                'raw_response': assistant_content
            }), 500
        
    except requests.Timeout:
        return jsonify({'error': 'Request timed out. Try with fewer or smaller documents.'}), 504
    except Exception as e:
        import traceback
        print(f"Error in analyze-multi: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/api/parse', methods=['POST'])
def parse():
    """Use Claude to intelligently parse stock analysis into sections"""
    try:
        data = request.json
        api_key = data.get('api_key')
        content = data.get('content', '')
        
        if not api_key:
            return jsonify({'error': 'API key is required'}), 400
        
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


# ============================================
# EMAIL ENDPOINTS
# ============================================

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
        
        if not all([smtp_server, email, password, recipient, subject, body]):
            return jsonify({'error': 'Missing required email fields'}), 400
        
        msg = MIMEMultipart()
        msg['From'] = email
        msg['To'] = recipient
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
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
        
        use_gmail = smtp_config.get('use_gmail', True)
        gmail_user = smtp_config.get('gmail_user', '')
        gmail_password = smtp_config.get('gmail_app_password', '')
        from_email = smtp_config.get('from_email', gmail_user)
        
        if not recipient:
            return jsonify({'error': 'Recipient email is required'}), 400
        
        if use_gmail and (not gmail_user or not gmail_password):
            return jsonify({'error': 'Gmail credentials required'}), 400
        
        msg = MIMEMultipart('alternative')
        msg['From'] = from_email
        msg['To'] = recipient
        msg['Subject'] = subject
        
        plain_text = html_body.replace('<h1>', '\n').replace('</h1>', '\n' + '='*50 + '\n')
        plain_text = plain_text.replace('<h2>', '\n\n').replace('</h2>', '\n' + '-'*30 + '\n')
        plain_text = plain_text.replace('<p>', '').replace('</p>', '\n')
        plain_text = plain_text.replace('<br>', '\n').replace('<em>', '').replace('</em>', '')
        
        msg.attach(MIMEText(plain_text, 'plain'))
        msg.attach(MIMEText(html_body, 'html'))
        
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
    """Send Analysis email via SMTP with HTML formatting"""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    try:
        data = request.json
        analysis = data.get('analysis', {})
        recipient = data.get('email')
        smtp_config = data.get('smtpConfig', {})
        
        ticker = analysis.get('ticker', 'Stock')
        company = analysis.get('company', '')
        
        # Default subject if not provided
        default_subject = f"{ticker} - Investment Analysis"
        subject = data.get('customSubject') or default_subject
        
        use_gmail = smtp_config.get('use_gmail', True)
        gmail_user = smtp_config.get('gmail_user', '')
        gmail_password = smtp_config.get('gmail_app_password', '')
        from_email = smtp_config.get('from_email', gmail_user)
        
        if not recipient:
            return jsonify({'error': 'Recipient email is required'}), 400
        
        if use_gmail and (not gmail_user or not gmail_password):
            return jsonify({'error': 'Gmail credentials required'}), 400
        
        thesis = analysis.get('thesis', {})
        signposts = analysis.get('signposts', [])
        threats = analysis.get('threats', [])
        
        # Build HTML email
        html_body = f"""
<html>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 700px;">
    <h1 style="color: #1a365d; border-bottom: 2px solid #2c5282; padding-bottom: 10px;">{ticker} - {company}</h1>
    
    <h2 style="color: #2c5282; margin-top: 25px;">1. Investment Thesis</h2>
    <p style="margin-left: 20px;">{thesis.get('summary', 'N/A')}</p>
"""
        
        if thesis.get('pillars'):
            html_body += '<ul style="margin-left: 20px;">'
            for pillar in thesis['pillars']:
                title = pillar.get('pillar', pillar.get('title', ''))
                desc = pillar.get('detail', pillar.get('description', ''))
                html_body += f'<li style="margin-bottom: 8px;"><strong>{title}:</strong> {desc}</li>'
            html_body += '</ul>'
        
        html_body += '<h2 style="color: #2c5282; margin-top: 25px;">2. Signposts (What We\'re Watching)</h2>'
        html_body += '<ul style="margin-left: 20px;">'
        for sp in signposts:
            metric = sp.get('metric', sp.get('signpost', ''))
            target = sp.get('target', '')
            timeframe = sp.get('timeframe', '')
            html_body += f'<li style="margin-bottom: 8px;"><strong>{metric}:</strong> {target}'
            if timeframe:
                html_body += f' <em>({timeframe})</em>'
            html_body += '</li>'
        html_body += '</ul>'
        
        html_body += '<h2 style="color: #2c5282; margin-top: 25px;">3. Thesis Threats (Where We Can Be Wrong)</h2>'
        html_body += '<ul style="margin-left: 20px;">'
        for threat in threats:
            threat_desc = threat.get('threat', '')
            likelihood = threat.get('likelihood', '')
            impact = threat.get('impact', '')
            triggers = threat.get('triggerPoints', '')
            html_body += f'<li style="margin-bottom: 10px;"><strong>{threat_desc}</strong>'
            if likelihood or impact:
                html_body += f'<br><span style="color: #666; font-size: 0.9em;">Likelihood: {likelihood} | Impact: {impact}</span>'
            if triggers:
                html_body += f'<br><span style="color: #666; font-size: 0.9em;">Watch for: {triggers}</span>'
            html_body += '</li>'
        html_body += '</ul>'
        
        html_body += """
</body>
</html>
"""
        
        # Plain text version
        plain_text = f"{ticker} - {company}\n\n"
        plain_text += "1. INVESTMENT THESIS\n"
        plain_text += f"{thesis.get('summary', 'N/A')}\n\n"
        
        if thesis.get('pillars'):
            for pillar in thesis['pillars']:
                title = pillar.get('pillar', pillar.get('title', ''))
                desc = pillar.get('detail', pillar.get('description', ''))
                plain_text += f"  - {title}: {desc}\n"
        
        plain_text += "\n2. SIGNPOSTS\n"
        for sp in signposts:
            metric = sp.get('metric', sp.get('signpost', ''))
            target = sp.get('target', '')
            plain_text += f"  - {metric}: {target}\n"
        
        plain_text += "\n3. THESIS THREATS\n"
        for threat in threats:
            plain_text += f"  - {threat.get('threat', '')}\n"
        
        msg = MIMEMultipart('alternative')
        msg['From'] = from_email
        msg['To'] = recipient
        msg['Subject'] = subject
        
        msg.attach(MIMEText(plain_text, 'plain'))
        msg.attach(MIMEText(html_body, 'html'))
        
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


# ============================================
# PDF EXTRACTION ENDPOINT
# ============================================

@app.route('/api/extract-pdf', methods=['POST'])
def extract_pdf():
    """
    Extract text from uploaded PDF file.
    Used by Research tab for document analysis.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read PDF and extract text
        from PyPDF2 import PdfReader
        import io
        
        pdf_bytes = file.read()
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        
        text_content = []
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_content.append(page_text)
        
        full_text = '\n\n'.join(text_content)
        
        if not full_text.strip():
            return jsonify({'error': 'Could not extract text from PDF. It may be scanned/image-based.'}), 400
        
        return jsonify({
            'text': full_text,
            'pages': len(pdf_reader.pages),
            'filename': file.filename
        })
        
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return jsonify({'error': f'Failed to extract PDF: {str(e)}'}), 500


# ============================================
# RESEARCH ANALYSIS ENDPOINT
# ============================================

@app.route('/api/research-analyze', methods=['POST'])
def research_analyze():
    """
    Deep analysis of sell-side research using customizable prompt frameworks.
    Calls Anthropic API directly for better control and reliability.
    
    Request body:
    {
        "text": "Full prompt with document content",
        "promptId": "executive-brief",
        "promptName": "Executive Brief",
        "apiKey": "sk-ant-..." (from user's Settings)
    }
    """
    try:
        data = request.json
        text = data.get('text', '')
        prompt_id = data.get('promptId', '')
        prompt_name = data.get('promptName', '')
        api_key = data.get('apiKey', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Get API key - from request or fallback to environment
        if not api_key:
            api_key = os.environ.get('ANTHROPIC_API_KEY', '')
        
        if not api_key:
            return jsonify({'error': 'No API key provided. Please add your API key in Settings.'}), 400
        
        # Initialize Anthropic client
        client = anthropic.Anthropic(api_key=api_key)
        
        # Call Claude API
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": text
                }
            ]
        )
        
        # Extract response text
        result_text = ""
        for block in message.content:
            if hasattr(block, 'text'):
                result_text += block.text
        
        return jsonify({
            'result': result_text,
            'promptId': prompt_id,
            'promptName': prompt_name,
            'usage': {
                'input_tokens': message.usage.input_tokens,
                'output_tokens': message.usage.output_tokens
            }
        })
        
    except anthropic.APIConnectionError as e:
        print(f"API Connection Error: {e}")
        return jsonify({'error': 'Failed to connect to Anthropic API. Check your internet connection.'}), 503
    except anthropic.RateLimitError as e:
        print(f"Rate Limit Error: {e}")
        return jsonify({'error': 'Rate limit exceeded. Please wait a moment and try again.'}), 429
    except anthropic.AuthenticationError as e:
        print(f"Auth Error: {e}")
        return jsonify({'error': 'Invalid API key. Please check your API key in Settings.'}), 401
    except anthropic.APIStatusError as e:
        print(f"API Status Error: {e}")
        return jsonify({'error': f'API error: {e.message}'}), e.status_code
    except Exception as e:
        print(f"Research analysis error: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================
# RESEARCH HIERARCHICAL ENDPOINTS
# ============================================

# --- Categories ---
@app.route('/api/research-categories', methods=['GET'])
def get_research_categories():
    """Get all research categories"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT id, name, type, created_at FROM research_categories ORDER BY created_at DESC')
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        return jsonify([{
            'id': row['id'],
            'name': row['name'],
            'type': row['type'],
            'createdAt': row['created_at'].isoformat() if row['created_at'] else None
        } for row in rows])
    except Exception as e:
        print(f"Error getting research categories: {e}")
        return jsonify([])


@app.route('/api/save-research-category', methods=['POST'])
def save_research_category():
    """Save a research category"""
    try:
        data = request.json
        cat_id = data.get('id', '')
        
        if not cat_id:
            return jsonify({'error': 'Category ID is required'}), 400
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute('''
            INSERT INTO research_categories (id, name, type)
            VALUES (%s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name, type = EXCLUDED.type
            RETURNING id
        ''', (cat_id, data.get('name', ''), data.get('type', 'ticker')))
        
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error saving research category: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/delete-research-category', methods=['POST'])
def delete_research_category():
    """Delete a research category and all its documents/analyses"""
    try:
        data = request.json
        cat_id = data.get('id', '')
        
        if not cat_id:
            return jsonify({'error': 'Category ID is required'}), 400
        
        conn = get_db_connection()
        cur = conn.cursor()
        # CASCADE will delete documents and analyses
        cur.execute('DELETE FROM research_categories WHERE id = %s', (cat_id,))
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error deleting research category: {e}")
        return jsonify({'error': str(e)}), 500


# --- Documents ---
@app.route('/api/research-documents', methods=['GET'])
def get_research_documents():
    """Get all research documents"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT id, category_id, name, content, file_names, smart_name, original_filename, published_date, has_stored_files, created_at FROM research_documents ORDER BY created_at DESC')
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        return jsonify([{
            'id': row['id'],
            'categoryId': row['category_id'],
            'name': row['name'],
            'content': row['content'],
            'fileNames': row['file_names'] or [],
            'smartName': row['smart_name'],
            'originalFilename': row['original_filename'],
            'publishedDate': row['published_date'],
            'hasStoredFiles': row['has_stored_files'] or False,
            'createdAt': row['created_at'].isoformat() if row['created_at'] else None
        } for row in rows])
    except Exception as e:
        print(f"Error getting research documents: {e}")
        return jsonify([])


@app.route('/api/save-research-document', methods=['POST'])
def save_research_document():
    """Save a research document"""
    try:
        data = request.json
        doc_id = data.get('id', '')
        
        print(f"📄 save_research_document: id={doc_id}, name={data.get('name', '')[:50]}")
        
        if not doc_id:
            return jsonify({'error': 'Document ID is required'}), 400
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute('''
            INSERT INTO research_documents (id, category_id, name, content, file_names, smart_name, original_filename, published_date, has_stored_files)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET 
                name = EXCLUDED.name,
                content = EXCLUDED.content,
                file_names = EXCLUDED.file_names,
                smart_name = EXCLUDED.smart_name,
                original_filename = EXCLUDED.original_filename,
                published_date = EXCLUDED.published_date,
                has_stored_files = EXCLUDED.has_stored_files
            RETURNING id
        ''', (
            doc_id,
            data.get('categoryId', ''),
            data.get('name', ''),
            data.get('content', ''),
            json.dumps(data.get('fileNames', [])),
            data.get('smartName'),
            data.get('originalFilename'),
            data.get('publishedDate'),
            data.get('hasStoredFiles', False)
        ))
        
        conn.commit()
        cur.close()
        conn.close()
        
        print(f"✅ Document saved: {doc_id}")
        return jsonify({'success': True})
    except Exception as e:
        print(f"❌ Error saving research document: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/save-research-file', methods=['POST'])
def save_research_file():
    """Save a file for a research document"""
    conn = None
    cur = None
    try:
        data = request.json
        if not data:
            print("❌ No JSON data received")
            return jsonify({'error': 'No JSON data received'}), 400
            
        document_id = data.get('documentId', '')
        filename = data.get('filename', '')
        file_type = data.get('fileType', '')
        file_data = data.get('fileData', '')
        file_size = data.get('fileSize', 0)
        
        print(f"📁 save_research_file: docId={document_id}, filename={filename}, fileType={file_type}, dataLen={len(file_data) if file_data else 0}, fileSize={file_size}")
        
        if not document_id or not filename:
            print(f"❌ Missing required fields: docId={document_id}, filename={filename}")
            return jsonify({'error': 'Document ID and filename are required'}), 400
        
        if not file_data:
            print(f"❌ No file data provided for {filename}")
            return jsonify({'error': 'No file data provided'}), 400
        
        # Check if document exists first
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute('SELECT id FROM research_documents WHERE id = %s', (document_id,))
        doc_exists = cur.fetchone()
        if not doc_exists:
            print(f"❌ Document {document_id} does not exist in research_documents table")
            cur.close()
            conn.close()
            return jsonify({'error': f'Document {document_id} not found - must save document first'}), 400
        
        print(f"✅ Document {document_id} exists, proceeding with file save")
        
        cur.execute('''
            INSERT INTO research_document_files (document_id, filename, file_type, file_data, file_size)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        ''', (document_id, filename, file_type, file_data, file_size))
        
        result = cur.fetchone()
        if result is None:
            print(f"❌ INSERT did not return an id")
            conn.rollback()
            cur.close()
            conn.close()
            return jsonify({'error': 'Insert failed - no id returned'}), 500
            
        inserted_id = result['id']  # Access by column name since RealDictCursor is default
        conn.commit()
        cur.close()
        conn.close()
        
        print(f"✅ File saved successfully: id={inserted_id}, filename={filename}")
        return jsonify({'success': True, 'id': inserted_id})
    except Exception as e:
        print(f"❌ Error saving research file: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        if cur:
            try:
                cur.close()
            except:
                pass
        if conn:
            try:
                conn.rollback()
                conn.close()
            except:
                pass
        return jsonify({'error': f'{type(e).__name__}: {str(e)}'}), 500


@app.route('/api/research-document-files/<document_id>', methods=['GET'])
def get_research_document_files(document_id):
    """Get stored files for a research document"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute('''
            SELECT id, filename, file_type, file_data, file_size, created_at 
            FROM research_document_files 
            WHERE document_id = %s
            ORDER BY created_at
        ''', (document_id,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        return jsonify([{
            'id': row['id'],
            'filename': row['filename'],
            'fileType': row['file_type'],
            'fileData': row['file_data'],
            'fileSize': row['file_size'],
            'createdAt': row['created_at'].isoformat() if row['created_at'] else None
        } for row in rows])
    except Exception as e:
        print(f"Error getting research document files: {e}")
        return jsonify([])


@app.route('/api/delete-research-file/<int:file_id>', methods=['DELETE'])
def delete_research_file(file_id):
    """Delete a stored research file"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('DELETE FROM research_document_files WHERE id = %s', (file_id,))
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error deleting research file: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/delete-research-document', methods=['POST'])
def delete_research_document():
    """Delete a research document and all its analyses"""
    try:
        data = request.json
        doc_id = data.get('id', '')
        
        if not doc_id:
            return jsonify({'error': 'Document ID is required'}), 400
        
        conn = get_db_connection()
        cur = conn.cursor()
        # CASCADE will delete analyses
        cur.execute('DELETE FROM research_documents WHERE id = %s', (doc_id,))
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error deleting research document: {e}")
        return jsonify({'error': str(e)}), 500


# --- Analyses ---
@app.route('/api/research-analyses', methods=['GET'])
def get_research_analyses():
    """Get all research analyses"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT id, document_id, prompt_id, prompt_name, prompt_icon, result, usage, created_at FROM research_analyses ORDER BY created_at DESC')
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        return jsonify([{
            'id': row['id'],
            'documentId': row['document_id'],
            'promptId': row['prompt_id'],
            'promptName': row['prompt_name'],
            'promptIcon': row['prompt_icon'],
            'result': row['result'],
            'usage': row['usage'],
            'createdAt': row['created_at'].isoformat() if row['created_at'] else None
        } for row in rows])
    except Exception as e:
        print(f"Error getting research analyses: {e}")
        return jsonify([])


@app.route('/api/save-research-analysis', methods=['POST'])
def save_research_analysis():
    """Save a research analysis"""
    try:
        data = request.json
        analysis_id = data.get('id', '')
        
        if not analysis_id:
            return jsonify({'error': 'Analysis ID is required'}), 400
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute('''
            INSERT INTO research_analyses (id, document_id, prompt_id, prompt_name, prompt_icon, result, usage)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET 
                result = EXCLUDED.result,
                usage = EXCLUDED.usage
            RETURNING id
        ''', (
            analysis_id,
            data.get('documentId', ''),
            data.get('promptId', ''),
            data.get('promptName', ''),
            data.get('promptIcon', ''),
            data.get('result', ''),
            json.dumps(data.get('usage', {}))
        ))
        
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error saving research analysis: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/delete-research-analysis', methods=['POST'])
def delete_research_analysis():
    """Delete a research analysis"""
    try:
        data = request.json
        analysis_id = data.get('id', '')
        
        if not analysis_id:
            return jsonify({'error': 'Analysis ID is required'}), 400
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('DELETE FROM research_analyses WHERE id = %s', (analysis_id,))
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error deleting research analysis: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/email-research', methods=['POST'])
def email_research():
    """Email a research result"""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    import markdown
    
    try:
        data = request.json
        recipient = data.get('email', '')
        subject = data.get('subject', 'Research Analysis')
        content = data.get('content', '')
        prompt_name = data.get('promptName', '')
        ticker = data.get('ticker', '')
        topic = data.get('topic', '')
        smtp_config = data.get('smtpConfig', {})
        
        if not recipient:
            return jsonify({'error': 'Recipient email is required'}), 400
        
        # Build HTML email
        header_info = []
        if ticker:
            header_info.append(f"<strong>Ticker:</strong> {ticker}")
        if topic:
            header_info.append(f"<strong>Topic:</strong> {topic}")
        if prompt_name:
            header_info.append(f"<strong>Framework:</strong> {prompt_name}")
        
        header_html = " | ".join(header_info) if header_info else ""
        
        # Convert markdown to HTML
        try:
            content_html = markdown.markdown(content, extensions=['tables', 'fenced_code'])
        except:
            content_html = f"<pre style='white-space: pre-wrap;'>{content}</pre>"
        
        html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; }}
                h1, h2, h3 {{ color: #1a1a2e; }}
                .header {{ background: linear-gradient(135deg, #0f172a, #1e293b); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                .header h1 {{ margin: 0 0 10px 0; color: white; }}
                .header-meta {{ font-size: 14px; opacity: 0.9; }}
                .content {{ background: #f8fafc; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0; }}
                table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                th, td {{ border: 1px solid #e2e8f0; padding: 10px; text-align: left; }}
                th {{ background: #f1f5f9; }}
                code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-size: 14px; }}
                pre {{ background: #1e293b; color: #e2e8f0; padding: 15px; border-radius: 8px; overflow-x: auto; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Research Analysis</h1>
                <div class="header-meta">{header_html}</div>
            </div>
            <div class="content">
                {content_html}
            </div>
            <p style="color: #64748b; font-size: 12px; margin-top: 30px; text-align: center;">
                Generated by Charlie - Equity Analyzer
            </p>
        </body>
        </html>
        """
        
        # Send email
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['To'] = recipient
        
        # Plain text version
        msg.attach(MIMEText(content, 'plain'))
        # HTML version
        msg.attach(MIMEText(html_body, 'html'))
        
        if smtp_config.get('use_gmail') and smtp_config.get('gmail_user') and smtp_config.get('gmail_app_password'):
            msg['From'] = smtp_config['gmail_user']
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(smtp_config['gmail_user'], smtp_config['gmail_app_password'])
                server.send_message(msg)
        else:
            return jsonify({'error': 'Email not configured. Please set up Gmail in Settings.'}), 400
        
        return jsonify({'success': True, 'message': 'Research email sent successfully'})
        
    except smtplib.SMTPAuthenticationError:
        return jsonify({'error': 'Gmail authentication failed. Check your email and app password.'}), 401
    except smtplib.SMTPException as e:
        return jsonify({'error': f'SMTP error: {str(e)}'}), 500
    except Exception as e:
        print(f"Error sending research email: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================
# HEALTH CHECK
# ============================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'database': 'postgresql'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("=" * 50)
    print("TDL Equity Analyzer - Backend Server")
    print("With PostgreSQL Database Support")
    print("=" * 50)
    print(f"Starting server on http://0.0.0.0:{port}")
    print("=" * 50)
    app.run(host='0.0.0.0', port=port, debug=False)
