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
        
        # Calculate total weight including both new and historical docs
        new_doc_weight = sum(doc.get('weight', 1) for doc in enabled_docs)
        hist_doc_weight = sum(hw.get('weight', 1) for hw in historical_weights)
        total_weight = new_doc_weight + hist_doc_weight
        
        # Add document weighting information at the start
        weight_info = "DOCUMENT WEIGHTING (use these weights to prioritize information):\n\n"
        
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
        
        # Add each document
        for doc in enabled_docs:
            doc_content = doc.get('fileData', '')
            doc_name = doc.get('filename', 'document.pdf')
            doc_type = doc.get('fileType', 'pdf')
            mime_type = doc.get('mimeType', 'application/pdf')
            doc_weight = doc.get('weight', 1)
            doc_pct = round((doc_weight / total_weight) * 100) if total_weight > 0 else 0
            
            if not doc_content:
                continue
            
            # Add document header with weight
            content.append({
                "type": "text",
                "text": f"\n=== DOCUMENT: {doc_name} (Weight: {doc_pct}%) ==="
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
    ]
}

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
            analysis_prompt = f"""Update this existing analysis with new information from the documents.

Existing Analysis:
{json.dumps(existing_analysis, indent=2)}

Review the new documents and:
1. Update or confirm the investment thesis
2. Add any new signposts or update existing ones
3. Add any new threats or update existing ones
4. Note what has changed
5. Update sources for each point based on all documents analyzed

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

For each pillar, signpost, and threat, include:
- "sources": Array of source documents that support this point, with filename and a brief excerpt
- "confidence": High/Medium/Low for pillars and signposts
- Use the actual document filenames provided

Return the updated analysis as JSON with the same structure, plus a "changes" array describing what's new or different.

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
            
            return jsonify({
                'analysis': analysis,
                'changes': changes,
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
