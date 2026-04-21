from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Cluster configuration
cluster_labels = {
    0: "Ambassadeurs (Fidèles & VIP)",
    1: "Acheteurs de Gros Volume",
    2: "Clients Actifs Standards",
    3: "Nouveaux / À Risque"
}
cluster_colors = {0: "purple", 1: "teal", 2: "amber", 3: "coral"}
badge_classes = {0: "badge-vip", 1: "badge-volume", 2: "badge-active", 3: "badge-new"}

# Load models
def load_resources():
    clf = joblib.load('models/best_model_churn.pkl')
    reg = joblib.load('models/regression_model_raw.pkl')
    kmeans = joblib.load('models/kmeans_model.pkl')
    sc = joblib.load('models/scaler.pkl')
    pca_mod = joblib.load('models/pca_model.pkl')
    return clf, reg, kmeans, sc, pca_mod, sc.feature_names_in_.tolist(), reg.feature_names_in_.tolist()

clf, reg, kmeans, sc, pca_mod, cols_scaler, cols_reg = load_resources()

# Helper functions
def align_features(df_input, target_columns):
    df_res = df_input.copy()
    for col in df_res.select_dtypes(include=['object']).columns:
        df_res[col] = pd.Categorical(df_res[col]).codes
    for col in target_columns:
        if col not in df_res.columns:
            df_res[col] = 0
    return df_res[target_columns].fillna(0)

def process_dataframe(df_raw):
    # Pipeline
    df_for_scaler = align_features(df_raw, cols_scaler)
    X_scaled = sc.transform(df_for_scaler)
    X_pca = pca_mod.transform(X_scaled)
    X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(10)])

    df_raw['Churn_Pred'] = clf.predict(X_pca_df)
    df_raw['Cluster_ID'] = kmeans.predict(X_pca_df)
    df_raw['Segment'] = df_raw['Cluster_ID'].map(cluster_labels)
    df_raw['Depense_Prevue_DT'] = np.expm1(reg.predict(align_features(df_raw, cols_reg))).round(2)

    return df_raw

def calculate_metrics(df_raw):
    total = len(df_raw)
    n_churn = int(df_raw['Churn_Pred'].sum())
    rev_total = df_raw['Depense_Prevue_DT'].sum()
    churn_rate = (n_churn / total * 100) if total > 0 else 0

    seg_counts = df_raw.groupby(['Cluster_ID', 'Segment']).size().reset_index(name='Count')
    seg_counts['Percentage'] = (seg_counts['Count'] / total * 100).round(1)
    
    return {
        'total': total,
        'n_churn': n_churn,
        'rev_total': rev_total,
        'churn_rate': churn_rate,
        'seg_counts': seg_counts.to_dict('records')
    }

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Retail Intelligence · GI2</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>◈</text></svg>">
    <link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        html, body {
            font-family: 'DM Sans', sans-serif;
            background: #0d0f12;
            background-image:
                radial-gradient(ellipse 60% 40% at 80% 10%, rgba(83,74,183,0.12) 0%, transparent 60%),
                radial-gradient(ellipse 40% 30% at 10% 80%, rgba(29,158,117,0.08) 0%, transparent 50%);
            color: #c9cdd6;
            min-height: 100vh;
        }
        .container { display: flex; min-height: 100vh; }
        .sidebar {
            width: 280px;
            background: #111318;
            border-right: 1px solid rgba(255,255,255,0.06);
            padding: 2rem 1.5rem;
            position: fixed;
            height: 100vh;
            overflow-y: auto;
        }
        .sidebar-logo {
            font-family: 'Syne', sans-serif;
            font-weight: 800;
            font-size: 1.15rem;
            letter-spacing: -0.02em;
            color: #fff;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid rgba(255,255,255,0.07);
            margin-bottom: 1.5rem;
        }
        .sidebar-logo span { color: #7F77DD; }
        .upload-section { margin-bottom: 1.5rem; }
        .upload-section h3 {
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
            color: #c9cdd6;
        }
        .file-input-wrapper { position: relative; overflow: hidden; display: block; }
        .file-input-wrapper input[type=file] { position: absolute; left: -9999px; }
        .file-input-label {
            display: block;
            padding: 2rem;
            background: #161a22;
            border: 1.5px dashed rgba(127,119,221,0.4);
            border-radius: 12px;
            text-align: center;
            cursor: pointer;
            transition: border-color 0.2s;
            font-size: 0.85rem;
            color: #6b7280;
        }
        .file-input-label:hover { border-color: rgba(127,119,221,0.7); }
        .file-name {
            margin-top: 0.5rem;
            font-size: 0.75rem;
            color: #7F77DD;
            word-break: break-all;
        }
        .divider {
            border: none;
            border-top: 1px solid rgba(255,255,255,0.06);
            margin: 1.5rem 0;
        }
        .sidebar-info {
            font-size: 0.78rem;
            color: #4b5563;
            line-height: 1.8;
        }
        .sidebar-info b { color: #6b7280; }
        .main-content {
            margin-left: 280px;
            padding: 2rem 2.5rem 4rem;
            max-width: 1400px;
            width: calc(100% - 280px);
        }
        .page-title {
            font-family: 'Syne', sans-serif;
            font-weight: 800;
            font-size: 2.4rem;
            letter-spacing: -0.03em;
            color: #f0f1f3;
            line-height: 1.1;
            margin-bottom: 0.25rem;
        }
        .page-subtitle {
            font-size: 0.9rem;
            color: #6b7280;
            font-weight: 300;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 2.5rem;
        }
        .empty-state {
            text-align: center;
            padding: 4rem 2rem;
            background: #161a22;
            border: 1.5px dashed rgba(127,119,221,0.25);
            border-radius: 16px;
            margin: 2rem auto;
            max-width: 600px;
        }
        .empty-state-icon {
            font-size: 2.5rem;
            margin-bottom: 1rem;
            color: #534AB7;
        }
        .empty-state-title {
            font-family: 'Syne', sans-serif;
            font-size: 1.1rem;
            font-weight: 600;
            color: #c9cdd6;
            margin-bottom: 0.5rem;
        }
        .empty-state-text {
            font-size: 0.85rem;
            color: #4b5563;
            line-height: 1.7;
        }
        .metric-row {
            display: flex;
            gap: 1rem;
            margin-bottom: 2rem;
            flex-wrap: wrap;
        }
        .metric-card {
            background: #161a22;
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            flex: 1;
            min-width: 160px;
            position: relative;
            overflow: hidden;
        }
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            border-radius: 12px 12px 0 0;
        }
        .metric-card.purple::before { background: linear-gradient(90deg, #534AB7, #7F77DD); }
        .metric-card.teal::before { background: linear-gradient(90deg, #0F6E56, #1D9E75); }
        .metric-card.amber::before { background: linear-gradient(90deg, #854F0B, #EF9F27); }
        .metric-card.coral::before { background: linear-gradient(90deg, #993C1D, #D85A30); }
        .metric-label {
            font-size: 0.7rem;
            font-weight: 500;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #6b7280;
            margin-bottom: 0.4rem;
        }
        .metric-value {
            font-family: 'Syne', sans-serif;
            font-size: 2rem;
            font-weight: 700;
            color: #f0f1f3;
            line-height: 1;
        }
        .section-header {
            font-family: 'Syne', sans-serif;
            font-size: 1rem;
            font-weight: 600;
            color: #c9cdd6;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid rgba(255,255,255,0.07);
            margin: 2rem 0 1rem;
        }
        .segment-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        .segment-card { text-align: center; }
        .segment-count {
            font-size: 0.75rem;
            color: #4b5563;
            margin-top: 0.25rem;
        }
        .data-table {
            width: 100%;
            background: #161a22;
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 2rem;
        }
        .table-wrapper {
            max-height: 400px;
            overflow-y: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        thead {
            position: sticky;
            top: 0;
            background: #1a1e27;
            z-index: 10;
        }
        th, td {
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            font-size: 0.85rem;
        }
        th {
            font-weight: 600;
            color: #c9cdd6;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        td { color: #9ca3af; }
        tbody tr:hover { background: rgba(127,119,221,0.05); }
        .insight-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
            margin: 1.5rem 0;
        }
        .insight-card {
            background: #161a22;
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 12px;
            padding: 1.25rem;
        }
        .insight-card-title {
            font-family: 'Syne', sans-serif;
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        .insight-card-body {
            font-size: 0.82rem;
            color: #6b7280;
            line-height: 1.6;
        }
        .insight-card.purple .insight-card-title { color: #AFA9EC; }
        .insight-card.teal .insight-card-title { color: #5DCAA5; }
        .insight-card.amber .insight-card-title { color: #FAC775; }
        .insight-card.coral .insight-card-title { color: #F0997B; }
        .loading {
            text-align: center;
            padding: 2rem;
            color: #7F77DD;
        }
        .error {
            background: rgba(226,75,74,0.15);
            color: #F09595;
            border: 1px solid rgba(226,75,74,0.3);
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .hidden { display: none; }
        @media (max-width: 768px) {
            .sidebar {
                width: 100%;
                position: relative;
                height: auto;
                border-right: none;
                border-bottom: 1px solid rgba(255,255,255,0.06);
            }
            .main-content {
                margin-left: 0;
                width: 100%;
            }
            .container { flex-direction: column; }
            .insight-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <aside class="sidebar">
            <div class="sidebar-logo">◈ Retail<span>Intelligence</span></div>
            <div class="upload-section">
                <h3>Import your data</h3>
                <div class="file-input-wrapper">
                    <input type="file" id="csvFile" accept=".csv">
                    <label for="csvFile" class="file-input-label">
                        📁 Choose CSV file
                        <div class="file-name" id="fileName"></div>
                    </label>
                </div>
            </div>
            <hr class="divider">
            <div class="sidebar-info">
                <b>Models loaded</b><br>
                · Churn classifier<br>
                · Revenue regressor<br>
                · K-Means clustering<br>
                · PCA (10 components)
            </div>
            <hr class="divider">
            <div class="sidebar-info" style="font-size:0.72rem; color:#374151;">
                Retail Intelligence System
            </div>
        </aside>
        <main class="main-content">
            <div class="page-title">Customer 360° Intelligence</div>
            <div class="page-subtitle">Segmentation · Churn · Revenue Forecast</div>
            <div id="emptyState" class="empty-state">
                <div class="empty-state-icon">◈</div>
                <div class="empty-state-title">Drop your CSV in the sidebar</div>
                <div class="empty-state-text">
                    Upload a customer dataset to run<br>segmentation, churn prediction &amp; revenue forecast.
                </div>
            </div>
            <div id="loadingState" class="loading hidden">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">⏳</div>
                <div>Processing your data...</div>
            </div>
            <div id="errorState" class="error hidden"></div>
            <div id="results" class="hidden">
                <div id="metricsRow" class="metric-row"></div>
                <div class="section-header">Segment Distribution</div>
                <div id="segmentGrid" class="segment-grid"></div>
                <div class="section-header">Prediction Results</div>
                <div class="data-table">
                    <div class="table-wrapper">
                        <table>
                            <thead id="tableHead"></thead>
                            <tbody id="tableBody"></tbody>
                        </table>
                    </div>
                </div>
                <div class="section-header">Cluster Analysis — Feature Means</div>
                <div class="data-table">
                    <div class="table-wrapper">
                        <table>
                            <thead id="clusterHead"></thead>
                            <tbody id="clusterBody"></tbody>
                        </table>
                    </div>
                </div>
                <div class="section-header">Interpretation Guide</div>
                <div class="insight-grid">
                    <div class="insight-card purple">
                        <div class="insight-card-title">◈ Ambassadeurs VIP</div>
                        <div class="insight-card-body">Highest <b>Depense_Prevue_DT</b> and frequency.
                        Prioritize retention offers, loyalty rewards, and exclusive previews.</div>
                    </div>
                    <div class="insight-card teal">
                        <div class="insight-card-title">◈ Acheteurs Volume</div>
                        <div class="insight-card-body">Peak <b>AvgQuantityPerTransaction</b>.
                        Target with bulk discounts and B2B partnership programs.</div>
                    </div>
                    <div class="insight-card amber">
                        <div class="insight-card-title">◈ Clients Actifs</div>
                        <div class="insight-card-body">Solid regulars with moderate spend.
                        Upsell opportunities and cross-category promotions work well.</div>
                    </div>
                    <div class="insight-card coral">
                        <div class="insight-card-title">◈ Nouveaux / À Risque</div>
                        <div class="insight-card-body">Low tenure + high <b>Churn_Pred</b>.
                        Activate onboarding sequences and early engagement campaigns immediately.</div>
                    </div>
                </div>
            </div>
        </main>
    </div>
    <script>
        const fileInput = document.getElementById('csvFile');
        const fileName = document.getElementById('fileName');
        const emptyState = document.getElementById('emptyState');
        const loadingState = document.getElementById('loadingState');
        const errorState = document.getElementById('errorState');
        const results = document.getElementById('results');

        fileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            fileName.textContent = file.name;
            emptyState.classList.add('hidden');
            errorState.classList.add('hidden');
            results.classList.add('hidden');
            loadingState.classList.remove('hidden');
            const formData = new FormData();
            formData.append('file', file);
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.error || 'Upload failed');
                }
                loadingState.classList.add('hidden');
                renderResults(data);
                results.classList.remove('hidden');
            } catch (error) {
                loadingState.classList.add('hidden');
                errorState.textContent = `Error: ${error.message}`;
                errorState.classList.remove('hidden');
            }
        });

        function renderResults(data) {
            const { metrics, data: tableData, cluster_analysis, cluster_colors } = data;
            const metricsHTML = `
                <div class="metric-card purple">
                    <div class="metric-label">Total Clients</div>
                    <div class="metric-value">${metrics.total.toLocaleString()}</div>
                </div>
                <div class="metric-card coral">
                    <div class="metric-label">At-Risk (Churn)</div>
                    <div class="metric-value">${metrics.n_churn.toLocaleString()}</div>
                </div>
                <div class="metric-card teal">
                    <div class="metric-label">Forecast Revenue</div>
                    <div class="metric-value" style="font-size:1.4rem;">${metrics.rev_total.toLocaleString('en-US', {maximumFractionDigits: 0})} DT</div>
                </div>
                <div class="metric-card amber">
                    <div class="metric-label">Churn Rate</div>
                    <div class="metric-value">${metrics.churn_rate.toFixed(1)}%</div>
                </div>
            `;
            document.getElementById('metricsRow').innerHTML = metricsHTML;
            let segmentHTML = '';
            metrics.seg_counts.forEach(seg => {
                const color = cluster_colors[seg.Cluster_ID] || 'purple';
                segmentHTML += `
                    <div class="metric-card ${color}">
                        <div class="metric-label" style="margin-bottom:0.6rem;">${seg.Segment}</div>
                        <div class="metric-value">${seg.Count.toLocaleString()}</div>
                        <div class="segment-count">${seg.Percentage}% of base</div>
                    </div>
                `;
            });
            document.getElementById('segmentGrid').innerHTML = segmentHTML;
            if (tableData.length > 0) {
                const columns = Object.keys(tableData[0]);
                const headHTML = '<tr>' + columns.map(col => `<th>${col}</th>`).join('') + '</tr>';
                const bodyHTML = tableData.map(row => 
                    '<tr>' + columns.map(col => `<td>${row[col]}</td>`).join('') + '</tr>'
                ).join('');
                document.getElementById('tableHead').innerHTML = headHTML;
                document.getElementById('tableBody').innerHTML = bodyHTML;
            }
            const clusterCols = Object.keys(Object.values(cluster_analysis)[0] || {});
            if (clusterCols.length > 0) {
                const clusterHeadHTML = '<tr><th>Cluster</th>' + clusterCols.map(col => `<th>${col}</th>`).join('') + '</tr>';
                const clusterBodyHTML = Object.entries(cluster_analysis).map(([cluster, values]) =>
                    '<tr><td><b>' + cluster + '</b></td>' + clusterCols.map(col => `<td>${values[col]}</td>`).join('') + '</tr>'
                ).join('');
                document.getElementById('clusterHead').innerHTML = clusterHeadHTML;
                document.getElementById('clusterBody').innerHTML = clusterBodyHTML;
            }
        }
    </script>
</body>
</html>'''

@app.route('/')
def index():
    return HTML_TEMPLATE

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.endswith('.csv'):
        try:
            df_raw = pd.read_csv(file)
            df_processed = process_dataframe(df_raw)
            metrics = calculate_metrics(df_processed)
            
            display_cols = ['Segment', 'Churn_Pred', 'Depense_Prevue_DT']
            other_cols = [c for c in df_processed.columns if c not in display_cols + ['Cluster_ID']]
            df_display = df_processed[display_cols + other_cols].copy()
            
            df_display['Churn_Pred'] = df_display['Churn_Pred'].map({1: '⚠ Churn', 0: '✓ Retained'})
            df_display['Depense_Prevue_DT'] = df_display['Depense_Prevue_DT'].apply(lambda x: f"{x:,.2f} DT")
            df_display.rename(columns={
                'Segment': 'Segment',
                'Churn_Pred': 'Churn Status',
                'Depense_Prevue_DT': 'Forecast Revenue'
            }, inplace=True)
            
            df_analyse = df_processed.groupby('Cluster_ID').mean(numeric_only=True).round(2)
            df_analyse.index = df_analyse.index.map(cluster_labels)
            
            return jsonify({
                'success': True,
                'metrics': metrics,
                'data': df_display.to_dict('records'),
                'cluster_analysis': df_analyse.to_dict('index'),
                'cluster_colors': cluster_colors,
                'cluster_labels': cluster_labels
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type. Please upload a CSV file.'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)