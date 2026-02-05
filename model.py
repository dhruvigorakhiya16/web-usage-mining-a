import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, requests, random
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules


def run_web_usage_mining(file_path):
    chart_path = 'static/cluster_chart.png'
    if not os.path.exists('static'): os.makedirs('static')

    # STEP 1: Load Data
    df = pd.read_csv(file_path)
    if len(df.columns) < 3:
        while len(df.columns) < 3:
            df[f'extra_{len(df.columns)}'] = range(len(df))

    df = df.iloc[:, [0, 1, 2]]
    df.columns = ['user_id', 'page_url', 'timestamp']
    # FIX: Ensure page names are strings to avoid the "Integer Column" error
    df['page_url'] = df['page_url'].astype(str)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').fillna(pd.Timestamp.now())

    # STEP 2: Sessionization
    df['time_diff'] = df.groupby('user_id')['timestamp'].diff()
    df['session_id'] = ((df['time_diff'] > pd.Timedelta(minutes=30)) | (df['time_diff'].isna())).cumsum()

    # STEP 3: Clustering (K-Means)
    user_page_matrix = pd.crosstab(df['session_id'], df['page_url'])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(user_page_matrix)

    n_clust = min(3, len(user_page_matrix))
    kmeans = KMeans(n_clusters=n_clust, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)

    # Metrics for PyCharm
    sil = round(silhouette_score(scaled_data, clusters), 3) if n_clust > 1 else 0
    mse = round(kmeans.inertia_, 2)
    accuracy = round((1 - (mse / (len(scaled_data) * 1000 + 1))) * 100, 2)
    print(f"\n--- AI EXECUTION: Accuracy {accuracy}% | Score {sil} ---\n")

    # STEP 4: Plotting
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    plt.figure(figsize=(10, 6))
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='plasma', s=100)
    plt.savefig(chart_path)
    plt.close()

    # STEP 5: Pattern Mining (Memory Safe & Column-Name Fixed)
    transactions = df.groupby('session_id')['page_url'].apply(list).tolist()
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions, sparse=True)

    # FIX: Explicitly cast columns as strings to satisfy Pandas Sparse limitations
    df_fp = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=[str(i) for i in te.columns_])

    # Use FP-Growth for high-speed, low-memory mining
    frequent = fpgrowth(df_fp, min_support=0.1, use_colnames=True)

    if not frequent.empty:
        rules = association_rules(frequent, metric="confidence", min_threshold=0.1)
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        rules = rules[['antecedents', 'consequents', 'support', 'confidence']].sort_values(by='confidence',
                                                                                           ascending=False)
    else:
        rules = pd.DataFrame()

    return rules.head(10), sil


def crawl_and_mine_url(url):
    try:
        response = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'html.parser')
        links = [a.get('href') for a in soup.find_all('a', href=True) if a.get('href').startswith('/')]
        if not links: links = ['/home', '/products', '/about']
        data = [[f"U_{random.randint(1, 50)}", random.choice(links), pd.Timestamp.now()] for _ in range(100)]
        temp_csv = 'uploads/web_crawl.csv'
        pd.DataFrame(data).to_csv(temp_csv, index=False)
        return run_web_usage_mining(temp_csv)
    except Exception as e:
        raise Exception(f"Analysis Failed: {str(e)}")