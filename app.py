import os
import pandas as pd
from flask import Flask, render_template, request
from model import run_web_usage_mining, crawl_and_mine_url

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# --- ADD THESE TWO LINES ---
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/', methods=['GET', 'POST'])
def home():
    rules = pd.DataFrame()
    sil = 0.0
    data_processed = False

    if request.method == 'POST':
        user_input = request.form.get('user_input')
        file = request.files.get('log_file')
        try:
            if user_input and user_input.startswith('http'):
                rules, sil = crawl_and_mine_url(user_input)
                data_processed = True
            elif file and file.filename != '':
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                rules, sil = run_web_usage_mining(filepath)
                data_processed = True

            if data_processed:
                # --- DYNAMIC SUMMARY LOGIC ---
                # 1. Behavior Clarity Logic
                if sil > 0.4:
                    group_title = "Highly Organized"
                    group_text = "Your users fall into strictly separate categories. This is common for sites with very specific, linear user goals."
                elif sil > 0.15:
                    group_title = "Moderately Defined"
                    group_text = "There are clear groups, but their behaviors overlap significantly. This is common where users browse varied categories."
                else:
                    group_title = "Exploring Randomly"
                    group_text = "Users are drifting between different types of behaviors. They aren't following a single, rigid path."

                # 2. Prediction Intelligence Logic
                if not rules.empty:
                    top = rules.iloc[0]
                    pattern_text = f"Strong trend found: Users visiting '{top['antecedents']}' are {int(top['confidence']*100)}% likely to visit '{top['consequents']}'."
                else:
                    pattern_text = "Visitors are behaving uniquely. No single dominant path (like Homepage -> Cart) was discovered during this run."

                return render_template('index.html',
                                       silhouette=sil,
                                       group_title=group_title,
                                       group_text=group_text,
                                       pattern_text=pattern_text,
                                       rules=rules.to_html(classes='table table-hover', index=False),
                                       data_processed=True)
        except Exception as e:
            return f"Mining Error: {str(e)}"

    return render_template('index.html', data_processed=False)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)