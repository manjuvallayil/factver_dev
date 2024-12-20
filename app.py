from flask import Flask, request, redirect, url_for, render_template, render_template_string

## use jinja2 template tp handle data handling with request and html
"""
{%...%} for statements 
{{...}} expressions to print output
{#...#} for comments
"""

app=Flask(__name__)

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/submit',methods=['POST','GET'])
def submit():
    if request.method=='POST':
        claim_id=str(request.form['claim_id'])
        claim_text=str(request.form['claim_text'])
        message= f'The claim to verify is with the id {claim_id} and with the text {claim_text}'    

        # Read HTML content of the plot
    with open("/home/qsh5523/Documents/factver_dev/SOI Visualization for Claim_50.html", "r") as f:
        plot_html = f.read()

    # return render_template('results.html', res = message, plot = plot_html)
    #return message
    return render_template_string("""
        <html>
        <head><title>Results</title></head>
        <body>
            <h1>Results</h1>
            <p>{{ res }}</p>
            <div>{{ plot|safe }}</div>
        </body>
        </html>
    """, res=message, plot=plot_html)

if __name__=='__main__':
    app.run(debug=True)
