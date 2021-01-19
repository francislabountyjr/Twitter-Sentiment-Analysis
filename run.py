# Imports from 3rd party libraries
import dash
import torch
import re
import torch.nn as nn
import pandas as pd
from transformers import BertTokenizer, BertModel
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from tweepy import AppAuthHandler, API, Cursor

# Imports from this application
from app import app, server
#from pages import index, predictions, insights, process

# Navbar docs: https://dash-bootstrap-components.opensource.faculty.ai/l/components/navbar
navbar = dbc.NavbarSimple(
    brand='Twitter Sentiment Analysis',
    brand_href='/', 
#    sticky='top',
    color='dark', 
    light=False, 
    dark=True
)

# Footer docs:
# dbc.Container, dbc.Row, dbc.Col: https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout
# html.P: https://dash.plot.ly/dash-html-components
# fa (font awesome) : https://fontawesome.com/icons/github-square?style=brands
# mr (margin right) : https://getbootstrap.com/docs/4.3/utilities/spacing/
# className='lead' : https://getbootstrap.com/docs/4.3/content/typography/#lead
footer = dbc.Container(
    dbc.Row(
        dbc.Col(
            html.P(
                [
                    html.Span('Francis LaBounty', className='mr-2'), 
                    html.A(html.I(className='fas fa-envelope-square mr-1'), href='mailto:labounty3d@gmail.com'), 
                    html.A(html.I(className='fab fa-github-square mr-1'), href='https://github.com/francislabountyjr/'), 
                    html.A(html.I(className='fab fa-linkedin mr-1'), href='https://www.linkedin.com/in/francislabounty/'), 
                   # html.A(html.I(className='fab fa-twitter-square mr-1'), href='https://twitter.com/<you>'), 
                ], 
                className='lead'
            )
        )
    )
)

# Layout docs:
# html.Div: https://dash.plot.ly/getting-started
# dcc.Location: https://dash.plot.ly/dash-core-components/location
# dbc.Container: https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout
column1 = dbc.Container(
    [
        dcc.Markdown(
            """
        
            ## BERT_GRU Sentiment Analysis

            This is a BERT-GRU transformer model that was trained on the Sentiment140 1.6 million tweet dataset.

            The goal was to be able to create a sentiment analysis model that generalized well to text that was informal and full of spelling and grammatical errors.

            I was able to achieve an accuracy of 85% on previously unseen to the model tweets from the test set (10% of total data).

            Below are two ways to try out the model. The first text box returns analysis of the given input text.

            The second text box returns analysis of the latest two pages of tweets using the input text as the search query.

            Here is a link to a cheatsheat of Twitter's search operators that I found helpful https://eriksoderstrom.com/twitter/

            """
        ),
        html.Div([
        dcc.Markdown("""
        ###### Input text for sentiment analysis
        """),
        dcc.Input(id='query', value='This is happy sample text', type='text'),
        html.Button(id='submit-button', type='submit', children='Submit'),
        html.Div(id='output_div')
        ]),
        html.Div([
        dcc.Markdown("""
        ###### Twitter search query for sentiment analysis on latest two pages of non-duplicate tweets
        """),
        dcc.Input(id='query_twitter', value='', type='text'),
        html.Button(id='submit-button-twitter', type='submit', children='Submit'),
        html.Div(id='output_div_twitter')
        ]),
    ],
)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False), 
    navbar, 
    column1, 
    html.Hr(), 
    footer
])

#Set up callback
@app.callback([Output('output_div_twitter', 'children'), Output('output_div', 'children')],
                [Input('submit-button-twitter', 'n_clicks'), Input('submit-button', 'n_clicks')],
                [State('query_twitter', 'value'), State('query', 'value')],
                )

#Define update_output function which takes in number of button clicks (not needed) and outputs desired predictions
def update_output(clicks_twitter, clicks, input_value_twitter, input_value):
    if input_value != '' and input_value_twitter == '':
        return [None, predict_sentiment(input_value)]
    elif input_value_twitter != '' and input_value == '':
        return [get_sentiments(input_value_twitter), None]
    elif input_value_twitter != '' and input_value != '':
        return [get_sentiments(input_value_twitter), predict_sentiment(input_value)]
    else:
        return [None, None]

# Run app server: https://dash.plot.ly/getting-started
if __name__ == '__main__':
    #Set random seed and define the current device
    SEED = 1234

    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Set up tweepy API access for Twitter search functionality
    #You fill in with your keys
    consumer_key=""
    consumer_secret=""
    auth = AppAuthHandler(consumer_key, consumer_secret)
    auth_api = API(auth)

    #Load tokenizer and assign special tokens to variables
    tokenizer = BertTokenizer.from_pretrained('assets/')

    init_token = tokenizer.cls_token
    eos_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token
    unk_token = tokenizer.unk_token

    init_token_idx = tokenizer.cls_token_id
    eos_token_idx = tokenizer.sep_token_id
    pad_token_idx = tokenizer.pad_token_id
    unk_token_idx = tokenizer.unk_token_id

    #Define max input token length
    max_input_length = 32

    #Define BERTGRUSentiment class to enable loading the trained pytorch model
    class BERTGRUSentiment(nn.Module):
        def __init__(self,
                    bert,
                    hidden_dim,
                    output_dim,
                    n_layers,
                    bidirectional,
                    dropout):
            
            super().__init__()
            
            self.bert = bert
            
            embedding_dim = bert.config.to_dict()['hidden_size']
            
            self.rnn = nn.GRU(embedding_dim,
                            hidden_dim,
                            num_layers = n_layers,
                            bidirectional = bidirectional,
                            batch_first = True,
                            dropout = 0 if n_layers < 2 else dropout)
            
            self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
            
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, text):
            
            #text = [batch size, sent len]
                    
            with torch.no_grad():
                embedded = self.bert(text)[0]
                    
            #embedded = [batch size, sent len, emb dim]
            
            _, hidden = self.rnn(embedded)
            
            #hidden = [n layers * n directions, batch size, emb dim]
            
            if self.rnn.bidirectional:
                hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
            else:
                hidden = self.dropout(hidden[-1,:,:])
                    
            #hidden = [batch size, hid dim]
            
            output = self.out(hidden)
            
            #output = [batch size, out dim]
            
            return output

    #Load the model and put it on the device
    model = torch.load('assets/bert_model_final')
    model = model.to(device)

    #Define graph_prediction function which takes the positive and negative percentages and outputs the graphs
    def graph_prediction(positive, negative):
        fig = go.Figure()

        #Change bar color and title text depending on what the model predicts
        if positive >= 55:
            fig.add_trace(go.Bar(
                y=[f'Positive: {positive}%', f'Negative: {negative}%'],
                x=[positive, negative],
                name='Percent Sure',
                orientation='h',
                marker=dict(
                    color=['rgba(60, 215, 62, 1.0)', 'rgba(215, 62, 60, 1.0)'],
                    line=dict(color='rgba(0, 0, 0, .8)', width=2)
                )
            ))
            fig.update_layout(barmode='overlay', title='This Text is Positive!')

        elif positive <= 45:
            fig.add_trace(go.Bar(
                y=[f'Positive: {positive}%', f'Negative: {negative}%'],
                x=[positive, negative],
                name='Percent Sure',
                orientation='h',
                marker=dict(
                    color=['rgba(60, 215, 62, 1.0)', 'rgba(215, 62, 60, 1.0)'],
                    line=dict(color='rgba(0, 0, 0, .8)', width=2)
                )
            ))
            fig.update_layout(barmode='overlay', title='This Text is Negative!')

        else:
            fig.add_trace(go.Bar(
                y=[f'Positive: {positive}%', f'Negative: {negative}%'],
                x=[positive, negative],
                name='Percent Sure',
                orientation='h',
                marker=dict(
                    color='rgba(66, 190, 209, 1.0)',
                    line=dict(color='rgba(246, 78, 139, .8)', width=2.5)
                )
            ))
            fig.update_layout(barmode='overlay', title='This Text is Neutral.')
        fig.add_trace(go.Bar(
            y=[f'Positive: {positive}%', f'Negative: {negative}%'],
            x=[100, 100],
            name='Total',
            orientation='h',
            marker=dict(
                color='rgba(58, 71, 80, 0.1)',
                line=dict(color='rgba(58, 71, 80, 0.2)', width=3)
            )
        ))
        return dcc.Graph(figure=fig)

    #Define predict_sentiment function which takes in text, does sentiment analysis, and then returns a graph_prediction
    def predict_sentiment(sentence):
        sentence = clean_text(sentence)
        model.eval()
        tokens = tokenizer.tokenize(sentence)
        tokens = tokens[:max_input_length-2]
        indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(0)
        prediction = torch.sigmoid(model(tensor))
        positive = round(prediction.item() * 100, 2)
        negative = round(100 - positive, 2)
        return graph_prediction(positive, negative)

    #Define clean_text function that takes in text and returns a cleaned up version of it
    def clean_text(text):
        text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
        text = re.sub(r"[^a-zA-z.!?'0-9#@&]", ' ', text)
        text = re.sub('\t', ' ',  text)
        text = re.sub(r" +", ' ', text)
        text = re.sub('&amp', '&',  text)
        return text.strip()

    #define get_sentiments function that takes in a Twitter search query and outputs plotly go figures
    def get_sentiments(query):
        results = [status._json for status in Cursor(auth_api.search, q=query, count=2, tweet_mode='extended', lang='en', ).items()]
        tweets = [clean_text(result['full_text']) for result in results]
        retweets = [result['retweet_count'] for result in results]
        favorites = [result['favorite_count'] for result in results]
        dates = [result['created_at'] for result in results]

        #Get sentiments and populate pandas df
        sentiments = []
        model.eval()
        for tweet in tweets:
            tokens = tokenizer.tokenize(tweet)
            tokens = tokens[:max_input_length-2]
            indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
            tensor = torch.LongTensor(indexed).to(device)
            tensor = tensor.unsqueeze(0)
            prediction = torch.sigmoid(model(tensor))
            if prediction >= 0.55:
                sentiments.append('positive')
            elif prediction <= 0.45:
                sentiments.append('negative')
            else:
                sentiments.append('neutral')
        df = pd.DataFrame({'tweets' : tweets, 'retweets' : retweets, 'favorites' : favorites, 'sentiments' : sentiments, 'dates' : dates}).drop_duplicates(subset=['tweets'])
        sentiment=['Positive', 'Negative', 'Neutral']

        #Make plotly go figures
        fig2 = go.Figure(data=[
            go.Bar(name='Retweets', x=sentiment, y=[df[df['sentiments'] == 'positive'].retweets.sum(), df[df['sentiments'] == 'negative'].retweets.sum(), df[df['sentiments'] == 'neutral'].retweets.sum()], marker=dict(
                color=['rgba(60, 215, 62, 1.0)', 'rgba(215, 62, 60, 1.0)', 'rgba(66, 190, 209, 1.0)'],
            )),
            go.Bar(name='Favorites', x=sentiment, y=[df[df['sentiments'] == 'positive'].favorites.sum(), df[df['sentiments'] == 'negative'].favorites.sum(), df[df['sentiments'] == 'neutral'].favorites.sum()], marker=dict(
                color=['rgba(60, 215, 62, 1.0)', 'rgba(215, 62, 60, 1.0)', 'rgba(66, 190, 209, 1.0)'],
            ))
        ],)

        fig3 = go.Figure(data=[
            go.Bar(name='Positive', x=['Positive'], y=[df[df['sentiments'] == 'positive'].tweets.count()], marker=dict(
                color='rgba(60, 215, 62, 1.0)',
            )),
            go.Bar(name='Negative', x=['Negative'], y=[df[df['sentiments'] == 'negative'].tweets.count()], marker=dict(
                color='rgba(215, 62, 60, 1.0)',
            )),
            go.Bar(name='Neutral', x=['Neutral'], y=[df[df['sentiments'] == 'neutral'].tweets.count()], marker=dict(
                color='rgba(66, 190, 209, 1.0)',
            )),
            ]
            )

        fig4 = go.Figure(data=[go.Table(
            header=dict(values=list(map(lambda x:x.title(), df.columns)),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[df.tweets, df.retweets, df.favorites, df.sentiments, df.dates],
                        fill_color='lavender',
                        align='left'))
        ])

        fig2.update_layout(barmode='group', title=f'Retweets and Favorites per Sentiment (Out of the last {df.tweets.count()} tweets)')
        fig3.update_layout(title='Number of Tweets per Sentiment')
        return dcc.Graph(figure=fig2), dcc.Graph(figure=fig3), dcc.Graph(figure=fig4)

    app.run_server(debug=True)
