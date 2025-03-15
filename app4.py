import numpy as np
import pandas as pd
import os 
import sys
import calendar
import datetime as dt
from datetime import date, timedelta, datetime
from shutil import get_terminal_size
pd.set_option('display.width', get_terminal_size()[0]) 
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings("ignore")

import dash
from dash import Dash, html, dcc, Input, Output, ctx, State
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions=True

####### READ IN AND JOIN DATASETS #######
sales = pd.read_csv("Audiobook+sales.csv");
revs = pd.read_csv("Audiobook+reviews.csv");
revs = revs.sort_values(by='Transaction ID')

def JOIN(table1,table2,on,how):
    return pd.merge(table1,table2,on = [on], how = how)

sales_revs = JOIN(sales,revs,'Transaction ID','outer');

#######  CONVERT DATES TO CORRECT FORMAT ########
# - CONVERT TO DATETIME
# - REMOVE NANS AS DON'T ALLOW THE GROUPING BELOW
# - SPLIT DATES INTO YEARS AND MONTHS
date = 'Review Date'

def fix_date(d_in):
    d_in[date] = pd.to_datetime(d_in[date])
    d_out = d_in[d_in[date] == d_in[date]] # Lose NaNs
    d_out['Year'] = d_out[date].astype(str).str[0:4] # Get year and month for drop down
    d_out['Month'] = d_out[date].astype(str).str[5:7] 
    d_out['Day'] = d_out[date].astype(str).str[8:10] 
    d_out = d_out.sort_values(by=date)
    d_out['YM'] = d_out['Year'] + '-' + d_out['Month']
    d_out['YMD'] = d_out['Year'] + '-' + d_out['Month'] + '-' + d_out['Day']
    return d_out
sr = fix_date(sales_revs)

def filter_date(df,freq,Date1,Date2):
    if freq == "Daily":
        f = 'YMD'
    else:
        f = 'YM'
    start_date= df[f].min(); end_date = df[f].max()
    dates = df[f].unique()
    df[f] = pd.to_datetime(df[f])
    date1 = pd.to_datetime(Date1); date2 = pd.to_datetime(Date2)
    if date1 > date2:
        d1 = date1; date1 = date2; date2 = d1
        D1 = Date1; Date1 = Date2; Date2 = D1
    dfd = df[(df[f] >= date1) & (df[f] <= date2)]
    dates = df[f].unique()
    return dfd,date1,date2,f,Date1,Date2
    
f = 'YM'  
start_date= sr[f].min(); end_date = sr[f].max()
dates = sr[f].unique();# print(sr[f].unique())

date1 = start_date; date2 = end_date # INITIALISE THE DATE RANGE OPTION

revs['Book No'] = revs['Audiobook name'].str.split('#').str[1]
revs['Book No'] = revs['Book No'].fillna('99')
books = list(revs['Book No'].unique())
tmp = sorted(books, key=int)
books = ['All'] + tmp
books[-1] = "Other"
books_ind = tmp
books_ind[-1] = "Other"

s1 = 90; s2=100

######### ONTO DASHBOARD ########
width = "15rem"
SIDEBAR_STYLE = {"position": "fixed","top": 0,"left": 0,"bottom": 0,"width": width,
                 "padding": "2rem 1rem","background-color": "black",  "color": "white"}

CONTENT_STYLE = {"margin-left": width, "margin-right": "0rem", "padding": "2rem 1rem",
                 "background-color": "black",  "color": "white"}

sidebar = html.Div([
    html.P("Audiobook Reviews",  style={'font-family':'Poppins','color': 'white',
                                        'font-weight':'bold',
                                         'fontSize': 24,'textAlign':'center'}),
    dbc.Nav(
        [
            dbc.NavLink("Responses", href="/", active="exact",
                        style={"background-color": "rgb(192, 192, 192)",
                               'width': '60%', 'margin-left': '40px', 'margin-right': '0px',
                               'font-weight':'bold','color': 'black','fontSize': 16}),
            html.P(), 
            dbc.NavLink("Ratings", href="/top", active="exact",
                        style={"background-color": "rgb(192, 192, 192)",
                               'width': '60%', 'margin-left': '40px', 'margin-right': '0px',
                               'font-weight':'bold','color': 'black','fontSize': 16}),     
        ],
        vertical=True,
        pills=False, # True hides inactive
    ),
    html.Br(),
    dbc.Row([
        dbc.Row([
            html.Div(children=[
                html.P("Frequency:", style={'textAlign':'center'}), # MIGHT REMOVE
                dcc.Dropdown(
                    ["Daily", "Monthly"],
                    "Monthly", 
                    id='freq-drop',
                    style={'width': '80%', 'margin-left': '20px', 'margin-right': '0px',
                           'font-weight':'bold','color': 'black','fontSize': 14})
            ]
                     ),
            html.Div(children=[
                html.Br(),
                html.P("Start date", style={'textAlign':'center'}),
                dcc.Dropdown(
                    sr[f].unique(),
                    sr[f].min(),
                    id='start-drop',
                    style={'width': '80%', 'margin-left': '20px', 'margin-right': '0px',
                           'font-weight':'bold','color': 'black','fontSize': 14})
            ]
                     ),
            html.Div(children=[
                html.Br(),
                html.P("End date", style={'textAlign':'center'}),
                dcc.Dropdown(
                    sr[f].unique(),
                    sr[f].max(),
                    id='end-drop',
                    style={'width': '80%', 'margin-left': '20px', 'margin-right': '0px',
                       'font-weight':'bold','color': 'black','fontSize': 14})
            ])
        ]),
    
    ]),
    
], style=dict(SIDEBAR_STYLE),
                   )

content = html.Div(id="page-content", style=CONTENT_STYLE)
app.layout = html.Div([dcc.Location(id="url"),sidebar,content])
@app.callback(Output("page-content", "children"),[Input("url", "pathname")]) # MORE PAGES

def render_page_content(pathname):
    if pathname == "/":
        return html.Div([   
            dbc.Row(children =[
                html.Div(id='textarea-output',style={'fontSize': 16,'font-weight':'bold'}),
                html.Br(),
                html.Div([dcc.Graph(id='rate_plot',style={'width': '120vh', 'height': '40vh'})]),# INITIAL ASPECT        
                dcc.Checklist(
                    ['Show running mean'], 
                    id='rate_check',
                    inputStyle={'margin-left': '0px', 'margin-right': '10px'}
                ),
            ]),
 
            dbc.Row(children =[
                dbc.Col([
                    html.Br(),
                    html.Div(id='textarea-output2',style={'fontSize': 16,'font-weight':'bold'}),
                    dcc.Graph(id='donut1'),
                ]),
                dbc.Col([ # NEED TO PUT ON SAME ROW
                    html.Br(),
                    html.Div(id='textarea-output3',style={'fontSize': 16,'font-weight':'bold'}),
                    html.Br(),
                    dcc.RadioItems(
                        books,
                        'All', 
                        id='book',
                        inputStyle={"display":"block",'margin-right': '10px'} 
                    ),
                
                    html.Div(dcc.Graph(id='histo', style = {'autosize':True,'height':400, 'width':420,"margin-right": "0px"})),
                ]),
            ])
        ])
    
    elif pathname == "/top":
        return html.Div([   
            dbc.Row(children=[ # TO GET text-stats in next column (same row)
                dbc.Col([ # LEFT SIDE OF PAGE
                    html.Div(id='text-comp',style={'fontSize': 16,'font-weight':'bold'}),
                    html.Br(),
                       dbc.Row([# FOR DROPDOWNS
                           html.P("Choose two books", style={'textAlign':'left','margin-left': '40px'}),
                           dcc.Dropdown(
                                books_ind,
                                "1", 
                                id='book1',
                                style={'width': '40%', 'margin-left': '40px', 'margin-right': '0px',
                                       'font-weight':'bold','color': 'black','fontSize': 14}
                            ),
                
                            dcc.Dropdown(
                                books_ind,
                                "2",
                                id='book2',
                                style={'width': '40%', 'margin-left': '10px', 'margin-right': '0px',
                                       'font-weight':'bold','color': 'black','fontSize': 14}),
                        html.Br(),
                           ]),
               
                        html.Div(dcc.Graph(id='histo1', style = {'autosize':True,'height':400,
                                                             'width':420,"margin-right": "0px"})),
                    
                    html.Div(id='text-stats1',style={'fontSize': 16,"margin-left": "80px","margin-top": "0px"}), 
                    html.Div(id='text-stats2',style={'fontSize': 16,"margin-left": "80px"}),
                        html.P(),
                    html.Div([
                        "Confidence interval [%]",
                        dcc.Slider(
                            s1,s2,0.1,
                            id='slider',
                            #marks={str(h) : {'label' : str(h), 'style':{'color':'white'}} for h in range(s1,s2,1)},
                            marks = {'90': {'label': '90'}, '95': {'label': '95'}, '99': {'label': '99'},'99.9': {'label': '99.9'}},
                            value=95,
                        ),
                    ], style={'width': "360px",'margin-left': '80px'},
                             ),
                    html.Br(),html.Br(),html.Br(),html.Br(), # DOEN'T CLEAR MORE SPACE AT BOTTOM
                   # ]),
                ],style={'width': '30%', 'margin-left': '0px',}, # COLUMN WIDTH HERE
                    ),
                
                 
                dbc.Col([ # RIGHT SIDE
                    html.Div(dcc.Graph(id='plot2', style = {'autosize':True,'height':470,
                                                             'width':420,"margin-right": "0px"})),
                    
                    html.Br(),html.Br(),
                   
                    html.Div(id='text-method',style={'fontSize': 16,"margin-left": "30px"}),
                    html.Div(id='text-conf',style={'fontSize': 16,"margin-left": "30px"}),
                    html.Div(id='text-result',style={'fontSize': 16,"margin-left": "30px"}),
                    html.Div(id='sv',style={'fontSize': 16,"margin-left": "30px"}),
                    html.Br(),
                    html.Div(id='text-last',style={'fontSize': 16,"margin-left": "30px"}),
                ],style={'width': '50%', 'margin-left': '0px',},
                    ),
                ])
        ])
                    
###################### CALLBACKS ######################
@app.callback( 
    Output('rate_plot', 'figure'),
    Output('textarea-output', 'children'),
    Input('start-drop', 'value'),
    Input('end-drop', 'value'),
    Input('freq-drop', 'value'), 
    Input('rate_check', 'value'),
    )

def rate_plot(Date1,Date2,freq,check): 
    font_size= 14
    srd,date1,date2,f,Date1,Date2  = filter_date(sr,freq,Date1,Date2) 

    tmp = srd.groupby(by=f).count();
    tmp['Date'] = tmp.index
    tmp = tmp.reset_index(); 
    tmp['n'] = tmp.index+1
    x = tmp['Date']; y = tmp.iloc[:, 1];
    total = tmp.iloc[:, 1].cumsum(); 
    rm = total/tmp['n']
    xmin = x.min(); ymax = y.max(); 
    
    rp = px.line(tmp, x=x, y=y, color=None); rp.update_traces(line=dict(width=3))

    if check != None and check != []:
        rp.add_trace(go.Scatter(x=x, y=rm,showlegend=False,line=dict(width=3,color="red")))
    
    rp.update_yaxes(showgrid=False, linecolor = 'white', linewidth = 2,mirror = True,
                    title="Number of rewiews",title_font_color="white",tickfont=dict(color="white"))
    rp.update_xaxes(showgrid=False, linecolor = 'white', linewidth = 2, mirror = True,
                    title_font_color="white",tickfont=dict(color="white"),range=[date1, date2])
    rp.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'},
                     width=870,margin={'l': 40, 'b': 0, 't': 20, 'r': 20}, font=dict(size=font_size),
                     hovermode='closest')

    total = total.iloc[-1]; total_s = f'{total:,}'
    rp.add_annotation(font=dict(color='white', size=0.9*font_size), x=date2 - (date2-date1)/4.8,
                      y=0.98*ymax ,showarrow=False, text=" %s to %s" %(Date1,Date2))

    return rp, "Number of reviews %s to %s (%s total)" %(Date1,Date2,total_s)

@app.callback(
    Output('donut1', 'figure'),
    Output('textarea-output2', 'children'),
    Input('start-drop', 'value'),
    Input('end-drop', 'value'),
    Input('freq-drop', 'value'), 
)

def donut(Date1,Date2,freq):
    rev = fix_date(revs); 
    revd,date1,date2,f,Date1,Date2  = filter_date(rev,freq,Date1,Date2) 
    #print(revd.describe())
    total = len(revd);
    tmp = revd.groupby(by='Audiobook name').count()
    
    tmp['Book name'] = tmp.index;
    tmp['No'] = tmp.iloc[:, 1]
    tmp['Perc'] = 100*tmp['No']/total
 
    tmp['Book No'] = tmp['Book name'].str.split('#').str[1] 
    tmp['Book No'] = tmp['Book No'].fillna(99)
    tmp['Book No'] = tmp['Book No'].astype(int)
    tmp = tmp.sort_values(by = 'Book No'); 
       
    donut = go.Figure(data=[go.Pie(labels=tmp['Book name'], values= tmp['No'], sort=False, hole=.4)])

    donut.update_traces(hoverinfo= 'label+percent', textinfo= 'value', #percent',
                        textfont_size=14, marker=dict(line=dict(color='#000000', width=2))) 
    
    donut.update_layout(width=450, height=450,legend=dict(y=0.5),#moving legend down
                        paper_bgcolor='rgba(0,0,0,0)', # MAKES PLOT TRANS[ARENT
                        font_color="white", margin={"pad": 0, "t": 0,"r": 50,"l": 20,"b": 0})

    total_s = f'{total:,}'
    return donut, "Number of reviews by book [hover for percentages]"

@app.callback(
    Output('histo', 'figure'),
    Output('textarea-output3', 'children'),
    Input('start-drop', 'value'),
    Input('end-drop', 'value'),
    Input('book', 'value'),
   )

def histo(Date1,Date2,book):
    font = 14
    rev = fix_date(revs);
    freq = "Monthly"
    revd,date1,date2,f,Date1,Date2 = filter_date(rev,freq,Date1,Date2)
    book_no = "Audiobook #%s" %(book); # print(book_no); print(book)
    if book == "All":
        revd=revd
    else:
        if book == "Other":
            revd=revd[revd['Audiobook name'] == "Other"]
        else: 
            revd=revd[revd['Audiobook name'] == book_no] 

    def group_dates(df,field):
        tmp = df.groupby(by=f).count();
        tmp['Date']= tmp.index
        tmp[field]=  tmp.iloc[:, 1]
        tmp2 = tmp[['Date','Reviews']]
        return tmp2
    rev_count = group_dates(revd,'Reviews'); #print(rev_count)
    rev_score =  revd.groupby(by=f).sum()
    rev_score['Date']= rev_score.index
    rev_stats = JOIN(rev_count,rev_score,'Date','inner');
    rev_stats['Mean score'] = rev_stats['Rating']/rev_stats['Reviews']
    ave = rev_stats['Mean score'].mean()

    #print(rev_stats,len(rev_stats)) # ALL 24 MONTHS EXCEPT 8 AND 10 - FORCE THIS
    from dateutil.rrule import rrule, MONTHLY
    months = [dt for dt in rrule(MONTHLY, dtstart=date1, until=date2)];
    nbins= len(months) # len(rev_stats) # OKAY, 24 GIVES  FIXED NUMBER OF BINS
    
    bar = px.histogram(x=rev_stats['Date'], y=rev_stats['Mean score'],nbins=nbins)
    
    bar.update_traces(marker_color='rgb(153,153,255)', marker_line_color='black',
                      marker_line_width=2, opacity=1)

    bar.update_xaxes(showgrid=False, linecolor = "white", linewidth = 3,mirror = True, 
                     title="",title_font=dict(size=font),tickfont=dict(size=font),
                     categoryorder="total descending")

    bar.update_yaxes(showgrid=False, linecolor = "white", linewidth = 3,mirror = True, 
                     title="Mean review score [out of 10]",
                     title_font=dict(size=font),tickfont=dict(size=font),
                     title_standoff = 40) # SPACE BETWEEN LABLE AND TICKS

    bar.update_layout(width=350, height=350,paper_bgcolor='rgba(0,0,0,0)', 
                      font_color="white",margin={"pad": 0,'l': 0, 'b': 0, 't': 40, 'r': 0},
                      font=dict(size=16), hovermode='closest',  bargap=0)


    bar.add_annotation(font=dict(color='black', size=0.9*font), x=date1 + (date2-date1)/3,
                       y=11,showarrow=False, text=f"<b>Book %s, mean score = %1.2f</b>" %(book,ave)) 
    
    return bar, "Mean review ratings by book"

@app.callback(
    Output('histo1', 'figure'),
    Output('text-comp', 'children'),
    Output('text-stats1', 'children'),
    Output('text-stats2', 'children'),
    Output('plot2', 'figure'),
    Output('text-method', 'children'),
    Output('text-conf', 'children'),
    Output('text-result', 'children'),
    Output('text-last', 'children'),
    Output('sv', 'children'),
    Input('freq-drop', 'value'),
    Input('start-drop', 'value'),
    Input('end-drop', 'value'),
    Input('book1', 'value'),
    Input('book2', 'value'),
    Input('slider', 'value'),
)
def histo2(freq,Date1,Date2,book1,book2,slide_value):
    font_size=14
    rev = fix_date(revs);
    #freq = "Monthly"
    revd,date1,date2,f,Date1,Date2 = filter_date(rev,freq,Date1,Date2)
    #print(revd)
    B1 = 1; B2 = 2
    book_no1 = "Audiobook #%s" %(book1);book_no2 = "Audiobook #%s" %(book2);
    if book1 == "Other":
        B1=revd[revd['Audiobook name'] == "Other"]
    else: 
        B1=revd[revd['Audiobook name'] == book_no1]

    if book2 == "Other":
        B2=revd[revd['Audiobook name'] == "Other"]
    else: 
        B2=revd[revd['Audiobook name'] == book_no2]

    both = pd.concat([B1, B2], ignore_index=True)
 
    X1 = B1["Rating"]; X2 = B2["Rating"];
    bar1 = go.Figure() # PUTS THE HISTOS SIDE-BY-SIDE
    bar1.add_trace(go.Histogram(x=X1, marker_color='#6666FF', name='%s' %(book_no1)))
    bar1.add_trace(go.Histogram(x=X2,  marker_color='red', name='%s'  %(book_no2)))
    
    ave1 = np.mean(B1['Rating']); SD1 = np.std(B1['Rating'],ddof=1); n1 = len(B1)
    ave2 = np.mean(B2['Rating']); SD2 = np.std(B2['Rating'],ddof=1); n2 = len(B2)
    
    color = 'red'
    bar1.update_traces(marker_line_color='black', marker_line_width=0, opacity=1)

    bar1.update_xaxes(showgrid=False, linecolor = "white", linewidth = 3,mirror = True, 
                      title="Rating",title_font=dict(size=font_size),tickfont=dict(size=font_size),
                      categoryorder="total descending")

    bar1.update_yaxes(showgrid=False, linecolor = "white", linewidth = 3,mirror = True, 
                      title="Total number of reviews", title_font=dict(size=font_size),
                      tickfont=dict(size=font_size),
                      title_standoff = 20) # SPACE BETWEEN LABLE AND TICKS

    bar1.update_layout(width=450, height=350,paper_bgcolor='rgba(0,0,0,0)', 
                       font_color="white",margin={"pad": 0,'l': 0, 'b': 0, 't': 40, 'r': 0},
                       font=dict(size=16), hovermode='closest',  bargap=0,
                       legend=dict(x=0,y=.95,traceorder="normal",
                                   font=dict(family="sans-serif", size=12, color="black"
                                             ),
                       ))

    
    tmp1 = B1.groupby(by=f).count();
    tmp1['Date'] = tmp1.index
    tmp1 = tmp1.reset_index(); 
    tmp1['n'] = tmp1.index+1
    x1 = tmp1['Date']; y1 = tmp1.iloc[:, 1];

    tmp2 = B2.groupby(by=f).count();
    tmp2['Date'] = tmp2.index
    tmp2 = tmp2.reset_index(); 
    tmp2['n'] = tmp2.index+1
    x2 = tmp2['Date']; y2 = tmp2.iloc[:, 1];
    
        
    rp2 = px.line(tmp1, x=x1, y=y1, color=None);
    rp2.update_traces(line=dict(width=3),showlegend = True,name='%s' %(book_no1))
    rp2.add_trace(go.Scatter(x=x2, y=y2, name='%s' %(book_no2),line=dict(width=3,color="red")))
    
    rp2.update_yaxes(showgrid=False, linecolor = 'white', linewidth = 2,mirror = True,
                    title="Number of rewiews", title_font=dict(size=font_size),
                     title_font_color="white",tickfont=dict(color="white",size=font_size))
    rp2.update_xaxes(showgrid=False, linecolor = 'white', linewidth = 2, mirror = True,
                    title_font_color="white", title_font=dict(size=font_size),
                     tickfont=dict(color="white",size=font_size),range=[date1, date2])
    rp2.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'},
                     width=420,margin={'l': 0, 'b': 0, 't': 160, 'r': 20}, font=dict(size=font_size),
                      hovermode='closest',legend=dict(x=0.55,y=.95,traceorder="normal",
                    font=dict(family="sans-serif", size=12, color="white"
                                    ),
                       ))
    
    ##### 2-SAMPLE TESTS #######
    ave= np.abs(ave1-ave2)
    from scipy.stats import norm
    def z_bit(con):
        p = 1-con/100; alpha = 0.5-(p/2) 
        pooled_var = SD1**2/n1 + SD2**2/n2
        pooled_SD = pooled_var**0.5
        pooled_SE = pooled_SD*(1/n1 + 1/n2)**0.5
        Z = norm.ppf(1-p/2,loc=0,scale=1) 
        CI = Z*pooled_SD
        return Z,CI,ave-CI,ave+CI

    def t_bit(con):
        p = 1-con/100;alpha = 0.5-(p/2) 
        npts = 100000
        xi = 0.0; yi = 0; xf = 100; 
        def gamma_f(value):
            x = [xi]; y = [yi]; gamma =0
            for i in range(1,npts):
                dx = (xf-xi)/npts           
                x = i*dx
                y = x**(value-1)*np.exp(-x)
                area =y*dx
                gamma = gamma + area
            return gamma

        var_ratio = SD1**2/SD2**2
                
        if (var_ratio >= 0.5) and (var_ratio < 2):
            var_text = "Variance ratio is %1.2f, so assuming equal variances" %(var_ratio)
            dof = n1 + n2 -2
            pooled_var = ((n1 - 1)*SD1**2 + (n2 - 1)*SD2**2)/(n1 + n2 -2)
            pooled_SD = pooled_var**0.5
            pooled_SE = pooled_SD*(1/n1 + 1/n2)**0.5; 

        else:
            var_text = "Variance ratio is %1.2f, so not assuming equal variances" %(var_ratio)
            A = SD1**2/n1; B =  SD2**2/n2
            dof = (A+B)**2/(A**2/(n1-1) + B**2/(n2-1))

            pooled_SE = (SD1**2/n1 + SD2**2/n2)**0.5;

        n = dof+1
        gamma_num = gamma_f(float(n)/2)
        gamma_den = gamma_f(float(dof)/2)
        stand =  gamma_num/(((np.pi*dof)**0.5)*gamma_den)

        npts = 100000; 
        xf = 5;x = [xi];yy = [yi]; total =0;y_total = 0; total = 0; area = 0; j =0
        dx = (xf-xi)/npts
        for i in range(npts):
            x = i*dx
            y = stand*(1+ x**2/dof)**float(-n/2)
            while(total < alpha):
                y_total = stand*(1+ (j*dx)**2/dof)**float(-n/2)
                area = y_total*dx
                total = total + area; 
                j = j+1
        T = dx*j; 
        CI = T*pooled_SE
        return dof,T,CI,ave-CI,ave+CI,var_text 

    ################################
    out1="%s to %s\n book rating comparisons" %(Date1,Date2)
    out2="Book %s: mean = %1.2f +/- %1.2f (%d ratings)" %(book1,ave1,SD1,n1)
    out3="Book %s: mean = %1.2f +/- %1.2f (%d ratings)" %(book2,ave2,SD2,n2),
    if (n1 > 30) & (n2 > 30):
        out4 = "Both samples are > 30, so using z-statitic"
        Z,CI,C1,C2 = z_bit(slide_value)
        out5 ="At a %1.1f%% confidence level, z-value is %1.3f" %(slide_value,Z)
        var_text = " "
                   
    else:
        out4 = "At least one sample is < 30, so using t-statitic"
        dof,T,CI,C1,C2,var_text = t_bit(slide_value)
        out5 ="At a %1.3f%% confidence level, t-value is %1.3f (%1.1f DoFs)" %(slide_value,T,dof)
               
    out6 = "Giving a mean difference of %1.2f +/- %1.2f (%1.2f to %1.2f)" %(ave,CI,C1,C2)

    if ave2 > ave1:
        high = book2; low = book1
    else:
        high = book1; low = book2
        
    if (C1 > 0) & (C2 > 0):
        out_last = "Book %s rated higher than book %s to %1.2f%% confidence" %(high,low,slide_value)
    else:
        out_last = "Book %s rated the same as Book %s to %1.2f%% confidence" %(high,low,slide_value)
        
    return bar1,out1,out2,out3,rp2,out4,out5,out6,out_last,var_text

### RUN ./Audio_books.py 
if __name__ == '__main__':
    #app.run()  # pythonanywhere
    app.run_server(host = '127.0.0.1', debug=False) # TO RUN LOCALLY, WITHOUT INTERNET
   
