import sys
sys.path.append("..")
import os
import pathlib
import numpy as np
import datetime as dt
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import dash 
import json
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from scipy.stats import rayleigh
# from .db.api import get_wind_data, get_wind_data_by_id
import plotly.graph_objects as go
import pandas as pd
import glob

# from transform_model import PATH_DIR
import plotly.express as px 

PATH_DIR = "."
app_color = {"graph_bg": "#082255", "graph_line": "#007ACE"}
with open(f'{PATH_DIR}/assets/training_result.json', 'r', encoding ='utf8') as json_file:
    training_result = json.load(json_file)

list_of_files_results = glob.glob(f'{PATH_DIR}/results/*.csv') 
latest_file = max(list_of_files_results, key=os.path.getctime).split("\\")[-1]
print(latest_file)
df = pd.read_csv(f"{PATH_DIR}/results/{latest_file}")
number_of_bets = len(df)
accurate_prediction = len(df[df["actual_outcome"] == df["bet_home_team"]])
accuracy = accurate_prediction/number_of_bets*100
#print(pd.to_datetime(pd.to_datetime(df["Start Time"]).sort_values(ascending=True).values[0]).strftime("%Y-%m-%d"))
start_date = pd.to_datetime(pd.to_datetime(df["Start Time"]).sort_values(ascending=True).values[0]).strftime("%Y-%m-%d")
end_date = pd.to_datetime(pd.to_datetime(df["Start Time"]).sort_values(ascending=False).values[0]).strftime("%Y-%m-%d")
bet_return= df["Won"].sum()/number_of_bets * 100

df = df.drop(["actual_outcome"], axis=1)

list_of_files_df = glob.glob(f'{PATH_DIR}/results/*.csv') 


df_past = pd.DataFrame({})
for i in list_of_files_df:
    df_past = pd.concat([df_past,pd.read_csv(i, parse_dates=["Start Time"], dayfirst=True)], axis=0, ignore_index=True)
df_past = df_past.drop_duplicates()
cumulative_start_date = pd.to_datetime(pd.to_datetime(df_past["Start Time"]).sort_values(ascending=True).values[0]).strftime("%Y-%m-%d")
cumulative_end_date = pd.to_datetime(pd.to_datetime(df_past["Start Time"]).sort_values(ascending=False).values[0]).strftime("%Y-%m-%d")
cumulative_number_of_bets = len(df_past)
cumulative_accurate_prediction = len(df_past[df_past["actual_outcome"] == df_past["bet_home_team"]])
cumulative_accuracy = cumulative_accurate_prediction/cumulative_number_of_bets*100
cumulative_bet_return= df_past["Won"].sum()/cumulative_number_of_bets * 100
df["Won"] = round(df["Won"],2)

df_past["Won"] = round(df_past["Won"],2)


def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns], style={'color':"white", "font-size":"14px"})

        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ], style={'color':"white","font-size":"12px","padding":"40px"})

    ]) 

tbale = generate_table(df)



fig = px.line(y=[training_result["train_accuracy_list"],training_result["test_accuracy_list"],training_result["high_test_accuracy_list"] ])

newnames = {'wide_variable_0':'Training', 'wide_variable_1': 'Testing','wide_variable_2':"High Certainty Testing"}
fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                      legendgroup = newnames[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])
                                     ))

fig.update_layout(
    xaxis_title="Epochs",
    yaxis_title="Accuracy(%)",
    legend_title="legend",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="White",

    ),
   #plot_bgcolor=app_color["graph_bg"],
    paper_bgcolor=app_color["graph_bg"],
    height=400,
    xaxis={
    "range": [0, 10000],
    "showline": True,
    "zeroline": False,
    "fixedrange": True,
#    "tickvals": [0, 50, 100, 150, 200],
#       "ticktext": ["200", "150", "100", "50", "0"],
#     "title": "Time Elapsed (sec)",
},
    yaxis={
        # "range": [
        #     min(0, min(df["Speed"])),
        #     max(45, max(df["Speed"]) + max(df["SpeedError"])),
        # ],
        "showgrid": True,
        "showline": True,
        "fixedrange": True,
        "zeroline": False,
        #   "gridcolor": app_color["graph_line"],
        #  "nticks": max(6, round(df["Speed"].iloc[-1] / 10)),
    },
)
         
fig.update_layout(legend=dict(
    orientation="v",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))


fig2 = go.Figure(data=go.Scatterpolar(
  r=[i["total_confident_accuracy"] for i in list(training_result["data"].values())],
  theta=list(str(round(float(i)*100,0))+"% ("+str(j) +" trades)" for i,j in zip(training_result["data"].keys(),[k["total_confident_guessed"] for k in list(training_result["data"].values())])),
  fill='toself',
   fillcolor="blue"
))

fig2.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True
    ),
  ),
  showlegend=False,
plot_bgcolor=app_color["graph_bg"],
paper_bgcolor=app_color["graph_bg"],
    font=dict(
        family="Courier New, monospace",
        size=12,
        color="White",

    ),
)

def create_dashboard(server):
    GRAPH_INTERVAL = os.environ.get("GRAPH_INTERVAL", 5000)

    app = dash.Dash(
        __name__,
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
        server = server,
        routes_pathname_prefix='/training/',
    )
    app.title = "Training Dashboard"
    app.color = "#111111"
    app.layout = html.Div(  
                          [
            # header
            html.Div(
                [
                    html.Div(
                        [
                            html.H4("Training and Testing Diagnostics", className="app__header__title"),
                            html.P(
                                 "",
                                className="app__header__title--grey",
                            ),
                        ],
                        className="app__header__desc",
                    ),

                html.Div(
                    [
                        html.A(
                            html.Button("HOME", className="link-button"),
                            href="/",
                        ),
                        html.A(
                            html.Button("REFRESH", className="link-button"),
                            href="/my-link",
                        ),
                        # html.A(
                        #     html.Img(
                        #         # src=app.get_asset_url("#"),
                        #         className="app__menu__img",
                        #     ),
                        #     href="#",
                        # ),
                    ],
                    className="app__header__logo",
                ),
            ],
            className="app__header",
        ),
            html.Div(
                [
                    # wind speed
                    html.Div(
                        [
                            html.Div(
                                [html.H6("Latest Model Training vs Testing Accuracy(%)", className="graph__title")]
                            ),
                            dcc.Graph(
                                id="wind-speed",
                                figure=fig,
                                                                        
                            ),
                        ],
                        className="six columns wind__speed__container",
                    ),
                    html.Div(
                        [
                            # histogram
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.H6(
                                                "Latest Model Confidence(%) Hit",
                                                className="graph__title",
                                            )
                                        ]
                                    ),
                                    # html.Div(
                                    #     [
                                    #         dcc.Slider(
                                    #             id="bin-slider",
                                    #             min=1,
                                    #             max=60,
                                    #             step=1,
                                    #             value=20,
                                    #             updatemode="drag",
                                    #             marks={
                                    #                 20: {"label": "20"},
                                    #                 40: {"label": "40"},
                                    #                 60: {"label": "60"},
                                    #             },
                                    #         )
                                    #     ],
                                    #     className="slider",
                                    # ),
                                    # html.Div(
                                    #     [
                                    #         dcc.Checklist(
                                    #             id="bin-auto",
                                    #             options=[
                                    #                 {"label": "Auto", "value": "Auto"}
                                    #             ],
                                    #             value=["Auto"],
                                    #             inputClassName="auto__checkbox",
                                    #             labelClassName="auto__label",
                                    #         ),
                                    #         html.P(
                                    #             "# of Bins: Auto",
                                    #             id="bin-size",
                                    #             className="auto__p",
                                    #         ),
                                    #     ],
                                    #     className="auto__container",
                                    # ),
                                    dcc.Graph(
                                        id="wind-histogram",
                                        figure=fig2
                                    ),
                                ],
                                className="graph__container first",
                            ),
                            # wind direction
                            # html.Div(
                            #     [
                            #         html.Div(
                            #             [
                            #                 html.H6(
                            #                     "WIND DIRECTION", className="graph__title"
                            #                 )
                            #             ]
                            #         ),
                            #         dcc.Graph(
                            #             id="wind-direction",
                            #             figure=fig2,
                            #         ),
                            #     ],
                            #     className="graph__container second",
                            # ),
                        ],
                        className="six columns histogram__direction",
                    ),
                ],
                className="app__content",
            ),

            html.Div(
                [
                    # wind speed
                    html.Div([
                        html.H6(f"Backtested from {start_date} to {end_date}", className="graph__title"),
                        dbc.Row(
                        [dbc.Col(
                            dbc.Card(
                                    [
                                        dbc.CardHeader(""),
                    
                                        dbc.CardBody(
                                            [
                                                html.H4("Accuracy", className="card-title"),
                                                html.P(f" {round(accuracy,2)} %", className="card-text"),
                                            ]
                                        ),
                                        dbc.CardFooter(""),
                                    ],
                                  #  style={"color":"white","width":"30%",
                                   # "background-color":f"{ app_color['graph_bg']}"
                                 #   },
                                ), 
                                #style={"width":"30%"}
                                ),
                        dbc.Col(
                        dbc.Card(
                                    [
                                        dbc.CardHeader("Advised"),
                                        dbc.CardBody(
                                            [
                                                html.H4("Trades", className="card-title"),
                                                html.P(number_of_bets, className="card-text"),
                                            ]
                                        ),
                                        dbc.CardFooter(""),
                                    ],
                                   # style={"color":"white",
                                   # "background-color":f"{ app_color['graph_bg']}"
                                   # },
                                ), 
                                #style={"width":"10%"}
                                ),
                        dbc.Col(
                            dbc.Card(
                                    [
                                        dbc.CardHeader("Percentage"),
                                        dbc.CardBody(
                                            [
                                                html.H4("Return", className="card-title"),
                                                html.P(round(bet_return,2), className="card-text"),
                                            ]
                                        ),
                                        dbc.CardFooter(""),
                                    ],
                                ), 
                                 )
                                ],
                            )],
                        className="six columns",
                        style={
                                    "background-color":"rgb(95, 106, 244)",
                                    }
                    ),

                  html.Div([
                        html.H6(f"Backtested from {cumulative_start_date} to {cumulative_end_date}", className="graph__title"),
                        dbc.Row(
                        [dbc.Col(
                            dbc.Card(
                                    [
                                        dbc.CardHeader(""),
                    
                                        dbc.CardBody(
                                            [
                                                html.H4("Accuracy", className="card-title"),
                                                html.P(f" {round(cumulative_accuracy,2)} %", className="card-text"),
                                            ]
                                        ),
                                        dbc.CardFooter(""),
                                    ],
                                  #  style={"color":"white","width":"30%",
                                   # "background-color":f"{ app_color['graph_bg']}"
                                 #   },
                                ), 
                                #style={"width":"30%"}
                                ),
                        dbc.Col(
                        dbc.Card(
                                    [
                                        dbc.CardHeader("Advised"),
                                        dbc.CardBody(
                                            [
                                                html.H4("Trades", className="card-title"),
                                                html.P(cumulative_number_of_bets, className="card-text"),
                                            ]
                                        ),
                                        dbc.CardFooter(""),
                                    ],
                                   # style={"color":"white",
                                   # "background-color":f"{ app_color['graph_bg']}"
                                   # },
                                ), 
                                #style={"width":"10%"}
                                ),
                        dbc.Col(
                            dbc.Card(
                                    [
                                        dbc.CardHeader("Percentage"),
                                        dbc.CardBody(
                                            [
                                                html.H4("Return", className="card-title"),
                                                html.P(round(cumulative_bet_return,2), className="card-text"),
                                            ]
                                        ),
                                        dbc.CardFooter(""),
                                    ],
                                ), 
                                 )
                                ],
                            )],
                        className="six columns histogram__direction",
                        style={
                                    "background-color": "rgb(0, 199, 148)",  #{ app_color['graph_bg']}"
                                    }
                    )
                    # html.Div(
                    # className="one-third column histogram__direction")
                    # html.Table(
                    #     [

                    #     ],
                    #     className="styled-table",
                    # ),
                ],
                className="app__content",
            ),
            html.Div(
                [
                    # wind speed
                    html.Div(
                        [
                            html.Div(
                                [html.H6("Past 7 Days Model Selection", className="graph__title")]
                            ), html.Div
                                ( tbale, style={"padding-left":"60px" , "padding-right":"60px"},
                                
                                ) ,                                                                      
                            
                        ], style={"background-color":f"{ app_color['graph_bg']}","margin-right":"23px"},
                        className="three-thirds columns",
                    ),
                ],
                className="app__content",
            ),
        
        ],
        className="app__container",
    )

    #wind_speed(app)
  #  wind_direction(app)
   # gen_wind(app)
   # deselect(app)
   # show_bins(app)
    
    return app.server

def show_bins(app):
    @app.callback(
    Output("bin-size", "children"),
    [Input("bin-auto", "value")],
    [State("bin-slider", "value")],
    )
    def show_num_bins(autoValue, slider_value):
        """ Display the number of bins. """

        if "Auto" in autoValue:
            return "# of Bins: Auto"
        return "# of Bins: " + str(int(slider_value))

def deselect(app):
    @app.callback(
        Output("bin-auto", "value"),
        [Input("bin-slider", "value")],
        [State("wind-speed", "figure")],
    )
    def deselect_auto(slider_value, wind_speed_figure):
        """ Toggle the auto checkbox. """

        # prevent update if graph has no data
        if "data" not in wind_speed_figure:
            raise PreventUpdate
        if not len(wind_speed_figure["data"]):
            raise PreventUpdate

        if wind_speed_figure is not None and len(wind_speed_figure["data"][0]["y"]) > 5:
            return [""]
        return ["Auto"] 

def gen_wind(app):
    @app.callback(
        Output("wind-histogram", "figure"),
        [Input("wind-speed-update", "n_intervals")],
        [
            State("wind-speed", "figure"),
            State("bin-slider", "value"),
            State("bin-auto", "value"),
        ],
    )
    def gen_wind_histogram(interval, wind_speed_figure, slider_value, auto_state):
        """
        Genererate wind histogram graph.

        :params interval: upadte the graph based on an interval
        :params wind_speed_figure: current wind speed graph
        :params slider_value: current slider value
        :params auto_state: current auto state
        """

        wind_val = []

        try:
            # Check to see whether wind-speed has been plotted yet
            if wind_speed_figure is not None:
                wind_val = wind_speed_figure["data"][0]["y"]
            if "Auto" in auto_state:
                bin_val = np.histogram(
                    wind_val,
                    bins=range(int(round(min(wind_val))), int(round(max(wind_val)))),
                )
            else:
                bin_val = np.histogram(wind_val, bins=slider_value)
        except Exception as error:
            raise PreventUpdate

        avg_val = float(sum(wind_val)) / len(wind_val)
        median_val = np.median(wind_val)

        pdf_fitted = rayleigh.pdf(
            bin_val[1], loc=(avg_val) * 0.55, scale=(bin_val[1][-1] - bin_val[1][0]) / 3
        )

        y_val = (pdf_fitted * max(bin_val[0]) * 20,)
        y_val_max = max(y_val[0])
        bin_val_max = max(bin_val[0])

        trace = dict(
            type="bar",
            x=bin_val[1],
            y=bin_val[0],
            marker={"color": app_color["graph_line"]},
            showlegend=False,
            hoverinfo="x+y",
        )

        traces_scatter = [
            {"line_dash": "dash", "line_color": "#2E5266", "name": "Average"},
            {"line_dash": "dot", "line_color": "#BD9391", "name": "Median"},
        ]

        scatter_data = [
            dict(
                type="scatter",
                x=[bin_val[int(len(bin_val) / 2)]],
                y=[0],
                mode="lines",
                line={"dash": traces["line_dash"], "color": traces["line_color"]},
                marker={"opacity": 0},
                visible=True,
                name=traces["name"],
            )
            for traces in traces_scatter
        ]

        trace3 = dict(
            type="scatter",
            mode="lines",
            line={"color": "#42C4F7"},
            y=y_val[0],
            x=bin_val[1][: len(bin_val[1])],
            name="Rayleigh Fit",
        )
        layout = dict(
            height=350,
            plot_bgcolor=app_color["graph_bg"],
            paper_bgcolor=app_color["graph_bg"],
            font={"color": "#fff"},
            xaxis={
                "title": "Wind Speed (mph)",
                "showgrid": False,
                "showline": False,
                "fixedrange": True,
            },
            yaxis={
                "showgrid": False,
                "showline": False,
                "zeroline": False,
                "title": "Number of Samples",
                "fixedrange": True,
            },
            autosize=True,
            bargap=0.01,
            bargroupgap=0,
            hovermode="closest",
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "xanchor": "center",
                "y": 1,
                "x": 0.5,
            },
            shapes=[
                {
                    "xref": "x",
                    "yref": "y",
                    "y1": int(max(bin_val_max, y_val_max)) + 0.5,
                    "y0": 0,
                    "x0": avg_val,
                    "x1": avg_val,
                    "type": "line",
                    "line": {"dash": "dash", "color": "#2E5266", "width": 5},
                },
                {
                    "xref": "x",
                    "yref": "y",
                    "y1": int(max(bin_val_max, y_val_max)) + 0.5,
                    "y0": 0,
                    "x0": median_val,
                    "x1": median_val,
                    "type": "line",
                    "line": {"dash": "dot", "color": "#BD9391", "width": 5},
                },
            ],
        )
        return dict(data=[trace, scatter_data[0], scatter_data[1], trace3], layout=layout)   

def get_current_time():
    """ Helper function to get the current time in seconds. """

    now = dt.datetime.now()
    total_time = (now.hour * 3600) + (now.minute * 60) + (now.second)
    return total_time

# def wind_direction(app):
#     @app.callback(
#         Output("wind-direction", "figure"), [Input("wind-speed-update", "n_intervals")]
#     )
#     def gen_wind_direction(interval):
#         """
#         Generate the wind direction graph.

#         :params interval: update the graph based on an interval
#         """

#         total_time = get_current_time()
#         df = get_wind_data_by_id(total_time)
#         val = df["Speed"].iloc[-1]
#         direction = [0, (df["Direction"][0] - 20), (df["Direction"][0] + 20), 0]

#         traces_scatterpolar = [
#             {"r": [0, val, val, 0], "fillcolor": "#084E8A"},
#             {"r": [0, val * 0.65, val * 0.65, 0], "fillcolor": "#B4E1FA"},
#             {"r": [0, val * 0.3, val * 0.3, 0], "fillcolor": "#EBF5FA"},
#         ]

#         data = [
#             dict(
#                 type="scatterpolar",
#                 r=traces["r"],
#                 theta=direction,
#                 mode="lines",
#                 fill="toself",
#                 fillcolor=traces["fillcolor"],
#                 line={"color": "rgba(32, 32, 32, .6)", "width": 1},
#             )
#             for traces in traces_scatterpolar
#         ]

#         layout = dict(
#             height=350,
#             plot_bgcolor=app_color["graph_bg"],
#             paper_bgcolor=app_color["graph_bg"],
#             font={"color": "#fff"},
#             autosize=False,
#             polar={
#                 "bgcolor": app_color["graph_line"],
#                 "radialaxis": {"range": [0, 45], "angle": 45, "dtick": 10},
#                 "angularaxis": {"showline": False, "tickcolor": "white"},
#             },
#             showlegend=False,
#         )

#         return dict(data=data, layout=layout)


# def wind_speed(app):
#     @app.callback(
#         Output("wind-speed", "figure"), [Input("wind-speed-update", "n_intervals")]
#     )
#     def gen_wind_speed(interval):
#         """
#         Generate the wind speed graph.

#         :params interval: update the graph based on an interval
#         """

#         total_time = get_current_time()
#         df = get_wind_data(total_time - 200, total_time)

#         trace = dict(
#             type="scatter",
#             y=df["Speed"],
#             line={"color": "#42C4F7"},
#             hoverinfo="skip",
#             error_y={
#                 "type": "data",
#                 "array": df["SpeedError"],
#                 "thickness": 1.5,
#                 "width": 2,
#                 "color": "#B4E8FC",
#             },
#             mode="lines",
#         )

#         layout = dict(
#             plot_bgcolor=app_color["graph_bg"],
#             paper_bgcolor=app_color["graph_bg"],
#             font={"color": "#fff"},
#             height=700,
#             xaxis={
#                 "range": [0, 200],
#                 "showline": True,
#                 "zeroline": False,
#                 "fixedrange": True,
#                 "tickvals": [0, 50, 100, 150, 200],
#                 "ticktext": ["200", "150", "100", "50", "0"],
#                 "title": "Time Elapsed (sec)",
#             },
#             yaxis={
#                 "range": [
#                     min(0, min(df["Speed"])),
#                     max(45, max(df["Speed"]) + max(df["SpeedError"])),
#                 ],
#                 "showgrid": True,
#                 "showline": True,
#                 "fixedrange": True,
#                 "zeroline": False,
#                 "gridcolor": app_color["graph_line"],
#                 "nticks": max(6, round(df["Speed"].iloc[-1] / 10)),
#             },
#         )

#         return dict(data=[trace], layout=layout)